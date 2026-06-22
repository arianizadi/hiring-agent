"""Review a candidate's personal projects from their actual repository code.

A resume can claim any project; what matters is the code. This module finds the
candidate's own repositories linked in the resume, pulls a strategic digest of
each via Gitingest (source files only, noise excluded, size-capped), has the LLM
review the real code for complexity and quality, and feeds that into the Self
Projects score — so credit reflects engineering substance, not a project name.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional

import requests

from models import ProjectCodeReview
from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

logger = logging.getLogger(__name__)

GITINGEST_BASE = "https://gitingest.com/api"

# Any github.com/{owner}/{repo} that is NOT a sub-resource (pull/issues/tree/...).
GITHUB_REPO_RE = re.compile(
    r"https?://github\.com/([^/\s]+)/([^/\s#?]+)/?$", re.IGNORECASE
)

# Strategic ingestion: skip binaries, lockfiles, vendored/build output, notebooks.
EXCLUDE_PATTERN = (
    "*.lock,*.png,*.jpg,*.jpeg,*.gif,*.svg,*.pdf,*.bin,*.so,*.o,*.a,*.zip,"
    "*.tar,*.gz,*.min.js,*.map,*.ipynb,node_modules/*,build/*,dist/*,out/*,"
    "target/*,third_party/*,vendor/*,external/*,.git/*"
)
MAX_FILE_SIZE_KB = 64
MAX_DIGEST_CHARS = 28000  # cap the code sent to the LLM
MAX_PROJECTS = 3  # bound cost/time


def parse_own_repo_links(
    links: List[Dict[str, str]], candidate_username: str
) -> List[Dict[str, str]]:
    """Find the candidate's own repositories among resume hyperlinks.

    Restricts to repos owned by the candidate (so we review their work, not
    arbitrary repos) and skips the profile README repo (owner == repo).
    """
    repos: List[Dict[str, str]] = []
    seen = set()
    cand = (candidate_username or "").lower()
    for link in links or []:
        url = (link.get("url") or "").strip()
        m = GITHUB_REPO_RE.match(url)
        if not m:
            continue
        owner, repo = m.group(1), m.group(2)
        if repo.lower() == ".git":
            continue
        if owner.lower() == repo.lower():  # profile repo, low signal
            continue
        if cand and owner.lower() != cand:  # only the candidate's own repos
            continue
        key = (owner.lower(), repo.lower())
        if key in seen:
            continue
        seen.add(key)
        repos.append({"owner": owner, "repo": repo, "url": url})
    return repos[:MAX_PROJECTS]


def ingest_repo(owner: str, repo: str) -> Optional[Dict[str, Any]]:
    """Pull a strategic code digest for a repo via Gitingest."""
    try:
        resp = requests.get(
            f"{GITINGEST_BASE}/{owner}/{repo}",
            params={
                "max_file_size": MAX_FILE_SIZE_KB,
                "pattern_type": "exclude",
                "pattern": EXCLUDE_PATTERN,
            },
            timeout=90,
        )
        if resp.status_code != 200:
            logger.warning(
                f"Gitingest failed for {owner}/{repo} (status {resp.status_code})"
            )
            return None
        data = resp.json()
    except Exception as e:
        logger.warning(f"Gitingest error for {owner}/{repo}: {e}")
        return None

    content = data.get("content") or ""
    truncated = len(content) > MAX_DIGEST_CHARS
    return {
        "short_repo_url": data.get("short_repo_url") or f"{owner}/{repo}",
        "summary": data.get("summary") or "",
        "tree": data.get("tree") or "",
        "content": content[:MAX_DIGEST_CHARS],
        "truncated": truncated,
    }


def _review_project_with_llm(
    provider, model, model_params, repo_name: str, digest: Dict[str, Any]
) -> Optional[ProjectCodeReview]:
    """Ask the LLM to judge the project's complexity and code quality."""
    system = (
        "You are a senior software engineer reviewing a candidate's personal "
        "project from its ACTUAL repository code. Judge real engineering "
        "complexity and code quality. Be skeptical of tutorial copies, "
        "boilerplate-only scaffolds, and empty or abandoned repos. Credit "
        "non-trivial architecture, correctness, and craftsmanship. JSON only."
    )
    body = (
        f"Project: {repo_name}\n\n"
        f"{digest.get('summary')}\n\n"
        f"File tree:\n{digest.get('tree')}\n\n"
        f"Source digest"
        + (" (truncated)" if digest.get("truncated") else "")
        + f":\n{digest.get('content')}\n\n"
        'Return a JSON object with exactly these keys: '
        '{"complexity": "trivial|basic|moderate|advanced", '
        '"quality": "poor|mixed|solid|excellent", '
        '"summary": "what it is and how it is built", '
        '"strengths": "notable strengths or empty", '
        '"concerns": "red flags or empty"}'
    )
    options = {
        "temperature": model_params.get("temperature", 0.1),
        "top_p": model_params.get("top_p", 0.9),
        "reasoning_effort": model_params.get("reasoning_effort"),
    }
    try:
        resp = provider.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": body},
            ],
            options=options,
            format=ProjectCodeReview.model_json_schema(),
        )
        text = extract_json_from_response(resp["message"]["content"])
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return ProjectCodeReview(**json.loads(text))
    except Exception as e:
        logger.warning(f"Project review LLM call failed for {repo_name}: {e}")
        return None


def review_personal_projects(
    links: List[Dict[str, str]], candidate_username: str
) -> Dict[str, Any]:
    """Find the candidate's own repos in the resume and review each one's code."""
    repos = parse_own_repo_links(links, candidate_username)
    if not repos:
        return {"projects": []}

    provider = initialize_llm_provider(DEFAULT_MODEL)
    model_params = MODEL_PARAMETERS.get(
        DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9}
    )

    results: List[Dict[str, Any]] = []
    for item in repos:
        repo_name = f"{item['owner']}/{item['repo']}"
        digest = ingest_repo(item["owner"], item["repo"])
        if not digest:
            results.append(
                {"repo": repo_name, "url": item["url"], "error": "could not ingest", "review": None}
            )
            continue
        review = _review_project_with_llm(
            provider, DEFAULT_MODEL, model_params, repo_name, digest
        )
        results.append(
            {
                "repo": repo_name,
                "url": item["url"],
                "files_summary": (digest.get("summary") or "").splitlines()[:3],
                "review": review,
            }
        )
    return {"projects": results}


def format_projects_for_eval(data: Dict[str, Any]) -> str:
    """Render reviewed projects as a ground-truth evidence block for the evaluator."""
    results = (data or {}).get("projects", [])
    if not results:
        return ""
    lines = [
        "\n\n=== VERIFIED PERSONAL PROJECTS (reviewed from repository code) ===",
        "Each project below was reviewed from its ACTUAL source code (pulled via "
        "Gitingest). Treat this as GROUND TRUTH for self_projects scoring: credit "
        "advanced, well-built projects; give little credit to projects rated "
        "trivial/basic or flagged as tutorial copies or empty scaffolds.",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"- {r['repo']}: {r['error']} ({r.get('url')})")
            continue
        rv = r.get("review")
        if rv:
            line = (
                f"- {r['repo']}: complexity={rv.complexity}, quality={rv.quality}; "
                f"{rv.summary}"
            )
            if rv.strengths:
                line += f"\n    strengths: {rv.strengths}"
            if rv.concerns:
                line += f"\n    concerns: {rv.concerns}"
        else:
            line = f"- {r['repo']}: (code fetched but review unavailable)"
        lines.append(line)
    return "\n".join(lines)


def format_projects_for_report(data: Dict[str, Any]) -> str:
    """Render reviewed projects for the console report."""
    results = (data or {}).get("projects", [])
    if not results:
        return ""
    out = ["\n🧱 PERSONAL PROJECTS REVIEWED (from repository code):", "-" * 60]
    for r in results:
        if r.get("error"):
            out.append(f"  • {r['repo']}: {r['error']}")
            continue
        rv = r.get("review")
        if rv:
            out.append(f"  • {r['repo']}")
            out.append(f"      complexity: {rv.complexity} | quality: {rv.quality}")
            out.append(f"      → {rv.summary}")
            if rv.strengths:
                out.append(f"      + {rv.strengths}")
            if rv.concerns:
                out.append(f"      ⚠ {rv.concerns}")
        else:
            out.append(f"  • {r['repo']}: code fetched but review unavailable")
    return "\n".join(out)
