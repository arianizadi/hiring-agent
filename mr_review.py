"""Fetch and evaluate the actual code in open source contributions (PR/MR links)
that a candidate lists on their resume.

Resumes often link to pull/merge requests as evidence of open source work, but a
link proves nothing about the change. This module finds those links, fetches the
real diff from the GitHub API, has the LLM review each contribution for substance
and code quality, and verifies the candidate actually authored it — so the
Open Source score reflects real, candidate-authored, substantive work instead of
the mere presence of a link.

Currently supports GitHub pull requests. GitLab merge requests are detected but
not yet fetched (the API shape differs); they are reported so they aren't silently
ignored.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from models import MRReview
from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

logger = logging.getLogger(__name__)

GITHUB_PR_RE = re.compile(
    r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)", re.IGNORECASE
)
GITLAB_MR_RE = re.compile(
    r"https?://gitlab\.com/(\S+?)/-/merge_requests/(\d+)", re.IGNORECASE
)

# Caps to keep the diff sent to the LLM bounded.
MAX_FILES = 30
MAX_PATCH_CHARS = 8000


def parse_contribution_links(
    links: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Split resume hyperlinks into GitHub PRs (parsed) and GitLab MRs (urls only)."""
    github_prs: List[Dict[str, str]] = []
    gitlab_mrs: List[str] = []
    seen = set()
    for link in links or []:
        url = (link.get("url") or "").strip()
        m = GITHUB_PR_RE.search(url)
        if m:
            owner, repo, number = m.group(1), m.group(2), m.group(3)
            key = (owner.lower(), repo.lower(), number)
            if key in seen:
                continue
            seen.add(key)
            github_prs.append(
                {
                    "owner": owner,
                    "repo": repo,
                    "number": number,
                    "url": url,
                    "anchor": link.get("text", ""),
                }
            )
        elif GITLAB_MR_RE.search(url):
            gitlab_mrs.append(url)
    return github_prs, gitlab_mrs


def fetch_github_pr(owner: str, repo: str, number: str) -> Optional[Dict[str, Any]]:
    """Fetch PR metadata and changed files via the GitHub API (cached, rate-limited)."""
    # Imported lazily to reuse github.py's caching + rate-limit handling without
    # creating an import cycle at module load.
    from github import _fetch_github_api

    base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    status, meta = _fetch_github_api(base)
    if status != 200 or not isinstance(meta, dict) or not meta:
        logger.warning(
            f"Could not fetch PR {owner}/{repo}#{number} (status {status})"
        )
        return None

    status_files, files = _fetch_github_api(base + "/files")
    if status_files != 200 or not isinstance(files, list):
        files = []
    return {"meta": meta, "files": files}


def _build_patch_text(files: List[Dict[str, Any]]) -> str:
    """Concatenate per-file patches into a single bounded diff for the LLM."""
    parts: List[str] = []
    total = 0
    for f in files[:MAX_FILES]:
        header = (
            f"--- {f.get('filename')} "
            f"({f.get('status')}, +{f.get('additions', 0)}/-{f.get('deletions', 0)})"
        )
        patch = f.get("patch") or ""
        remaining = MAX_PATCH_CHARS - total
        if remaining <= 0:
            parts.append("... [diff truncated]")
            break
        if len(patch) > remaining:
            patch = patch[:remaining] + "\n... [file diff truncated]"
        parts.append(header + ("\n" + patch if patch else " [no textual diff]"))
        total += len(patch)
    if len(files) > MAX_FILES:
        parts.append(f"... [{len(files) - MAX_FILES} more file(s) omitted]")
    return "\n".join(parts)


def _review_pr_with_llm(
    provider, model, model_params, facts: str
) -> Optional[MRReview]:
    """Ask the LLM to judge the contribution's substance and quality."""
    system = (
        "You are a senior software engineer assessing an open source contribution "
        "that a job candidate listed on their resume. Judge the ACTUAL code change "
        "for substance and quality, not the existence of a link. Be skeptical of "
        "trivial changes (typo fixes, version bumps, whitespace, one-line tweaks) "
        "and of pull requests NOT authored by the candidate. Respond with JSON only."
    )
    user = (
        facts
        + '\n\nReturn a JSON object with exactly these keys: '
        '{"substance": "trivial|minor|moderate|significant", '
        '"quality": "poor|mixed|solid|excellent", '
        '"summary": "what the change actually does", '
        '"concerns": "red flags, or empty string"}'
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
                {"role": "user", "content": user},
            ],
            options=options,
            format=MRReview.model_json_schema(),
        )
        text = extract_json_from_response(resp["message"]["content"])
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return MRReview(**json.loads(text))
    except Exception as e:
        logger.warning(f"PR review LLM call failed: {e}")
        return None


def _summarize_pr(pr: Dict[str, Any], candidate_username: str) -> Dict[str, Any]:
    """Flatten the API payload into the fields we score and display."""
    meta = pr["meta"]
    base_repo = (meta.get("base") or {}).get("repo") or {}
    author = (meta.get("user") or {}).get("login") or ""
    authored: Optional[bool] = None
    if candidate_username and author:
        authored = author.lower() == candidate_username.lower()
    merged = bool(meta.get("merged"))
    return {
        "repo": base_repo.get("full_name"),
        "stars": base_repo.get("stargazers_count"),
        "number": meta.get("number"),
        "title": meta.get("title"),
        "state": "merged" if merged else (meta.get("state") or "unknown"),
        "merged": merged,
        "additions": meta.get("additions", 0),
        "deletions": meta.get("deletions", 0),
        "changed_files": meta.get("changed_files"),
        "author": author,
        "authored_by_candidate": authored,
    }


def review_resume_contributions(
    links: List[Dict[str, str]], candidate_username: str
) -> Dict[str, Any]:
    """Find PR/MR links in the resume, fetch the diffs, and review each one."""
    github_prs, gitlab_mrs = parse_contribution_links(links)
    if not github_prs and not gitlab_mrs:
        return {"github": [], "gitlab_unsupported": []}

    provider = initialize_llm_provider(DEFAULT_MODEL)
    model_params = MODEL_PARAMETERS.get(
        DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9}
    )

    results: List[Dict[str, Any]] = []
    for item in github_prs:
        pr = fetch_github_pr(item["owner"], item["repo"], item["number"])
        if not pr:
            results.append(
                {
                    "url": item["url"],
                    "repo": f"{item['owner']}/{item['repo']}",
                    "number": item["number"],
                    "error": "could not fetch from GitHub API",
                    "review": None,
                }
            )
            continue
        summary = _summarize_pr(pr, candidate_username)
        facts = (
            f"Repository: {summary['repo']}"
            + (f" ({summary['stars']} stars)" if summary["stars"] is not None else "")
            + f"\nPR #{summary['number']}: {summary['title']}\n"
            f"State: {summary['state']}; "
            f"+{summary['additions']}/-{summary['deletions']} across "
            f"{summary['changed_files']} file(s)\n"
            f"PR author: {summary['author'] or 'unknown'}; "
            f"candidate GitHub: {candidate_username or 'unknown'}; "
            f"authored_by_candidate: {summary['authored_by_candidate']}\n\n"
            f"Diff (truncated):\n{_build_patch_text(pr['files'])}"
        )
        review = _review_pr_with_llm(provider, DEFAULT_MODEL, model_params, facts)
        summary["url"] = item["url"]
        summary["review"] = review
        results.append(summary)

    return {"github": results, "gitlab_unsupported": gitlab_mrs}


def _authored_label(authored: Optional[bool]) -> str:
    if authored is True:
        return "yes"
    if authored is False:
        return "NO — not authored by the candidate"
    return "unknown"


def format_contributions_for_eval(data: Dict[str, Any]) -> str:
    """Render reviewed contributions as a ground-truth evidence block for the evaluator."""
    results = (data or {}).get("github", [])
    if not results:
        return ""
    lines = [
        "\n\n=== VERIFIED OPEN SOURCE CONTRIBUTIONS (reviewed from actual PR diffs) ===",
        "Each entry was fetched live from the GitHub API and its real diff was reviewed. "
        "Treat this as GROUND TRUTH for open source scoring: credit substantive, "
        "candidate-authored, merged contributions; give little or no credit for "
        "trivial changes or for any PR where authored_by_candidate is NO.",
    ]
    for r in results:
        if r.get("error"):
            lines.append(
                f"- {r.get('repo')}#{r.get('number')}: {r['error']} ({r.get('url')})"
            )
            continue
        stars = f" ({r['stars']}★)" if r.get("stars") is not None else ""
        line = (
            f"- {r['repo']}{stars} PR #{r['number']} \"{r['title']}\" — {r['state']}, "
            f"+{r['additions']}/-{r['deletions']} in {r['changed_files']} file(s); "
            f"authored_by_candidate: {_authored_label(r.get('authored_by_candidate'))}"
        )
        rv = r.get("review")
        if rv:
            line += f"\n    substance={rv.substance}, quality={rv.quality}; {rv.summary}"
            if rv.concerns:
                line += f"\n    concerns: {rv.concerns}"
        lines.append(line)
    if data.get("gitlab_unsupported"):
        lines.append(
            f"(Note: {len(data['gitlab_unsupported'])} GitLab merge-request link(s) "
            "detected but not fetched — diff review not yet supported for GitLab.)"
        )
    return "\n".join(lines)


def format_contributions_for_report(data: Dict[str, Any]) -> str:
    """Render reviewed contributions for the console report."""
    results = (data or {}).get("github", [])
    if not results:
        return ""
    out = [
        "\n🔎 OPEN SOURCE CONTRIBUTIONS REVIEWED (from actual PR diffs):",
        "-" * 60,
    ]
    for r in results:
        if r.get("error"):
            out.append(f"  • {r.get('repo')}#{r.get('number')}: {r['error']}")
            continue
        stars = f" ({r['stars']}★)" if r.get("stars") is not None else ""
        authored = r.get("authored_by_candidate")
        flag = "✅ candidate" if authored else ("⛔ NOT candidate" if authored is False else "❔ unknown")
        out.append(
            f"  • {r['repo']}{stars} PR #{r['number']} — {r['state']}, "
            f"+{r['additions']}/-{r['deletions']} in {r['changed_files']} file(s)"
        )
        out.append(f"      author: {r['author'] or 'unknown'} [{flag}]")
        rv = r.get("review")
        if rv:
            out.append(f"      substance: {rv.substance} | quality: {rv.quality}")
            out.append(f"      → {rv.summary}")
            if rv.concerns:
                out.append(f"      ⚠ {rv.concerns}")
    if data.get("gitlab_unsupported"):
        out.append(
            f"  (+{len(data['gitlab_unsupported'])} GitLab MR link(s) detected — not yet supported)"
        )
    return "\n".join(out)
