import os
import sys
import json
import logging
import csv
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume, EvaluationData
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import (
    transform_evaluation_response,
    convert_json_resume_to_text,
    convert_github_data_to_text,
    convert_blog_data_to_text,
)
from config import DEVELOPMENT_MODE
from roles import get_role_profile, resolve_role_key, list_roles
from mr_review import (
    review_resume_contributions,
    format_contributions_for_eval,
    format_contributions_for_report,
)
from project_review import (
    review_personal_projects,
    format_projects_for_eval,
    format_projects_for_report,
)

logger = logging.getLogger(__name__)


def extract_pdf_hyperlinks(pdf_path: str) -> List[Dict[str, str]]:
    """Extract embedded hyperlinks (anchor text + URL) from a resume PDF.

    The per-section LLM extraction keeps anchor text but discards the underlying
    URI, so contribution links (e.g. PR/MR links to external repos) are lost.
    This recovers them so the evaluator can use them as evidence.
    """
    import fitz

    links: List[Dict[str, str]] = []
    seen = set()
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri")
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                try:
                    anchor = page.get_textbox(link["from"]).replace("\n", " ").strip()
                except Exception:
                    anchor = ""
                links.append({"text": anchor, "url": uri})
        doc.close()
    except Exception as e:
        logger.warning(f"Could not extract PDF hyperlinks: {e}")
    return links


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation: EvaluationData,
    candidate_name: str = "Candidate",
    role_profile: dict = None,
):
    """Print evaluation results in a readable format."""
    role_profile = role_profile or {}
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS FOR: {candidate_name}")
    if role_profile.get("label"):
        print(f"🎯 Scored as: {role_profile['label']}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available")
        return

    # Calculate overall score
    total_score = 0
    max_score = 0

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]

            # Log warning if score was capped
            if category_score < category_data["score"]:
                print(
                    f"⚠️  Warning: {category_name} score capped from {category_data['score']} to {category_score} (max: {category_data['max']})"
                )

    # Add bonus points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    # Subtract deductions
    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    # Ensure total score doesn't exceed maximum possible score
    max_possible_score = max_score + 20  # 120 (100 categories + 20 bonus)
    if total_score > max_possible_score:
        total_score = max_possible_score
        print(f"⚠️  Warning: Total score capped at maximum possible value")

    # Overall Score
    print(f"\n🎯 OVERALL SCORE: {total_score:.1f}/{max_score}")

    # Detailed Scores
    print("\n📈 DETAILED SCORES:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        # Category maximums for this role (fall back to the SWE defaults)
        category_maxes = role_profile.get(
            "weights",
            {
                "open_source": 35,
                "self_projects": 30,
                "production": 25,
                "technical_skills": 10,
            },
        )

        # Open Source
        if hasattr(evaluation.scores, "open_source") and evaluation.scores.open_source:
            os_score = evaluation.scores.open_source
            capped_score = min(os_score.score, category_maxes["open_source"])
            print(f"🌐 Open Source:          {capped_score}/{os_score.max}")
            print(f"   Evidence: {os_score.evidence}")
            print()

        # Self Projects
        if (
            hasattr(evaluation.scores, "self_projects")
            and evaluation.scores.self_projects
        ):
            sp_score = evaluation.scores.self_projects
            capped_score = min(sp_score.score, category_maxes["self_projects"])
            print(f"🚀 Self Projects:        {capped_score}/{sp_score.max}")
            print(f"   Evidence: {sp_score.evidence}")
            print()

        # Production Experience
        if hasattr(evaluation.scores, "production") and evaluation.scores.production:
            prod_score = evaluation.scores.production
            capped_score = min(prod_score.score, category_maxes["production"])
            print(f"🏢 Production Experience: {capped_score}/{prod_score.max}")
            print(f"   Evidence: {prod_score.evidence}")
            print()

        # Technical Skills
        if (
            hasattr(evaluation.scores, "technical_skills")
            and evaluation.scores.technical_skills
        ):
            tech_score = evaluation.scores.technical_skills
            capped_score = min(tech_score.score, category_maxes["technical_skills"])
            print(f"💻 Technical Skills:     {capped_score}/{tech_score.max}")
            print(f"   Evidence: {tech_score.evidence}")
            print()

    # Bonus Points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    # Deductions
    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        print(f"\n⚠️  DEDUCTIONS: -{evaluation.deductions.total}")
        print("-" * 30)
        if evaluation.deductions.reasons:
            print(f"   {evaluation.deductions.reasons}")

    # Key Strengths
    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    # Areas for Improvement
    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    # Score Improvement Tips (actionable + points likely left on the table)
    if (
        hasattr(evaluation, "score_improvement_tips")
        and evaluation.score_improvement_tips
    ):
        print(f"\n💡 HOW TO INCREASE YOUR SCORE:")
        print("-" * 30)
        for i, tip in enumerate(evaluation.score_improvement_tips, 1):
            gain = f"  [{tip.potential_gain}]" if tip.potential_gain else ""
            print(f"  {i}. {tip.area}{gain}")
            print(f"     → {tip.suggestion}")

    print("\n" + "=" * 80)


def _format_embedded_links(pdf_links: List[Dict[str, str]]) -> str:
    """Render embedded PDF hyperlinks as an evidence block for the evaluator."""
    if not pdf_links:
        return ""
    lines = [
        "\n\n=== EMBEDDED RESUME LINKS (extracted from PDF hyperlinks) ===",
        "These hyperlinks were embedded in the candidate's resume. Treat links to "
        "pull/merge requests in EXTERNAL repositories (i.e. not owned by the "
        "candidate) as direct evidence of open source contributions to those "
        "projects. Anchor text -> URL:",
    ]
    for link in pdf_links:
        text = link.get("text") or ""
        url = link.get("url") or ""
        lines.append(f"- {text} -> {url}")
    return "\n".join(lines)


def _evaluate_resume(
    resume_data: JSONResume,
    github_data: dict = None,
    blog_data: dict = None,
    pdf_links: List[Dict[str, str]] = None,
    contributions_text: str = "",
    projects_text: str = "",
    role_profile: dict = None,
) -> Optional[EvaluationData]:
    """Evaluate the resume using AI and display results."""

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(
        model_name=DEFAULT_MODEL,
        model_params=model_params,
        role_profile=role_profile,
    )

    # Convert JSON resume data to text
    resume_text = convert_json_resume_to_text(resume_data)

    # Add embedded PDF hyperlinks (recovers PR/MR links dropped during extraction)
    resume_text += _format_embedded_links(pdf_links)

    # Add verified open source contributions reviewed from actual PR diffs
    if contributions_text:
        resume_text += contributions_text

    # Add verified personal projects reviewed from actual repository code
    if projects_text:
        resume_text += projects_text

    # Add GitHub data if available
    if github_data:
        github_text = convert_github_data_to_text(github_data)
        resume_text += github_text

    # Add blog data if available
    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    # Evaluate the enhanced resume
    evaluation_result = evaluator.evaluate_resume(resume_text)

    # print(evaluation_result)

    return evaluation_result


def is_valid_resume_data(resume_data: JSONResume) -> bool:
    """Check if the resume data has at least some extracted core content."""
    if not resume_data:
        return False
    core_sections = [
        resume_data.basics,
        resume_data.work,
        resume_data.education,
        resume_data.skills,
        resume_data.projects,
    ]
    return any(section is not None for section in core_sections)


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def main(pdf_path, role=None):
    # Create cache filename based on PDF path
    cache_filename = (
        f"cache/resumecache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )
    github_cache_filename = (
        f"cache/githubcache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )

    resume_data = None
    cache_loaded = False

    # Check if cache exists and we're in development mode
    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}")
        try:
            cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
            loaded_resume = JSONResume(**cached_data)
            if not is_valid_resume_data(loaded_resume):
                raise ValueError("Cached resume data contains no core content")
            resume_data = loaded_resume
            cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid cache file {cache_filename}: {e}")
            print("Ignoring cache and reprocessing PDF...")
            try:
                os.remove(cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid cache file {cache_filename}: {delete_err}"
                )

    if not cache_loaded:
        logger.debug(
            f"Extracting data from PDF"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)

        if resume_data == None:
            return None

        if DEVELOPMENT_MODE:
            if is_valid_resume_data(resume_data):
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                Path(cache_filename).write_text(
                    json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                logger.warning(
                    "Newly extracted resume data is empty/invalid. Skipping cache write."
                )

    # Check if cache exists and we're in development mode
    github_data = {}
    github_cache_loaded = False
    if DEVELOPMENT_MODE and os.path.exists(github_cache_filename):
        print(f"Loading cached data from {github_cache_filename}")
        try:
            loaded_github = json.loads(
                Path(github_cache_filename).read_text(encoding="utf-8")
            )
            if (
                not isinstance(loaded_github, dict)
                or not loaded_github
                or "profile" not in loaded_github
            ):
                raise ValueError("Cached GitHub data is invalid or empty")
            github_data = loaded_github
            github_cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid GitHub cache file {github_cache_filename}: {e}")
            print("Ignoring GitHub cache and refetching...")
            try:
                os.remove(github_cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid GitHub cache file {github_cache_filename}: {delete_err}"
                )

    if not github_cache_loaded:
        # Add validation to handle None values
        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            print(
                f"Fetching GitHub data"
                + (
                    " and caching to " + github_cache_filename
                    if DEVELOPMENT_MODE
                    else ""
                )
            )
            github_data = fetch_and_display_github_info(github_profile.url)

            if (
                DEVELOPMENT_MODE
                and github_data
                and isinstance(github_data, dict)
                and "profile" in github_data
            ):
                os.makedirs(os.path.dirname(github_cache_filename), exist_ok=True)
                Path(github_cache_filename).write_text(
                    json.dumps(github_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    # Recover embedded hyperlinks (PR/MR links) that section extraction drops
    pdf_links = extract_pdf_hyperlinks(pdf_path)
    if pdf_links:
        print(f"🔗 Recovered {len(pdf_links)} embedded link(s) from the PDF")

    # Determine the candidate's GitHub username (for PR authorship verification)
    candidate_username = ""
    if isinstance(github_data, dict):
        candidate_username = (github_data.get("profile") or {}).get("username") or ""
    if not candidate_username and resume_data and resume_data.basics:
        gh_profile = find_profile(resume_data.basics.profiles, "Github")
        if gh_profile:
            from github import extract_github_username

            candidate_username = extract_github_username(gh_profile.url) or ""

    # Fetch and review the actual code of any PR/MR links found in the resume
    contributions = review_resume_contributions(pdf_links, candidate_username)
    contributions_report = format_contributions_for_report(contributions)
    if contributions_report:
        print(contributions_report)

    # Review the candidate's own project repos from their actual code (Gitingest)
    projects = review_personal_projects(pdf_links, candidate_username)
    projects_report = format_projects_for_report(projects)
    if projects_report:
        print(projects_report)

    # Score against the role the caller chose (no auto-detection). Unknown/None
    # falls back to the software-engineering default profile.
    role_profile = get_role_profile(role)
    print(f"🎯 Target role: {role_profile['label']}")

    score = _evaluate_resume(
        resume_data,
        github_data,
        pdf_links=pdf_links,
        contributions_text=format_contributions_for_eval(contributions),
        projects_text=format_projects_for_eval(projects),
        role_profile=role_profile,
    )

    # Get candidate name for display
    candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    # Print evaluation results in readable format
    print_evaluation_results(score, candidate_name, role_profile=role_profile)

    if DEVELOPMENT_MODE:
        csv_row = transform_evaluation_response(
            file_name=os.path.basename(pdf_path),
            evaluation=score,
            resume_data=resume_data,
            github_data=github_data,
        )

        # Write CSV row to file
        csv_path = "resume_evaluations.csv"
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the row
            writer.writerow(csv_row)

    return score


def _usage():
    print("Usage: python score.py <pdf_path> --role <role>")
    print(f"       roles: {', '.join(list_roles())}")


if __name__ == "__main__":
    # Minimal arg parsing: positional pdf_path plus required --role <value>.
    role = None
    positional = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--role", "-r"):
            if i + 1 >= len(args):
                print("Error: --role requires a value")
                exit(1)
            role = args[i + 1]
            i += 2
        elif arg.startswith("--role="):
            role = arg.split("=", 1)[1]
            i += 1
        else:
            positional.append(arg)
            i += 1

    if not positional:
        _usage()
        exit(1)
    pdf_path = positional[0]

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        exit(1)

    if not role:
        print("Error: --role is required (the user must choose the target role).")
        _usage()
        exit(1)

    role_key = resolve_role_key(role)
    if not role_key:
        print(f"Error: unknown role '{role}'. Choose one of: {', '.join(list_roles())}")
        exit(1)

    main(pdf_path, role=role_key)
