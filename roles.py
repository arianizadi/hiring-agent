"""Role-aware scoring profiles.

The base rubric is weighted for a software-engineering candidate (open source 35,
self projects 30, production 25, technical 10). That under-serves other roles — a
data candidate's value is pipelines, modeling, and data work, not open source.

The caller passes one of a small set of premade role choices; each profile reweights
the four scoring categories (the weights always sum to 100) and gives the evaluator
role-specific guidance on what to value. There is no auto-detection — the role is an
explicit choice. Unknown/empty roles fall back to the software-engineering default,
which reproduces the original behavior exactly.
"""

from typing import Dict, List, Optional

DEFAULT_ROLE = "software-engineer"

ROLE_PROFILES: Dict[str, Dict] = {
    "software-engineer": {
        "label": "Software Engineer",
        "focus": "engineering depth across systems/projects, code quality, and real-world impact",
        "weights": {
            "open_source": 35,
            "self_projects": 30,
            "production": 25,
            "technical_skills": 10,
        },
        "guidance": {
            "open_source": "Credit substantive, candidate-authored contributions to external projects (merged PRs to popular repos, GSoC).",
            "self_projects": "Reward complex, well-engineered projects with real architecture; discount tutorial copies.",
            "production": "Internship/production engineering, ownership, shipped features, measurable impact.",
            "technical_skills": "Breadth and depth across languages, systems, tooling, and problem solving.",
        },
    },
    "data-science": {
        "label": "Data Scientist / Data Engineer",
        "focus": "data pipelines, modeling and experimentation, and measurable data/ML impact",
        "weights": {
            "production": 35,
            "technical_skills": 25,
            "self_projects": 25,
            "open_source": 15,
        },
        "guidance": {
            "production": "Weight production data and ML work: ETL/ELT pipelines, orchestration (Airflow/Dagster), warehouses (Snowflake/BigQuery/Redshift), streaming (Kafka/Spark), data modeling, data quality/validation, models/experiments shipped to production, A/B tests, and scale/reliability (volume, throughput, SLAs).",
            "technical_skills": "SQL depth, Python/Pandas/NumPy, Spark, dbt, the ML stack (PyTorch/TensorFlow/scikit-learn), statistics, cloud data services, warehouses, and BI/visualization. Algorithms/systems-design depth matters less here.",
            "self_projects": "Credit real data/ETL/ML/analytics projects even without a hosted demo — a notebook, pipeline, or dashboard is fine. Reward rigor (clean data work, evaluation, reproducibility); discount copy-paste notebooks.",
            "open_source": "Contributions to data/ML tooling are a plus but NOT central — do not heavily penalize their absence; this category is intentionally low-weight for this role.",
        },
    },
}

# Forgiving aliases so casual inputs map to a premade choice.
ROLE_ALIASES: Dict[str, str] = {
    "swe": "software-engineer",
    "sde": "software-engineer",
    "software": "software-engineer",
    "software-engineer": "software-engineer",
    "software engineer": "software-engineer",
    "backend": "software-engineer",
    "ds": "data-science",
    "data": "data-science",
    "data-science": "data-science",
    "data science": "data-science",
    "data-scientist": "data-science",
    "data scientist": "data-science",
    "data-engineer": "data-science",
    "data engineer": "data-science",
    "data-engineering": "data-science",
    "ml": "data-science",
    "mle": "data-science",
}

# Fail fast if a profile is misconfigured.
for _key, _profile in ROLE_PROFILES.items():
    assert (
        sum(_profile["weights"].values()) == 100
    ), f"Role '{_key}' weights must sum to 100, got {sum(_profile['weights'].values())}"


def list_roles() -> List[str]:
    """Canonical premade role choices the user can pass."""
    return list(ROLE_PROFILES.keys())


def resolve_role_key(role: Optional[str]) -> Optional[str]:
    """Map a user-supplied role string to a canonical key, or None if unrecognized."""
    if not role:
        return None
    return ROLE_ALIASES.get(role.strip().lower())


def get_role_profile(role: Optional[str]) -> Dict:
    """Return the profile for a role, falling back to the software-engineering default."""
    key = resolve_role_key(role) or DEFAULT_ROLE
    return ROLE_PROFILES[key]
