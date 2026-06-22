from typing import List, Optional, Dict, Tuple, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to the LLM provider."""
        ...


class Location(BaseModel):
    """Location information for JSON Resume format."""

    address: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None
    countryCode: Optional[str] = None
    region: Optional[str] = None


class Profile(BaseModel):
    """Social profile information for JSON Resume format."""

    network: Optional[str] = None
    username: Optional[str] = None
    url: str


class Basics(BaseModel):
    """Basic information for JSON Resume format."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = None


class Work(BaseModel):
    """Work experience for JSON Resume format."""

    name: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Volunteer(BaseModel):
    """Volunteer experience for JSON Resume format."""

    organization: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Education(BaseModel):
    """Education information for JSON Resume format."""

    institution: Optional[str] = None
    url: Optional[str] = None
    area: Optional[str] = None
    studyType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    score: Optional[str] = None
    courses: Optional[List[str]] = None


class Award(BaseModel):
    """Award information for JSON Resume format."""

    title: Optional[str] = None
    date: Optional[str] = None
    awarder: Optional[str] = None
    summary: Optional[str] = None


class Certificate(BaseModel):
    """Certificate information for JSON Resume format."""

    name: Optional[str] = None
    date: Optional[str] = None
    issuer: Optional[str] = None
    url: Optional[str] = None


class Publication(BaseModel):
    """Publication information for JSON Resume format."""

    name: Optional[str] = None
    publisher: Optional[str] = None
    releaseDate: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class Skill(BaseModel):
    """Skill information for JSON Resume format."""

    name: Optional[str] = None
    level: Optional[str] = None
    keywords: Optional[List[str]] = None


class Language(BaseModel):
    """Language information for JSON Resume format."""

    language: Optional[str] = None
    fluency: Optional[str] = None


class Interest(BaseModel):
    """Interest information for JSON Resume format."""

    name: Optional[str] = None
    keywords: Optional[List[str]] = None


class Reference(BaseModel):
    """Reference information for JSON Resume format."""

    name: Optional[str] = None
    reference: Optional[str] = None


class Project(BaseModel):
    """Project information for JSON Resume format."""

    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[List[str]] = None
    url: Optional[str] = None
    technologies: Optional[List[str]] = None
    skills: Optional[List[str]] = None


class BasicsSection(BaseModel):
    """Basics section containing basic information."""

    basics: Optional[Basics] = None


class WorkSection(BaseModel):
    """Work section containing a list of work experiences."""

    work: Optional[List[Work]] = None


class EducationSection(BaseModel):
    """Education section containing a list of education entries."""

    education: Optional[List[Education]] = None


class SkillsSection(BaseModel):
    """Skills section containing a list of skill categories."""

    skills: Optional[List[Skill]] = None


class ProjectsSection(BaseModel):
    """Projects section containing a list of projects."""

    projects: Optional[List[Project]] = None


class AwardsSection(BaseModel):
    """Awards section containing a list of awards."""

    awards: Optional[List[Award]] = None


class JSONResume(BaseModel):
    """Complete JSON Resume format model."""

    basics: Optional[Basics] = None
    work: Optional[List[Work]] = None
    volunteer: Optional[List[Volunteer]] = None
    education: Optional[List[Education]] = None
    awards: Optional[List[Award]] = None
    certificates: Optional[List[Certificate]] = None
    publications: Optional[List[Publication]] = None
    skills: Optional[List[Skill]] = None
    languages: Optional[List[Language]] = None
    interests: Optional[List[Interest]] = None
    references: Optional[List[Reference]] = None
    projects: Optional[List[Project]] = None


class CategoryScore(BaseModel):
    score: float = Field(ge=0, description="Score achieved in this category")
    max: int = Field(gt=0, description="Maximum possible score")
    evidence: str = Field(min_length=1, description="Evidence supporting the score")


class Scores(BaseModel):
    open_source: CategoryScore
    self_projects: CategoryScore
    production: CategoryScore
    technical_skills: CategoryScore


class BonusPoints(BaseModel):
    total: float = Field(ge=0, le=20, description="Total bonus points")
    breakdown: str = Field(description="Breakdown of bonus points")


class Deductions(BaseModel):
    total: float = Field(
        ge=0,
        description="Total deduction points (stored as positive, applied as negative)",
    )
    reasons: str = Field(description="Reasons for deductions")


class ImprovementTip(BaseModel):
    """An actionable way to raise the score, including points likely left on the table."""

    area: str = Field(
        description="What to improve or what was likely missed (e.g. 'Early-stage engineer bonus', 'Open Source evidence')"
    )
    suggestion: str = Field(
        description="Concrete, actionable advice the candidate can apply"
    )
    potential_gain: Optional[str] = Field(
        default=None,
        description="Rough points this could add, e.g. '+2-3 bonus' or '+5 open_source'",
    )


class EvaluationData(BaseModel):
    scores: Scores
    bonus_points: BonusPoints
    deductions: Deductions
    key_strengths: List[str] = Field(min_items=1, max_items=5)
    areas_for_improvement: List[str] = Field(min_items=1, max_items=5)
    score_improvement_tips: List[ImprovementTip] = Field(
        default_factory=list,
        max_items=8,
        description=(
            "Actionable tips to increase the score, including points likely left "
            "on the table that the resume did not make explicit — e.g. unclaimed "
            "bonuses such as early-stage engineer / first 10-20 employees, small "
            "team / startup founder or co-founder, GSoC, portfolio, or technical blogs."
        ),
    )


class MRReview(BaseModel):
    """LLM assessment of one open source contribution (PR/MR) fetched from its diff."""

    substance: str = Field(
        description="How meaningful the change is: trivial | minor | moderate | significant"
    )
    quality: str = Field(
        description="Code quality of the change: poor | mixed | solid | excellent"
    )
    summary: str = Field(
        min_length=1, description="What the change actually does, in 1-2 sentences"
    )
    concerns: Optional[str] = Field(
        default=None,
        description="Red flags (e.g. trivial change, not authored by the candidate), or empty",
    )


class ProjectCodeReview(BaseModel):
    """LLM assessment of a candidate's personal project, reviewed from its repo code."""

    complexity: str = Field(
        description="Technical complexity of the project: trivial | basic | moderate | advanced"
    )
    quality: str = Field(
        description="Code quality and craftsmanship: poor | mixed | solid | excellent"
    )
    summary: str = Field(
        min_length=1,
        description="What the project is and how it is built, in 1-3 sentences",
    )
    strengths: Optional[str] = Field(
        default=None, description="Notable engineering strengths, or empty"
    )
    concerns: Optional[str] = Field(
        default=None,
        description="Red flags (tutorial copy, boilerplate-only, empty/abandoned), or empty",
    )


class GitHubProfile(BaseModel):
    """Pydantic model for GitHub profile data."""

    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    avatar_url: Optional[str] = None
    blog: Optional[str] = None
    twitter_username: Optional[str] = None
    hireable: Optional[bool] = None


class OllamaProvider:
    """Ollama LLM provider implementation."""

    def __init__(self):
        import ollama

        self.client = ollama

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama."""

        ollama_options = options.copy() if options else {}

        # remove steam from ollama options
        ollama_options.pop("stream", None)

        # Add num_ctx 32K context window to options
        ollama_options["num_ctx"] = 32768

        # convert to chat params
        chat_params = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }

        # add it to top level
        if "stream" in kwargs:
            chat_params["stream"] = kwargs["stream"]

        if "format" in kwargs:
            chat_params["format"] = kwargs["format"]

        return self.client.chat(**chat_params)


class GeminiProvider:
    """Google Gemini API provider implementation."""

    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Google Gemini API."""
        # Map options to Gemini parameters
        generation_config = {}
        if options:
            if "temperature" in options:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options:
                generation_config["top_p"] = options["top_p"]

        # Create a Gemini model
        gemini_model = self.client.GenerativeModel(
            model_name=model, generation_config=generation_config
        )

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        # Send the chat request
        response = gemini_model.generate_content(gemini_messages)

        # Convert Gemini response to Ollama-like format for compatibility
        return {"message": {"role": "assistant", "content": response.text}}


class OpenAIProvider:
    """OpenAI API provider implementation."""

    # Whether to send OpenAI's response_format={"type": "json_object"} flag.
    # Native OpenAI supports it; aggregators routing to other vendors may not,
    # so subclasses can disable it and rely on prompt-driven JSON instead.
    supports_json_mode = True

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """True for OpenAI reasoning models (o-series, gpt-5 family)."""
        name = (model or "").lower().split("/")[-1]  # drop any 'openai/' prefix
        return name.startswith(("o1", "o3", "o4", "gpt-5"))

    def __init__(self, api_key: str, base_url: str = None):
        from openai import OpenAI

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to the OpenAI Chat Completions API.

        Returns an Ollama-like dict ({"message": {"role", "content"}}) so the
        rest of the pipeline stays provider-agnostic.
        """
        options = options or {}

        params: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
        }

        if self._is_reasoning_model(model):
            # Reasoning models (o-series, gpt-5*) reject custom temperature/top_p
            # and are tuned via reasoning_effort instead.
            effort = options.get("reasoning_effort")
            if effort:
                params["reasoning_effort"] = effort
        else:
            if options.get("temperature") is not None:
                params["temperature"] = options["temperature"]
            if options.get("top_p") is not None:
                params["top_p"] = options["top_p"]

        # Structured output. The downstream code validates against Pydantic
        # models itself, so JSON mode (guaranteed-valid JSON) is the robust
        # choice and avoids brittle strict-schema rejections. When json_object
        # mode is unsupported (e.g. aggregators routing to non-OpenAI vendors),
        # fall back to prompt-driven JSON, which the pipeline already parses.
        if kwargs.get("format") is not None:
            if self.supports_json_mode:
                params["response_format"] = {"type": "json_object"}
            # Ensure the literal word "json" is present (required by json mode,
            # and a reliable nudge for prompt-only providers).
            has_json = any(
                "json" in (m.get("content") or "").lower() for m in params["messages"]
            )
            if not has_json:
                params["messages"] = params["messages"] + [
                    {"role": "system", "content": "Respond with valid JSON only."}
                ]

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        return {"message": {"role": "assistant", "content": content}}


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — OpenAI-compatible API, single key for many models.

    Use any OpenRouter model id as DEFAULT_MODEL (e.g. ``openai/gpt-4o``,
    ``anthropic/claude-3.7-sonnet``, ``google/gemini-2.5-pro``,
    ``deepseek/deepseek-chat``).
    """

    # OpenRouter routes to many vendors; not all accept OpenAI's json_object
    # flag, so rely on prompt-driven JSON (the pipeline parses it either way).
    supports_json_mode = False

    def __init__(
        self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"
    ):
        super().__init__(api_key=api_key, base_url=base_url)
