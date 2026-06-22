"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Dict, Optional
from models import (
    ModelProvider,
    OllamaProvider,
    GeminiProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from prompt import (
    MODEL_PROVIDER_MAPPING,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    PROVIDER,
)

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Text with markdown code block syntax removed
    """

    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8 :]

    # Remove leading ```json if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    # Remove trailing ``` if present
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name.

    Args:
        model_name: The name of the model to use

    Returns:
        An initialized LLM provider (either OllamaProvider or GeminiProvider)
    """
    # Resolve provider: explicit per-model mapping wins; otherwise fall back to
    # the provider configured via the LLM_PROVIDER env var (so custom model
    # names route correctly), and finally to Ollama.
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name)
    if model_provider is None:
        try:
            model_provider = ModelProvider(PROVIDER)
        except ValueError:
            model_provider = ModelProvider.OLLAMA

    if model_provider == ModelProvider.OPENAI:
        if not OPENAI_API_KEY:
            logger.warning("⚠️ OpenAI API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using OpenAI API provider with model {model_name}")
        return OpenAIProvider(api_key=OPENAI_API_KEY)

    if model_provider == ModelProvider.OPENROUTER:
        if not OPENROUTER_API_KEY:
            logger.warning("⚠️ OpenRouter API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using OpenRouter API provider with model {model_name}")
        return OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
        return GeminiProvider(api_key=GEMINI_API_KEY)

    logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return OllamaProvider()
