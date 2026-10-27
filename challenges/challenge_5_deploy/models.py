"""Multi-provider model factory for ADK agents."""

import os
import re

from anthropic import AsyncAnthropic
from google.adk.integrations.openai import OpenAILlm
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.lite_llm import LiteLlm

_OPENAI_MODEL_PATTERN = re.compile(r"gpt-.*|o\d+-.*")


def _build_openai_model(name: str) -> OpenAILlm:
    # base_url/api_key default to None here, which makes the OpenAI SDK fall
    # back to its own env vars (OPENAI_BASE_URL, OPENAI_API_KEY) if unset.
    return OpenAILlm(
        model=name,
        base_url=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def _build_anthropic_model(name: str) -> AnthropicLlm:
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not base_url:
        # No custom endpoint: let the Anthropic SDK read ANTHROPIC_API_KEY itself.
        return AnthropicLlm(model=name)

    # AnthropicLlm has no base_url field; routing through a non-default
    # endpoint means building the client ourselves.
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), base_url=base_url)
    return AnthropicLlm(model=name, client=client)


def build_model(name: str):
    """Build an ADK-compatible model object from a short provider/model name.

    Uses ADK's native OpenAI/Anthropic integrations where available, falling
    back to LiteLLM only for providers ADK doesn't support directly (e.g.
    "xai/grok-4", "groq/llama3-70b").

    Args:
        name: e.g. "gemini-3.5-flash", "gpt-4o", "claude-sonnet-4-5",
            "xai/grok-4".

    Returns:
        A string (for Gemini) or an ADK model instance, suitable for
        Agent(model=...).
    """
    if name.startswith("gemini-"):
        return name
    if _OPENAI_MODEL_PATTERN.fullmatch(name):
        return _build_openai_model(name)
    if name.startswith("claude-"):
        return _build_anthropic_model(name)
    return LiteLlm(model=name)
