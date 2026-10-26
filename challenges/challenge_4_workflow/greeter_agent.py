import os

from google.adk.agents import LlmAgent

from .callbacks import chained_before_callback, log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

GREETER_INSTRUCTIONS = """
You are the entry point for a research assistant. Restate the user's request
as one clear, self-contained question that a researcher could answer without
seeing the original message. Reply with only that question.
"""

# Only node to sees raw user input, so must have input validation here.
greeter_agent = LlmAgent(
    name="greeter",
    model=build_model(AGENT_MODEL),
    description="Clarifies the user's request into a single research question.",
    instruction=GREETER_INSTRUCTIONS,
    output_key="question",
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)
