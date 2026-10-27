"""Critique: reviews the coordinator's draft for accuracy, clarity, and safety."""

import os

from google.adk.agents import LlmAgent

from .callbacks import log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

CRITIQUE_INSTRUCTIONS = """
You are an editor reviewing a draft reply from an emergency preparedness
assistant. The reader may be stressed and in danger.

Draft reply:
{draft_answer}

List specific, actionable improvements: is the most urgent information first?
Is anything unclear, unsupported, or missing (e.g. a safety step the reader
needs)? Is it short enough to read quickly? If the draft is already good, say
so and list only minor polish. Reply with the list only.
"""

critique_agent = LlmAgent(
    name="critique",
    model=build_model(AGENT_MODEL),
    description="Reviews the draft reply and lists improvements.",
    instruction=CRITIQUE_INSTRUCTIONS,
    output_key="critique_notes",
    after_model_callback=log_model_response,
)
