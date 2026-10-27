"""Refine: rewrites the draft using the critique. Last node, so its reply reaches the user."""

import os

from google.adk.agents import LlmAgent

from .callbacks import log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

REFINE_INSTRUCTIONS = """
Rewrite the draft reply so it addresses every point in the editor's notes.

Draft reply:
{draft_answer}

Editor's notes:
{critique_notes}

Write for someone who may be stressed and short on time: most urgent
information first, plain language, short paragraphs or numbered steps. Reply
with only the final message to the user. Do not mention the draft, the notes,
or the editing process.
"""

refine_agent = LlmAgent(
    name="refine",
    model=build_model(AGENT_MODEL),
    description="Produces the final user-facing reply by applying the critique to the draft.",
    instruction=REFINE_INSTRUCTIONS,
    output_key="final_answer",
    after_model_callback=log_model_response,
)
