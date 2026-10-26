import os

from google.adk.agents import LlmAgent

from .callbacks import log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

CRITIQUE_INSTRUCTIONS = """
You are an editor reviewing a draft answer.

Question:
{question}

Draft answer:
{draft_answer}

List specific, actionable improvements: missing facts, unclear wording,
unsupported claims, structure. If the draft already answers the question
well, say so and list only minor polish. Reply with the list only.
"""

critique_agent = LlmAgent(
    name="critique",
    model=build_model(AGENT_MODEL),
    description="Reviews the draft answer and suggests improvements.",
    instruction=CRITIQUE_INSTRUCTIONS,
    output_key="critique_notes",
    after_model_callback=log_model_response,
)
