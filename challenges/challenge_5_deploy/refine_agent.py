import os

from google.adk.agents import LlmAgent

from .callbacks import log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

REFINE_INSTRUCTIONS = """
Rewrite the draft answer so it addresses every point in the editor's notes.

Question:
{question}

Draft answer:
{draft_answer}

Editor's notes:
{critique_notes}

Reply with only the improved answer, written for the user. Do not mention
the draft, the notes, or the editing process.
"""

refine_agent = LlmAgent(
    name="refine",
    model=build_model(AGENT_MODEL),
    description="Produces the final answer by applying the critique to the draft.",
    instruction=REFINE_INSTRUCTIONS,
    output_key="final_answer",
    after_model_callback=log_model_response,
)
