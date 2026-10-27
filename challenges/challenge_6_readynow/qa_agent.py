"""QA sub-agent: general emergency-preparedness and safety guidance."""

import os

from google.adk.agents import LlmAgent

from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

QA_AGENT_INSTRUCTIONS = """
You answer general emergency-preparedness and safety questions: what to pack,
how to prepare for a specific hazard, what to do during and after an event,
how to help others safely. Give clear, numbered steps where it helps. Follow
FEMA and Red Cross guidance. If a question needs live data (current weather,
news, routes), say that another part of the assistant handles it.
"""

qa_agent = LlmAgent(
    name="qa_agent",
    model=build_model(AGENT_MODEL),
    description="Answers general emergency-preparedness and safety questions.",
    instruction=QA_AGENT_INSTRUCTIONS,
    mode="single_turn",
)
