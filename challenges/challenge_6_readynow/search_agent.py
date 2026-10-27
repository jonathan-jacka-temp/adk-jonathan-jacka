"""Search sub-agent: current news and alerts via Google Search."""

import os

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

SEARCH_AGENT_INSTRUCTIONS = """
You find current news, official alerts, and shelter or evacuation notices for
a US location or emergency. Search Google and return the key facts with
sources. Prefer official sources (FEMA, state/local emergency management,
NWS, Red Cross).
"""

# Kept on its own agent since google_search is a built-in grounding tool.
# Mixing it with custom function tools fails on the API call.
search_agent = LlmAgent(
    name="search_agent",
    model=build_model(AGENT_MODEL),
    description="Searches the web for current emergency news, alerts, and shelter information.",
    instruction=SEARCH_AGENT_INSTRUCTIONS,
    tools=[google_search],
    mode="single_turn",
)
