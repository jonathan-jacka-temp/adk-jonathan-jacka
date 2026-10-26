import os

from google.adk.agents import Agent
from google.adk.tools import google_search

from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

SEARCH_AGENT_INSTRUCTIONS = """
You answer questions by searching Google. Return a concise, factual answer
with the key details the user asked for.
"""

# Kept this as its own agent since google_search is a built-in grounding tool.
# Mixing it with custom function tools fails on the API call.
search_agent = Agent(
    name="search_agent",
    model=build_model(AGENT_MODEL),
    description="Searches Google to answer general or current-events questions.",
    instruction=SEARCH_AGENT_INSTRUCTIONS,
    tools=[google_search],
    mode="single_turn",
)
