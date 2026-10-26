"""Search: answers the clarified question using Google Search."""

import os

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from .callbacks import log_model_response
from .models import build_model

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

SEARCH_INSTRUCTIONS = """
Research and answer this question using Google Search:

{question}

Write a complete first-draft answer with the key facts. Cite specifics where
you can. This draft will be reviewed and improved by another agent.
"""

# Kept on its own agent since google_search is a built-in grounding tool.
# Mixing it with custom function tools fails on the API call.
search_agent = LlmAgent(
    name="search",
    model=build_model(AGENT_MODEL),
    description="Produces a first-draft answer from Google Search results.",
    instruction=SEARCH_INSTRUCTIONS,
    tools=[google_search],
    output_key="draft_answer",
    after_model_callback=log_model_response,
)
