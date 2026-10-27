"""Routes sub-agent: driving directions to safety via OSRM."""

import os

from google.adk.agents import LlmAgent

from .models import build_model
from .tools import get_route

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

ROUTES_AGENT_INSTRUCTIONS = """
You provide evacuation and travel routes between two US places.
Call get_route with the origin and destination. Report total distance and
drive time first, then the key turns. If no route is found, say so plainly
and suggest the user confirm both place names.
"""

routes_agent = LlmAgent(
    name="routes_agent",
    model=build_model(AGENT_MODEL),
    description="Gives driving directions, distance, and time between two US places.",
    instruction=ROUTES_AGENT_INSTRUCTIONS,
    tools=[get_route],
    mode="single_turn",
)
