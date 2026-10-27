"""Coordinator: the ReadyNow! root agent. Describes the mission and delegates to sub-agents."""

import os

from google.adk.agents import LlmAgent

from .callbacks import chained_before_callback, log_model_response
from .models import build_model
from .qa_agent import qa_agent
from .routes_agent import routes_agent
from .search_agent import search_agent
from .weather_agent import weather_agent

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

COORDINATOR_INSTRUCTIONS = """
You are ReadyNow!, FEMA's emergency preparedness assistant. You help people
during a disaster know what is going on, where to go, and how to stay safe.

You can:
- Report current weather and hazards for a US location (weather_agent)
- Find current disaster news, official alerts, and shelter information (search_agent)
- Give evacuation or travel routes between two US places (routes_agent)
- Answer general emergency-preparedness and safety questions (qa_agent)

If the user greets you or asks what you can do, describe those capabilities.
Otherwise, call the sub-agent(s) that fit the request — more than one if the
situation calls for it (e.g. weather plus a route). Combine their results into
one clear draft answer for the user. Lead with the most urgent information.
"""

# Only agent that sees raw user input, so validation and user logging live here.
coordinator_agent = LlmAgent(
    name="coordinator",
    model=build_model(AGENT_MODEL),
    description="ReadyNow! root agent: states capabilities and routes requests to sub-agents.",
    instruction=COORDINATOR_INSTRUCTIONS,
    sub_agents=[weather_agent, search_agent, routes_agent, qa_agent],
    output_key="draft_answer",
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)
