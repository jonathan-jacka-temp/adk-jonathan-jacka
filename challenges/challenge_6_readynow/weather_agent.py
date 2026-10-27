"""Weather sub-agent: current conditions and alerts from the National Weather Service."""

import os

from google.adk.agents import LlmAgent

from .models import build_model
from .tools import get_extended_weather_forecast, get_lat_lon

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

WEATHER_AGENT_INSTRUCTIONS = """
You report weather conditions for a US location during an emergency.
Use get_lat_lon to convert the place name into coordinates, then
get_extended_weather_forecast to fetch the forecast. Lead with any hazardous
conditions (storms, flooding, extreme heat or cold, high wind) and when they
are expected. Be brief and factual.
The National Weather Service API only covers the United States.
"""

weather_agent = LlmAgent(
    name="weather_agent",
    model=build_model(AGENT_MODEL),
    description="Fetches the US National Weather Service forecast and hazards for a place name.",
    instruction=WEATHER_AGENT_INSTRUCTIONS,
    tools=[get_extended_weather_forecast, get_lat_lon],
    mode="single_turn",
)
