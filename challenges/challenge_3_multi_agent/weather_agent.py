import os

from google.adk.agents import Agent

from .models import build_model
from .tools import get_extended_weather_forecast, get_lat_lon

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

WEATHER_AGENT_INSTRUCTIONS = """
You are a weather assistant for locations in the United States.
Use get_lat_lon to convert a place name into coordinates, then
get_extended_weather_forecast to fetch the forecast. Summarize conditions and
call out any severe weather (storms, extreme heat/cold, high wind) as an alert.
The National Weather Service API only covers the United States; if asked about
a non-US location, say so and decline.
"""

weather_agent = Agent(
    name="weather_agent",
    model=build_model(AGENT_MODEL),
    description="Fetches the US National Weather Service forecast for a place name.",
    instruction=WEATHER_AGENT_INSTRUCTIONS,
    tools=[get_extended_weather_forecast, get_lat_lon],
    mode="single_turn",
)
