"""Challenge 1: Real-Time Weather Alerts Agent.

Draft/test locally with:

    uv run python -m challenges.challenge_1_weather_alerts.agent
"""

import os

from google.adk.agents import Agent

from .models import build_model
from .tools import get_extended_weather_forecast, get_lat_lon

# e.g. WEATHER_AGENT_MODEL=gpt-4o uv run python agent.py
WEATHER_AGENT_MODEL = os.environ.get("WEATHER_AGENT_MODEL", "gemini-3.5-flash")

WEATHER_AGENT_INSTRUCTIONS = """
You are a helpful weather alerts assistant for locations in the United States.
Use get_lat_lon to convert a user's place name into coordinates, then
get_extended_weather_forecast to fetch the forecast. Summarize conditions and
call out any severe weather (storms, extreme heat/cold, high wind) as an alert.
The National Weather Service API only covers the United States; if asked about
a non-US location, say so and decline.
"""

weather_agent = Agent(
    name="Fred",
    model=build_model(WEATHER_AGENT_MODEL),
    description="Fred the Friendly Weather Agent.",
    instruction=WEATHER_AGENT_INSTRUCTIONS,
    tools=[get_extended_weather_forecast, get_lat_lon],
)

# adk run / adk web load this module and look for root_agent.
root_agent = weather_agent


if __name__ == "__main__":
    test_cities = ["Washington, DC", "Chicago, IL", "Seattle, WA", "Miami, FL"]

    for city in test_cities:
        coords = get_lat_lon(city)
        if coords is None:
            print(f"{city}: geocoding FAILED")
            continue

        forecast = get_extended_weather_forecast(coords["lat"], coords["lon"])
        if forecast:
            period = forecast[0]
            print(
                f"{city}: ({coords['lat']:.4f}, {coords['lon']:.4f}) -> "
                f"{period['name']}: {period.get('detailedForecast', '')[:80]}"
            )
        else:
            print(f"{city}: geocoded OK, forecast fetch FAILED")
