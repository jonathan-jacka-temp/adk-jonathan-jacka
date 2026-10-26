"""Challenge 2: Enhancing Agents with Callbacks.
"""

import logging
import os

from google.adk.agents import Agent

from .callbacks import chained_before_callback, log_model_response
from .models import build_model
from .tools import get_extended_weather_forecast, get_lat_lon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

# adk run / adk web load this module and look for root_agent.
root_agent = weather_agent


if __name__ == "__main__":
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    async def run_turn(runner: Runner, user_id: str, session_id: str, message: str) -> None:
        logger.info("--- user: %s ---", message)
        content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                logger.info("%s", event.content.parts[0].text)

    async def main() -> None:
        session_service = InMemorySessionService()
        runner = Runner(
            agent=weather_agent,
            app_name="weather_alerts_challenge_2",
            session_service=session_service,
        )
        session = await session_service.create_session(
            app_name="weather_alerts_challenge_2", user_id="local-test-user"
        )
        for message in [
            "What's the weather like in Chicago?",
            "What's the weather like in Paris, France?",
        ]:
            await run_turn(runner, "local-test-user", session.id, message)

    asyncio.run(main())

