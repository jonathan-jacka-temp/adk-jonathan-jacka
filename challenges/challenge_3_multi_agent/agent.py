"""Challenge 3: Developing Multi-Agent Systems.

Root agent delegates to a weather sub-agent and a Google Search sub-agent
with mode="single_turn" to expose as tools automatically.  This made more
sense to me for a UX to keep root as the 'single voice' the user talks to.
Draft/test locally with:

    uv run python -m challenges.challenge_3_multi_agent.agent
"""

import logging
import os

from google.adk.agents import Agent

from .callbacks import chained_before_callback, log_model_response
from .models import build_model
from .search_agent import search_agent
from .weather_agent import weather_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# e.g. AGENT_MODEL=gpt-4o uv run python agent.py
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

ROOT_AGENT_INSTRUCTIONS = """
You are a helpful assistant. You can check US weather and search the web.
- For weather questions about a US location, call weather_agent.
- For any other factual or current-events question, call search_agent.
Use the sub-agent's result to answer the user directly. If a request is not
about weather or something searchable, say what you can help with.
"""

root_agent = Agent(
    name="root_agent",
    model=build_model(AGENT_MODEL),
    description="Routes user requests to the weather or search sub-agent.",
    instruction=ROOT_AGENT_INSTRUCTIONS,
    sub_agents=[weather_agent, search_agent],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)


if __name__ == "__main__":
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    APP_NAME = "multi_agent_challenge_3"
    USER_ID = "local-test-user"

    async def run_turn(runner: Runner, session_id: str, message: str) -> None:
        logger.info("=== user: %s", message)
        content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            for call in event.get_function_calls():
                logger.info("[%s] -> calls %s(%s)", event.author, call.name, call.args)
            for response in event.get_function_responses():
                logger.info("[%s] <- %s returned", event.author, response.name)
            if event.content and event.content.parts and event.content.parts[0].text:
                logger.info("[%s] %s", event.author, event.content.parts[0].text)

    async def main() -> None:
        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        for message in [
            "What's the weather like in Chicago?",
            "Who won the most recent Super Bowl?",
            "What's the weather like in Paris, France?",
        ]:
            await run_turn(runner, session.id, message)

    asyncio.run(main())
