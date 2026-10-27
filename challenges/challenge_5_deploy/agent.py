"""Challenge 5: Deploying Agents.

The challenge-4 Workflow, unchanged: greeter, search, critique, and refine run
in a fixed order, handing off through session state via output_key and
{placeholders}. This file stays the ADK entrypoint (root_agent); the deploy to
Agent Engine lives in deploy.py.

Draft/test locally with:
    uv run python -m challenges.challenge_5_deploy.agent
"""

import logging

from google.adk.workflow import START, Workflow

from .critique_agent import critique_agent
from .greeter_agent import greeter_agent
from .refine_agent import refine_agent
from .search_agent import search_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


root_agent = Workflow(
    name="answer_team",
    description="Answers a question, then critiques and refines the answer before replying.",
    edges=[(START, greeter_agent, search_agent, critique_agent, refine_agent)],
)


if __name__ == "__main__":
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    APP_NAME = "workflow_challenge_4"
    USER_ID = "local-test-user"

    async def run_turn(runner: Runner, session_id: str, message: str) -> None:
        logger.info("=== user: %s", message)
        content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                logger.info("[%s]\n%s", event.author, event.content.parts[0].text)

    async def main() -> None:
        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

        await run_turn(
            runner, session.id, "Explain what the Google ADK is and when I'd use it."
        )

        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session.id
        )
        logger.info("=== session.state keys written by the pipeline:")
        for key, value in session.state.items():
            logger.info("  %s: %s...", key, str(value)[:80])

    asyncio.run(main())
