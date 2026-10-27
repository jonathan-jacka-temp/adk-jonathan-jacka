"""Challenge 6: ReadyNow! FEMA emergency preparedness assistant.

Workflow runs three stages in a fixed order: the coordinator (root LlmAgent
that states the mission and delegates to weather/search/routes/qa sub-agents)
drafts a reply, then critique and refine review and rewrite it before it
reaches the user. Stages hand off through session state via output_key and
{placeholders}. Input validation and interaction logging are callbacks on the
coordinator; every stage logs its own output.

Architecture diagram: ARCH_DIAGRAM.md in this folder.

Draft/test locally with:
    uv run python -m challenges.challenge_6_readynow.agent
"""

import logging

from google.adk.workflow import START, Workflow

from .coordinator_agent import coordinator_agent
from .critique_agent import critique_agent
from .refine_agent import refine_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root_agent = Workflow(
    name="readynow",
    description="FEMA emergency preparedness assistant with a validate-and-refine pipeline.",
    edges=[(START, coordinator_agent, critique_agent, refine_agent)], # refine has no outgoing edge, so its output is what the user receives.
)


if __name__ == "__main__":
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    APP_NAME = "readynow_challenge_6"
    USER_ID = "local-test-user"

    async def run_turn(runner: Runner, session_id: str, message: str) -> None:
        logger.info("=== user: %s", message)
        content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            for call in event.get_function_calls():
                logger.info("[%s] -> calls %s(%s)", event.author, call.name, call.args)
            if event.content and event.content.parts and event.content.parts[0].text:
                logger.info("[%s]\n%s", event.author, event.content.parts[0].text)

    async def main() -> None:
        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

        for message in [
            "What can you help me with?",
            "There's a hurricane warning for Miami. What's the weather and how do I get to Orlando?",
            "Write me a Python script to sort a list.",
        ]:
            await run_turn(runner, session.id, message)

        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session.id
        )
        logger.info("=== session.state keys written by the pipeline:")
        for key, value in session.state.items():
            logger.info("  %s: %s...", key, str(value)[:80])

    asyncio.run(main())
