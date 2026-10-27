"""
Basic e2e test that swaps each stage's model for a stub that echoes a canned reply
plus the instruction it received, so the test runs with no API key and no network
while still pushing the real Workflow through the real Runner.

Proves:

- the four stages run in the declared order
- each stage's output_key lands in session state
- each downstream {placeholder} is resolved before the model sees it
- the user receives refine's output, not an intermediate stage's
"""

import asyncio
from unittest.mock import patch

import pytest

from . import callbacks
from .critique_agent import critique_agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from google.genai import types as genai_types
from .greeter_agent import greeter_agent
from .refine_agent import refine_agent
from .search_agent import search_agent

STAGE_REPLIES = {
    "greeter": "Q: What is the Google ADK and when should it be used?",
    "search": "DRAFT: ADK is a framework for building LLM agents.",
    "critique": "NOTES: add a concrete use case.",
    "refine": "FINAL: ADK is a framework for building LLM agents, e.g. a support bot.",
}


class StubModel(BaseLlm):
    """Replies with a canned string and records the resolved instruction it was given."""

    model: str = "stub"
    reply: str
    seen_instructions: list = []

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["stub"]

    async def generate_content_async(self, llm_request, stream=False):
        instruction = llm_request.config.system_instruction if llm_request.config else ""
        self.seen_instructions.append(str(instruction))
        yield LlmResponse(
            content=genai_types.Content(
                role="model", parts=[genai_types.Part.from_text(text=self.reply)]
            )
        )


@pytest.fixture
def stubbed_pipeline():
    """Yields (workflow, stubs). Builds a fresh Workflow after stubbing.

    Workflow clones each node at construction time, so stubbing the module-level
    agents after importing agent.root_agent has no effect on it — the Workflow
    has to be built here, after the models are swapped.
    """
    stages = {
        "greeter": greeter_agent,
        "search": search_agent,
        "critique": critique_agent,
        "refine": refine_agent,
    }
    originals = {name: (agent.model, agent.tools) for name, agent in stages.items()}
    stubs = {}
    for name, agent in stages.items():
        stub = StubModel(reply=STAGE_REPLIES[name], seen_instructions=[])
        agent.model = stub
        stubs[name] = stub
    search_agent.tools = []  # google_search needs a real model; irrelevant to this test

    workflow = Workflow(
        name="answer_team",
        edges=[(START, greeter_agent, search_agent, critique_agent, refine_agent)],
    )
    yield workflow, stubs

    for name, agent in stages.items():
        agent.model, agent.tools = originals[name]


async def _run_pipeline(workflow: Workflow, message: str):
    session_service = InMemorySessionService()
    runner = Runner(agent=workflow, app_name="test", session_service=session_service)
    session = await session_service.create_session(app_name="test", user_id="u")
    content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])

    authors, texts = [], []
    with patch.object(callbacks, "_classify", return_value="OK"):
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=content):
            if event.content and event.content.parts and event.content.parts[0].text:
                authors.append(event.author)
                texts.append(event.content.parts[0].text)

    session = await session_service.get_session(app_name="test", user_id="u", session_id=session.id)
    return authors, texts, session.state


def test_stages_run_in_declared_order(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    authors, _, _ = asyncio.run(_run_pipeline(workflow, "explain the google adk"))
    assert authors == ["greeter", "search", "critique", "refine"]


def test_each_stage_writes_its_output_key_to_state(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    _, _, state = asyncio.run(_run_pipeline(workflow, "explain the google adk"))
    assert state["question"] == STAGE_REPLIES["greeter"]
    assert state["draft_answer"] == STAGE_REPLIES["search"]
    assert state["critique_notes"] == STAGE_REPLIES["critique"]
    assert state["final_answer"] == STAGE_REPLIES["refine"]


def test_downstream_placeholders_are_resolved_from_state(stubbed_pipeline):
    workflow, stubs = stubbed_pipeline
    asyncio.run(_run_pipeline(workflow, "explain the google adk"))

    search_saw = stubs["search"].seen_instructions[0]
    critique_saw = stubs["critique"].seen_instructions[0]
    refine_saw = stubs["refine"].seen_instructions[0]

    assert STAGE_REPLIES["greeter"] in search_saw
    assert STAGE_REPLIES["search"] in critique_saw
    assert STAGE_REPLIES["critique"] in refine_saw

    for resolved in (search_saw, critique_saw, refine_saw):
        for placeholder in ("{question}", "{draft_answer}", "{critique_notes}"):
            assert placeholder not in resolved


def test_final_reply_to_user_is_refine_output(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    _, texts, _ = asyncio.run(_run_pipeline(workflow, "explain the google adk"))
    assert texts[-1] == STAGE_REPLIES["refine"]
