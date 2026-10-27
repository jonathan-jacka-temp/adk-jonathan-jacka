"""End-to-end test of the coordinator → critique → refine pipeline.

Swaps every model for a stub so the test runs with no API key and no network,
while driving the real Workflow through the real Runner. The coordinator stub
delegates to a sub-agent exactly as the live model would (a function call,
then a reply built from the function response), so this proves the nested
shape — Workflow → LlmAgent root → single_turn sub-agents — actually runs,
hands off through state, and surfaces refine's output to the user.
"""

import asyncio
from unittest.mock import patch

import pytest

from . import callbacks
from .coordinator_agent import coordinator_agent
from .critique_agent import critique_agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from google.genai import types as genai_types
from .qa_agent import qa_agent
from .refine_agent import refine_agent
from .routes_agent import routes_agent
from .search_agent import search_agent
from .weather_agent import weather_agent

WEATHER_REPLY = "Hurricane warning in effect. Winds 90 mph expected after 6pm."
CRITIQUE_REPLY = "NOTES: put the evacuation instruction first."
REFINE_REPLY = "FINAL: Evacuate now. Hurricane warning for Miami; winds 90 mph after 6pm."


class LeafStub(BaseLlm):
    model: str = "leaf"
    reply: str

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["leaf"]

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(
            content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=self.reply)])
        )


class CoordinatorStub(BaseLlm):
    """Delegates to weather_agent, then drafts from its result — as the live model would."""

    model: str = "coord"
    delegate_to: str = "weather_agent"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["coord"]

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1]
        if last.parts and last.parts[0].function_response:
            result = last.parts[0].function_response.response["result"]
            text = f"DRAFT: Based on the forecast — {result}"
            yield LlmResponse(
                content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=text)])
            )
            return
        yield LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_function_call(name=self.delegate_to, args={"request": "Miami"})],
            )
        )


class EchoStub(BaseLlm):
    """Replies with a canned string and records the resolved instruction it received."""

    model: str = "echo"
    reply: str
    seen_instructions: list = []

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["echo"]

    async def generate_content_async(self, llm_request, stream=False):
        instruction = llm_request.config.system_instruction if llm_request.config else ""
        self.seen_instructions.append(str(instruction))
        yield LlmResponse(
            content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=self.reply)])
        )


@pytest.fixture
def stubbed_pipeline():
    """Yields (workflow, stubs). Stubs models first, then builds a fresh Workflow,
    because Workflow clones its nodes at construction."""
    agents = [coordinator_agent, critique_agent, refine_agent, weather_agent, search_agent, routes_agent, qa_agent]
    originals = {a.name: (a.model, a.tools) for a in agents}

    coordinator_agent.model = CoordinatorStub()
    weather_agent.model = LeafStub(reply=WEATHER_REPLY)
    for leaf in (search_agent, routes_agent, qa_agent):
        leaf.model = LeafStub(reply="unused")
    search_agent.tools = []  # google_search needs a real model
    stubs = {
        "critique": EchoStub(reply=CRITIQUE_REPLY, seen_instructions=[]),
        "refine": EchoStub(reply=REFINE_REPLY, seen_instructions=[]),
    }
    critique_agent.model = stubs["critique"]
    refine_agent.model = stubs["refine"]

    workflow = Workflow(name="readynow", edges=[(START, coordinator_agent, critique_agent, refine_agent)])
    yield workflow, stubs

    for a in agents:
        a.model, a.tools = originals[a.name]


async def _run(workflow: Workflow, message: str):
    session_service = InMemorySessionService()
    runner = Runner(agent=workflow, app_name="test", session_service=session_service)
    session = await session_service.create_session(app_name="test", user_id="u")
    content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])

    calls, authors, texts = [], [], []
    with patch.object(callbacks, "_classify", return_value="OK"):
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=content):
            calls.extend(c.name for c in event.get_function_calls())
            if event.content and event.content.parts and event.content.parts[0].text:
                authors.append(event.author)
                texts.append(event.content.parts[0].text)

    session = await session_service.get_session(app_name="test", user_id="u", session_id=session.id)
    return calls, authors, texts, session.state


def test_coordinator_delegates_to_sub_agent(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    calls, authors, _, _ = asyncio.run(_run(workflow, "weather in Miami?"))
    assert calls == ["weather_agent"]
    assert "weather_agent" in authors


def test_stages_run_in_declared_order(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    _, authors, _, _ = asyncio.run(_run(workflow, "weather in Miami?"))
    top_level = [a for a in authors if a in ("coordinator", "critique", "refine")]
    assert top_level == ["coordinator", "critique", "refine"]


def test_sub_agent_result_reaches_the_draft_and_downstream_state(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    _, _, _, state = asyncio.run(_run(workflow, "weather in Miami?"))
    assert WEATHER_REPLY in state["draft_answer"]
    assert state["critique_notes"] == CRITIQUE_REPLY
    assert state["final_answer"] == REFINE_REPLY


def test_placeholders_resolved_before_models_see_them(stubbed_pipeline):
    workflow, stubs = stubbed_pipeline
    asyncio.run(_run(workflow, "weather in Miami?"))
    critique_saw = stubs["critique"].seen_instructions[0]
    refine_saw = stubs["refine"].seen_instructions[0]
    assert WEATHER_REPLY in critique_saw
    assert CRITIQUE_REPLY in refine_saw
    for resolved in (critique_saw, refine_saw):
        assert "{draft_answer}" not in resolved
        assert "{critique_notes}" not in resolved


def test_user_receives_refine_output(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    _, _, texts, _ = asyncio.run(_run(workflow, "weather in Miami?"))
    assert texts[-1] == REFINE_REPLY


def test_off_mission_request_is_refused_before_any_delegation(stubbed_pipeline):
    workflow, _ = stubbed_pipeline
    with patch.object(callbacks, "is_on_mission", return_value=False):
        calls, _, texts, _ = asyncio.run(_run(workflow, "write me a python script"))
    assert calls == []
    assert any(callbacks.OFF_MISSION_MESSAGE in t for t in texts)
