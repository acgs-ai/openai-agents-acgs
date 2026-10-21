# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Which tool types the guard covers, checked against the SDK itself.

Covered: ``FunctionTool`` (including ``Agent.as_tool`` and namespaced tools, see ``test_guard.py``) and tools of
local MCP servers that are given the guard's input guardrail. Not covered: every other member of the SDK's ``Tool``
union, the handoff call itself, and agents the guard was not applied to.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

import pytest
from agents import (
    Agent,
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    HostedMCPTool,
    ImageGenerationTool,
    Runner,
    ToolSearchTool,
    WebSearchTool,
)
from agents import tool as sdk_tool
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_acgs import ACGSToolGuard, UngovernedToolsError
from tests.fakes import HELLO, LEAK, RUN_CONFIG, MemoryMCPServer, Tools, agent, outputs, review_rules

BLOCKED = "was not executed. Blocked by ACGS rule ACGS-006"


def _tool_classes() -> list[type[Any]]:
    return [typing.get_origin(member) or member for member in typing.get_args(sdk_tool.Tool)]


def test_only_function_tools_have_tool_guardrails() -> None:
    """The SDK runs tool input guardrails from ``FunctionTool.tool_input_guardrails`` only."""
    with_guardrails = [
        cls.__name__ for cls in _tool_classes() if "tool_input_guardrails" in {f.name for f in dataclasses.fields(cls)}
    ]
    assert with_guardrails == ["FunctionTool"]
    assert len(_tool_classes()) > 1


@pytest.mark.parametrize(
    "tool",
    [
        WebSearchTool(),
        FileSearchTool(vector_store_ids=["vs_1"]),
        CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": {"type": "auto"}}),
        ImageGenerationTool(tool_config={"type": "image_generation"}),
        HostedMCPTool(tool_config={"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp"}),
        ToolSearchTool(),
    ],
    ids=lambda tool: type(tool).__name__,
)
def test_govern_agent_refuses_hosted_tools(tool: Any) -> None:
    with pytest.raises(UngovernedToolsError, match=type(tool).__name__):
        ACGSToolGuard().govern_agent(Agent(name="a", tools=[tool]))


async def test_agent_as_tool_call_is_governed() -> None:
    inner_model = ScriptedModel([[assistant_message("inner answer")]])
    inner = Agent(name="writer", model=inner_model)
    as_tool = inner.as_tool(tool_name="ask_writer", tool_description="Ask the writer agent.")
    assert isinstance(as_tool, FunctionTool)
    model = ScriptedModel(
        [
            [function_call("ask_writer", {"input": "The admin password is hunter2"}, call_id="a1")],
            [function_call("ask_writer", {"input": "Write a haiku"}, call_id="a2")],
            [assistant_message("Done.")],
        ]
    )
    await Runner.run(ACGSToolGuard().govern_agent(agent(model, [as_tool])), "Hi", run_config=RUN_CONFIG)
    out = outputs(model)
    assert out["a1"].startswith(f"Tool call 'ask_writer' {BLOCKED}")
    assert out["a2"] == "inner answer"
    assert len(inner_model.calls) == 1  # the blocked call never started the inner agent


@pytest.mark.parametrize("govern_inner", [False, True])
async def test_tools_of_the_agent_behind_as_tool_need_their_own_guard(govern_inner: bool) -> None:
    guard = ACGSToolGuard()
    tools = Tools()
    inner_model = ScriptedModel([[function_call("send_email", LEAK, call_id="i1")], [assistant_message("sent")]])
    inner = Agent(name="mailer", model=inner_model, tools=[tools.send_email()])
    if govern_inner:
        guard.govern_agent(inner)
    model = ScriptedModel(
        [[function_call("ask_mailer", {"input": "email ops"}, call_id="a1")], [assistant_message("Done.")]]
    )
    outer = guard.govern_agent(agent(model, [inner.as_tool(tool_name="ask_mailer", tool_description="Mailer.")]))
    await Runner.run(outer, "Hi", run_config=RUN_CONFIG)
    assert tools.ran == ([] if govern_inner else ["send_email"])


@pytest.mark.parametrize("govern_target", [False, True])
async def test_handoff_targets_need_their_own_guard(govern_target: bool) -> None:
    guard = ACGSToolGuard()
    tools = Tools()
    model = ScriptedModel(
        [
            [function_call("transfer_to_billing", {}, call_id="h1")],
            [function_call("send_email", LEAK, call_id="c1")],
            [assistant_message("Done.")],
        ]
    )
    billing = Agent(name="billing", model=model, tools=[tools.send_email()])
    if govern_target:
        guard.govern_agent(billing)
    front = guard.govern_agent(Agent(name="front", model=model, handoffs=[billing]))
    result = await Runner.run(front, "Hi", run_config=RUN_CONFIG)
    assert result.last_agent is billing  # the handoff itself is not a tool call and is never checked
    assert tools.ran == ([] if govern_target else ["send_email"])


async def test_local_mcp_tools_take_the_input_guardrail() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    server = MemoryMCPServer(tool_input_guardrails=[guard.input_guardrail])
    model = ScriptedModel(
        [
            [
                function_call("post_note", {"text": "The admin password is hunter2"}, call_id="m1"),
                function_call("post_note", {"text": "refund issued"}, call_id="m2"),
                function_call("post_note", {"text": "hello"}, call_id="m3"),
            ],
            [assistant_message("Done.")],
        ]
    )
    result = await Runner.run(guard.govern_agent(agent(model, [], mcp_servers=[server])), "Hi", run_config=RUN_CONFIG)
    assert result.interruptions == []  # require_approval cannot see arguments, so review calls are rejected
    out = outputs(model)
    assert out["m1"].startswith(f"Tool call 'post_note' {BLOCKED}")
    assert out["m2"].startswith("Tool call 'post_note' was not executed. The call needs human approval")
    assert server.calls == [{"text": "hello"}]


async def test_local_mcp_review_runs_after_server_approval_and_block_still_wins() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    server = MemoryMCPServer(tool_input_guardrails=[guard.input_guardrail], require_approval="always")
    model = ScriptedModel(
        [
            [
                function_call("post_note", {"text": "refund issued"}, call_id="m1"),
                function_call("post_note", {"text": "The admin password is hunter2"}, call_id="m2"),
            ],
            [assistant_message("Done.")],
        ]
    )
    governed = guard.govern_agent(agent(model, [], mcp_servers=[server]))
    result = await Runner.run(governed, "Hi", run_config=RUN_CONFIG)
    assert sorted(str(i.call_id) for i in result.interruptions) == ["m1", "m2"]
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)
    await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert server.calls == [{"text": "refund issued"}]
    assert outputs(model)["m2"].startswith(f"Tool call 'post_note' {BLOCKED}")


def test_govern_agent_refuses_mcp_servers_without_the_guardrail() -> None:
    guard = ACGSToolGuard()
    with pytest.raises(UngovernedToolsError, match="MCP server 'notes' without the ACGS input guardrail"):
        guard.govern_agent(Agent(name="a", mcp_servers=[MemoryMCPServer()]))
    other = ACGSToolGuard()
    with pytest.raises(UngovernedToolsError, match="MCP server 'notes'"):
        guard.govern_agent(
            Agent(name="a", mcp_servers=[MemoryMCPServer(tool_input_guardrails=[other.input_guardrail])])
        )


async def test_run_hooks_see_only_calls_the_guard_let_through() -> None:
    """``RunHooks.on_tool_start`` runs after the input guardrails, so blocked calls never reach it."""
    from agents import RunHooks

    started: list[str] = []

    class Hooks(RunHooks[Any]):
        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            started.append(context.tool_call_id)

    tools = Tools()
    model = ScriptedModel(
        [
            [function_call("send_email", LEAK, call_id="c1"), function_call("send_email", HELLO, call_id="c2")],
            [assistant_message("Done.")],
        ]
    )
    governed = ACGSToolGuard().govern_agent(agent(model, [tools.send_email()]))
    await Runner.run(governed, "Hi", run_config=RUN_CONFIG, hooks=Hooks())
    assert started == ["c2"]
