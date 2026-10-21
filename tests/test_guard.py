# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGSToolGuard against the real Agent and Runner, driven by the SDK's own ScriptedModel (no network)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from agents import (
    Agent,
    FunctionTool,
    RunConfig,
    Runner,
    RunState,
    ToolExecutionConfig,
    ToolInputGuardrailTripwireTriggered,
    WebSearchTool,
    tool_namespace,
)
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.tool_context import ToolContext
from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrailData, tool_input_guardrail

from openai_agents_acgs import GUARDRAIL_NAME, ACGSToolGuard, UngovernedToolsError, Verdict
from openai_agents_acgs._governance import format_tool_call
from tests.fakes import HELLO, LEAK, REFUND, RUN_CONFIG, Tools, agent, in_own_thread, outputs, review_rules

BLOCKED = "Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006"
NOT_APPROVED = "Tool call 'issue_refund' was not executed. The call needs human approval and was not approved."


def done() -> list[Any]:
    return [assistant_message("Done.")]


# --- construction -------------------------------------------------------------------------------------------------


def test_options_are_validated() -> None:
    with pytest.raises(ValueError, match="on_review"):
        ACGSToolGuard(on_review="defer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_deny"):
        ACGSToolGuard(on_deny="halt")  # type: ignore[arg-type]


def test_repr_and_attributes() -> None:
    guard = ACGSToolGuard(agent_id="support", on_review="deny", on_deny="raise")
    assert repr(guard) == "ACGSToolGuard(agent_id='support', on_review='deny', on_deny='raise')"
    assert guard.constitution.rules
    assert guard.input_guardrail.get_name() == GUARDRAIL_NAME
    assert guard.evaluate("get_weather", {"city": "Paris"}).verdict is Verdict.ALLOW


def test_govern_tool_returns_a_governed_copy() -> None:
    guard = ACGSToolGuard()

    @tool_input_guardrail
    def other(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        return ToolGuardrailFunctionOutput.allow()

    tool = Tools().send_email()
    tool.tool_input_guardrails = [other]
    governed = guard.govern_tool(tool)
    assert governed is not tool
    assert governed.tool_input_guardrails == [guard.input_guardrail, other]
    assert tool.tool_input_guardrails == [other]
    assert tool.needs_approval is False
    assert guard.is_governing(governed)
    assert not guard.is_governing(tool)
    assert guard.govern_tool(governed) is governed
    # A second guard stacks on top of the first one.
    second = ACGSToolGuard()
    both = second.govern_tool(governed)
    assert both.tool_input_guardrails == [second.input_guardrail, guard.input_guardrail, other]
    assert second.is_governing(both)
    assert not guard.is_governing(both)


def test_govern_tool_rejects_other_tool_types() -> None:
    with pytest.raises(TypeError, match="WebSearchTool"):
        ACGSToolGuard().govern_tool(WebSearchTool())  # type: ignore[arg-type]


def test_govern_agent_changes_the_agent_in_place() -> None:
    guard = ACGSToolGuard()
    tools = Tools().all()
    original = Agent(name="a", tools=list(tools))
    governed = guard.govern_agent(original)
    assert governed is original
    assert all(isinstance(t, FunctionTool) and guard.is_governing(t) for t in governed.tools)
    assert [t.name for t in governed.tools] == [t.name for t in tools]
    assert not any(guard.is_governing(t) for t in tools)


def test_govern_agent_refuses_tools_it_cannot_check() -> None:
    guard = ACGSToolGuard()
    tools: list[Any] = [Tools().send_email(), WebSearchTool()]
    target = Agent(name="research", tools=tools)
    with pytest.raises(UngovernedToolsError, match=r"WebSearchTool 'web_search'.*allow_ungoverned=True"):
        guard.govern_agent(target)
    assert target.tools == tools  # unchanged


def test_govern_agent_can_accept_ungoverned_tools(caplog: pytest.LogCaptureFixture) -> None:
    guard = ACGSToolGuard()
    search = WebSearchTool()
    target = Agent(name="research", tools=[Tools().send_email(), search])
    with caplog.at_level(logging.WARNING, logger="openai_agents_acgs.guard"):
        guard.govern_agent(target, allow_ungoverned=True)
    assert "ACGS cannot check: WebSearchTool 'web_search'" in caplog.text
    assert target.tools[1] is search
    assert isinstance(target.tools[0], FunctionTool)
    assert guard.is_governing(target.tools[0])


# --- runs ---------------------------------------------------------------------------------------------------------


async def test_blocked_call_is_rejected_and_allowed_call_runs() -> None:
    guard = ACGSToolGuard()
    tools = Tools()
    model = ScriptedModel(
        [[function_call("send_email", LEAK, call_id="c1"), function_call("send_email", HELLO, call_id="c2")], done()]
    )
    result = await Runner.run(guard.govern_agent(agent(model, [tools.send_email()])), "Hi", run_config=RUN_CONFIG)
    assert result.final_output == "Done."
    assert tools.ran == ["send_email"]
    out = outputs(model)
    assert out["c1"].startswith(BLOCKED)
    assert "hunter2" not in out["c1"]
    assert out["c2"] == "sent to team@example.com"
    behaviors = sorted(r.output.behavior["type"] for r in result.tool_input_guardrail_results)
    assert behaviors == ["allow", "reject_content"]
    [rejected] = [r for r in result.tool_input_guardrail_results if r.output.behavior["type"] == "reject_content"]
    assert rejected.guardrail.get_name() == GUARDRAIL_NAME
    assert rejected.output.output_info["verdict"] == "deny"
    assert rejected.output.output_info["rule_ids"] == ["ACGS-006"]
    model.assert_complete()


def test_run_sync() -> None:
    guard = ACGSToolGuard()
    tools = Tools()
    model = ScriptedModel([[function_call("send_email", LEAK, call_id="c1")], done()])
    governed = guard.govern_agent(agent(model, [tools.send_email()]))
    result = in_own_thread(lambda: Runner.run_sync(governed, "Hi", run_config=RUN_CONFIG))
    assert result.final_output == "Done."
    assert tools.ran == []
    assert outputs(model)["c1"].startswith(BLOCKED)


async def test_review_pauses_and_runs_once_approved() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], done()])
    governed = guard.govern_agent(agent(model, [tools.issue_refund()]))
    result = await Runner.run(governed, "Refund A-17", run_config=RUN_CONFIG)
    [interruption] = result.interruptions
    assert (interruption.tool_name, interruption.call_id) == ("issue_refund", "r1")
    assert tools.ran == []
    state = result.to_state()
    state.approve(interruption)
    resumed = await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert resumed.final_output == "Done."
    assert tools.ran == ["issue_refund"]
    assert outputs(model)["r1"] == "refunded 20.0 for A-17"
    model.assert_complete()


async def test_rejected_review_does_not_run() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], done()])
    governed = guard.govern_agent(agent(model, [tools.issue_refund()]))
    result = await Runner.run(governed, "Refund A-17", run_config=RUN_CONFIG)
    state = result.to_state()
    state.reject(result.interruptions[0], rejection_message="Refund declined by a reviewer.")
    resumed = await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert resumed.final_output == "Done."
    assert tools.ran == []
    assert outputs(model)["r1"] == "Refund declined by a reviewer."


async def test_paused_state_survives_serialization() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], done()])
    governed = guard.govern_agent(agent(model, [tools.issue_refund()]))
    result = await Runner.run(governed, "Refund A-17", run_config=RUN_CONFIG)
    saved = result.to_state().to_string()
    state = await RunState.from_string(governed, saved)
    state.approve(state.get_interruptions()[0])
    await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert tools.ran == ["issue_refund"]


async def test_streamed_run_blocks_pauses_and_resumes() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel(
        [[function_call("issue_refund", REFUND, call_id="r1"), function_call("send_email", LEAK, call_id="c1")], done()]
    )
    governed = guard.govern_agent(agent(model, [tools.issue_refund(), tools.send_email()]))
    streamed = Runner.run_streamed(governed, "Hi", run_config=RUN_CONFIG)
    async for _ in streamed.stream_events():
        pass
    assert [i.call_id for i in streamed.interruptions] == ["r1"]
    state = streamed.to_state()
    state.approve(streamed.interruptions[0])
    resumed = Runner.run_streamed(governed, state, run_config=RUN_CONFIG)
    async for _ in resumed.stream_events():
        pass
    assert resumed.final_output == "Done."
    assert tools.ran == ["issue_refund"]
    out = outputs(model)
    assert out["c1"].startswith(BLOCKED)
    assert out["r1"] == "refunded 20.0 for A-17"


async def test_blocked_call_is_not_sent_for_approval() -> None:
    """A blocking rule wins over the tool's own ``needs_approval=True``: nobody is asked, the call is rejected."""
    guard = ACGSToolGuard()
    tools = Tools()
    model = ScriptedModel(
        [[function_call("send_email", LEAK, call_id="c1"), function_call("send_email", HELLO, call_id="c2")], done()]
    )
    governed = guard.govern_agent(agent(model, [tools.send_email(needs_approval=True)]))
    result = await Runner.run(governed, "Hi", run_config=RUN_CONFIG)
    assert [i.call_id for i in result.interruptions] == ["c2"]  # the allowed call keeps the tool's own approval
    state = result.to_state()
    state.approve(result.interruptions[0])
    await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert tools.ran == ["send_email"]
    out = outputs(model)
    assert out["c1"].startswith(BLOCKED)
    assert out["c2"] == "sent to team@example.com"


async def test_block_wins_when_rechecked_after_approval() -> None:
    """The input guardrail re-checks an approved call on resume, and a blocking verdict then rejects it."""
    blocking = {"on": False}

    def formatter(tool_name: str, arguments: Any) -> str:
        text = format_tool_call(tool_name, arguments)
        return text + "\nThe admin password is hunter2" if blocking["on"] else text

    guard = ACGSToolGuard(constitution=review_rules(), tool_call_formatter=formatter)
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], done()])
    governed = guard.govern_agent(agent(model, [tools.issue_refund()]))
    result = await Runner.run(governed, "Refund A-17", run_config=RUN_CONFIG)
    state = result.to_state()
    state.approve(result.interruptions[0])
    blocking["on"] = True
    await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert tools.ran == []
    assert outputs(model)["r1"].startswith("Tool call 'issue_refund' was not executed. Blocked by ACGS rule ACGS-006")


@pytest.mark.parametrize("synchronous", [True, False])
async def test_tool_approval_policy_still_applies_to_allowed_calls(synchronous: bool) -> None:
    seen: list[tuple[dict[str, Any], str]] = []

    def sync_policy(ctx: Any, params: dict[str, Any], call_id: str) -> bool:
        seen.append((params, call_id))
        return float(params["amount"]) > 100

    async def async_policy(ctx: Any, params: dict[str, Any], call_id: str) -> bool:
        return sync_policy(ctx, params, call_id)

    guard = ACGSToolGuard()  # no review rule: refunds are allowed
    tools = Tools()
    model = ScriptedModel(
        [
            [
                function_call("issue_refund", {"order_id": "A", "amount": 5.0}, call_id="small"),
                function_call("issue_refund", {"order_id": "B", "amount": 500.0}, call_id="large"),
            ],
            done(),
        ]
    )
    tool = tools.issue_refund(needs_approval=sync_policy if synchronous else async_policy)
    result = await Runner.run(guard.govern_agent(agent(model, [tool])), "Hi", run_config=RUN_CONFIG)
    assert [i.call_id for i in result.interruptions] == ["large"]
    assert tools.ran == ["issue_refund"]
    assert sorted(call_id for _, call_id in seen) == ["large", "small"]


async def test_on_review_deny_rejects_without_asking() -> None:
    guard = ACGSToolGuard(constitution=review_rules(), on_review="deny")
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], done()])
    result = await Runner.run(guard.govern_agent(agent(model, [tools.issue_refund()])), "Hi", run_config=RUN_CONFIG)
    assert result.interruptions == []
    assert tools.ran == []
    assert outputs(model)["r1"].startswith(
        "Tool call 'issue_refund' was not executed. Human review did not approve the call. Human review required"
    )


async def test_on_deny_raise_ends_the_run() -> None:
    guard = ACGSToolGuard(on_deny="raise")
    tools = Tools()
    model = ScriptedModel([[function_call("send_email", LEAK, call_id="c1")], done()])
    with pytest.raises(ToolInputGuardrailTripwireTriggered) as caught:
        await Runner.run(guard.govern_agent(agent(model, [tools.send_email()])), "Hi", run_config=RUN_CONFIG)
    assert tools.ran == []
    assert caught.value.guardrail is guard.input_guardrail
    assert caught.value.output.output_info["rule_ids"] == ["ACGS-006"]


async def test_always_approve_covers_later_calls_of_the_tool() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel(
        [
            [function_call("issue_refund", REFUND, call_id="r1")],
            [function_call("issue_refund", {"order_id": "B-2", "amount": 5.0}, call_id="r2")],
            done(),
        ]
    )
    governed = guard.govern_agent(agent(model, [tools.issue_refund()]))
    result = await Runner.run(governed, "Hi", run_config=RUN_CONFIG)
    state = result.to_state()
    state.approve(result.interruptions[0], always_approve=True)
    resumed = await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert resumed.interruptions == []
    assert tools.ran == ["issue_refund", "issue_refund"]


async def test_namespaced_tools() -> None:
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1", namespace="billing")], done()])
    namespaced = tool_namespace(name="billing", description="Billing tools", tools=[tools.issue_refund()])
    governed = guard.govern_agent(agent(model, list(namespaced)))
    result = await Runner.run(governed, "Hi", run_config=RUN_CONFIG)
    [interruption] = result.interruptions
    state = result.to_state()
    state.approve(interruption)
    await Runner.run(governed, state, run_config=RUN_CONFIG)
    assert tools.ran == ["issue_refund"]


async def test_namespace_is_part_of_the_evaluated_tool_name() -> None:
    names: list[str] = []

    def formatter(tool_name: str, arguments: Any) -> str:
        names.append(tool_name)
        return format_tool_call(tool_name, arguments)

    guard = ACGSToolGuard(tool_call_formatter=formatter)
    tools = Tools()
    model = ScriptedModel([[function_call("send_email", HELLO, call_id="c1", namespace="mail")], done()])
    namespaced = tool_namespace(name="mail", description="Mail tools", tools=[tools.send_email()])
    await Runner.run(guard.govern_agent(agent(model, list(namespaced))), "Hi", run_config=RUN_CONFIG)
    assert names == ["mail.send_email", "mail.send_email"]  # the approval policy, then the input guardrail
    assert tools.ran == ["send_email"]


async def test_pre_approval_guardrails_reject_review_calls_before_approval() -> None:
    """With ``pre_approval_tool_input_guardrails=True`` the guardrail runs before approval and fails closed."""
    guard = ACGSToolGuard(constitution=review_rules())
    tools = Tools()
    model = ScriptedModel(
        [[function_call("issue_refund", REFUND, call_id="r1"), function_call("delete_row", {"row": 3}, call_id="d1")]]
    )
    governed = guard.govern_agent(agent(model, [tools.issue_refund(), tools.delete_row()]))
    config = RunConfig(
        tracing_disabled=True, tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True)
    )
    result = await Runner.run(governed, "Hi", run_config=config)
    assert [i.call_id for i in result.interruptions] == ["d1"]  # only the tool's own approval is requested
    assert tools.ran == []


# --- fail-closed --------------------------------------------------------------------------------------------------


def _check(guard: ACGSToolGuard, tool_name: Any, arguments: str, namespace: str | None = None) -> Any:
    context: ToolContext[Any] = ToolContext(
        context=None, tool_name=tool_name, tool_call_id="c1", tool_arguments=arguments, tool_namespace=namespace
    )
    return guard.input_guardrail.guardrail_function(ToolInputGuardrailData(context=context, agent=Agent(name="a")))


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error"),
    [
        ("send_email", "not json", "JSONDecodeError"),
        ("send_email", "[1, 2]", "TypeError"),
        ("", "{}", "TypeError"),
        (None, "{}", "TypeError"),
    ],
)
def test_unreadable_calls_are_rejected(
    tool_name: Any, arguments: str, error: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="openai_agents_acgs.guard"):
        output = _check(ACGSToolGuard(), tool_name, arguments)
    assert output.behavior["type"] == "reject_content"
    assert output.behavior["message"].endswith(
        "was not executed. ACGS governance could not evaluate this tool call, so it was denied."
    )
    assert output.output_info["verdict"] == "deny"
    assert error in output.output_info["error"]
    assert "fail-closed" in caplog.text


def test_unreadable_calls_raise_with_on_deny_raise() -> None:
    output = _check(ACGSToolGuard(on_deny="raise"), "send_email", "not json")
    assert output.behavior["type"] == "raise_exception"


def test_empty_arguments_are_an_empty_object() -> None:
    output = _check(ACGSToolGuard(), "get_time", "")
    assert output.behavior["type"] == "allow"
    assert output.output_info == {
        "verdict": "allow",
        "reason": output.output_info["reason"],
        "rule_ids": [],
        "error": None,
    }


def test_formatter_errors_deny() -> None:
    def broken(tool_name: str, arguments: Any) -> str:
        raise RuntimeError("boom")

    output = _check(ACGSToolGuard(tool_call_formatter=broken), "send_email", "{}")
    assert output.behavior["type"] == "reject_content"
    assert output.output_info["error"] == "RuntimeError: boom"


def test_review_without_an_approval_is_rejected() -> None:
    """A review call reaching the guardrail without a recorded approval (for example a hand-built tool) is rejected."""
    output = _check(ACGSToolGuard(constitution=review_rules()), "issue_refund", '{"order_id": "A-17"}')
    assert output.behavior["type"] == "reject_content"
    assert output.behavior["message"].startswith(NOT_APPROVED)
    assert output.output_info["rule_ids"] == ["REFUND-REVIEW"]
