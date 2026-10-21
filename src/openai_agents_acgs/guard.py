# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for OpenAI Agents SDK function tools: a tool input guardrail plus an approval policy."""

from __future__ import annotations

import copy
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from acgs_lite import Constitution
from agents import Agent, FunctionTool, RunContextWrapper
from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData

from openai_agents_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

GUARDRAIL_NAME = "acgs"
"""The name of the ACGS tool input guardrail, as reported in ``RunResult.tool_input_guardrail_results``."""

_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."

ApprovalPolicy = Callable[[RunContextWrapper[Any], dict[str, Any], str], Awaitable[bool]]
"""The callable form of ``FunctionTool.needs_approval``: ``(run_context, params, call_id) -> bool``."""


class UngovernedToolsError(ValueError):
    """Raised by ``ACGSToolGuard.govern_agent`` when an agent has tools the guard cannot check."""


class ACGSToolGuard:
    """Check OpenAI Agents SDK function tool calls against an ACGS constitution before the tool runs.

    The SDK has no run-wide hook that can stop a tool call, so the guard is attached to each
    ``FunctionTool`` with two documented per-tool mechanisms:

    - ``input_guardrail``, a tool input guardrail. The SDK runs it right before the tool executes,
      including after a human approved the call. Blocked calls are rejected
      (``ToolGuardrailFunctionOutput.reject_content``) or, with ``on_deny="raise"``, end the run
      (``raise_exception``). Calls that need review run only once approved.
    - An approval policy set as the tool's ``needs_approval``. Calls that only match review rules
      pause the run with a ``ToolApprovalItem`` interruption, which you approve or reject on the
      ``RunState`` and resume. A tool's own ``needs_approval`` still applies to allowed calls.

    A blocking rule always wins: a blocked call is rejected even after approval, and nobody is asked
    to approve it. Any error while evaluating a call rejects it instead of failing the run.

    Use ``govern_agent(agent)`` to apply the guard to every function tool of an agent, or
    ``govern_tool(tool)`` for one tool.

    Example:
        >>> from openai_agents_acgs import ACGSToolGuard
        >>> guard = ACGSToolGuard()
        >>> guard.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"}).verdict
        <Verdict.DENY: 'deny'>
    """

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "openai-agents",
        on_review: Literal["approve", "deny"] = "approve",
        on_deny: Literal["reject", "raise"] = "reject",
        tool_call_formatter: ToolCallFormatter | None = None,
    ) -> None:
        """Create the guard.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to
                ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: What to do when only review rules match. ``"approve"`` requires approval
                through the SDK's human-in-the-loop flow. ``"deny"`` rejects the call.
            on_deny: What to do with a denied call. ``"reject"`` skips the tool and sends the reason
                to the model as the tool result. ``"raise"`` ends the run with
                ``ToolInputGuardrailTripwireTriggered``.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces
                ``format_tool_call``.

        Raises:
            ValueError: If ``on_review`` or ``on_deny`` is invalid.
        """
        if on_review not in ("approve", "deny"):
            msg = f"on_review must be 'approve' or 'deny', got {on_review!r}"
            raise ValueError(msg)
        if on_deny not in ("reject", "raise"):
            msg = f"on_deny must be 'reject' or 'raise', got {on_deny!r}"
            raise ValueError(msg)
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)
        self.constitution: Constitution = self._governor.constitution
        self.agent_id = agent_id
        self.on_review = on_review
        self.on_deny = on_deny
        self.input_guardrail: ToolInputGuardrail[Any] = ToolInputGuardrail(
            guardrail_function=self._check, name=GUARDRAIL_NAME
        )
        """The tool input guardrail. Also pass it to local MCP servers as ``tool_input_guardrails``."""

    def __repr__(self) -> str:
        """Show the options, not the constitution."""
        return f"ACGSToolGuard(agent_id={self.agent_id!r}, on_review={self.on_review!r}, on_deny={self.on_deny!r})"

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate a tool call without running an agent. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.

        Returns:
            The governance decision.
        """
        return self._governor.evaluate(tool_name, arguments)

    def is_governing(self, tool: FunctionTool) -> bool:
        """Whether ``govern_tool`` already applied this guard to ``tool``.

        Args:
            tool: A function tool.

        Returns:
            True if the tool has this guard's input guardrail and approval policy.
        """
        guardrails = tool.tool_input_guardrails or []
        return any(guardrail is self.input_guardrail for guardrail in guardrails) and (
            getattr(tool.needs_approval, "__acgs_guard__", None) is self
        )

    def govern_tool(self, tool: FunctionTool) -> FunctionTool:
        """Return a copy of ``tool`` that this guard checks. The original tool is not changed.

        The copy runs ``input_guardrail`` before its own input guardrails, and its ``needs_approval``
        becomes an approval policy that asks for approval when ACGS requires review, never for a
        blocked call, and otherwise defers to the tool's own ``needs_approval``.

        Args:
            tool: A function tool, for example one made with ``@function_tool`` or ``Agent.as_tool``.

        Returns:
            The governed copy, or ``tool`` itself if this guard already governs it.

        Raises:
            TypeError: If ``tool`` is not a ``FunctionTool``.
        """
        if not isinstance(tool, FunctionTool):
            msg = f"ACGSToolGuard can only govern FunctionTool instances, got {type(tool).__name__}."
            raise TypeError(msg)
        if self.is_governing(tool):
            return tool
        governed = copy.copy(tool)  # FunctionTool.__copy__ rebinds the invoker to the copy
        governed.tool_input_guardrails = [self.input_guardrail, *(tool.tool_input_guardrails or [])]
        governed.needs_approval = self._approval_policy(tool.qualified_name, own=tool.needs_approval)
        return governed

    def govern_agent(self, agent: Agent[Any], *, allow_ungoverned: bool = False) -> Agent[Any]:
        """Apply the guard to every function tool of ``agent``, in place, and return the agent.

        ``agent.tools`` is replaced with a list where each ``FunctionTool`` is the copy returned by
        ``govern_tool``, so handoffs and other references to this agent see the governed tools.
        Handoff targets and the agents behind ``Agent.as_tool`` keep their own tools: call
        ``govern_agent`` on each of them too.

        Args:
            agent: The agent whose tools to govern.
            allow_ungoverned: Accept tools the guard cannot check (hosted tools, built-in
                execution tools, and local MCP servers without ``input_guardrail``) and only log a
                warning for them. By default they raise ``UngovernedToolsError``.

        Returns:
            The same agent.

        Raises:
            UngovernedToolsError: If the agent has tools the guard cannot check and
                ``allow_ungoverned`` is false. The agent is not changed.
        """
        ungoverned = [
            f"{type(tool).__name__} '{getattr(tool, 'name', '?')}'"
            for tool in agent.tools
            if not isinstance(tool, FunctionTool)
        ]
        ungoverned += [
            f"MCP server '{server.name}' without the ACGS input guardrail"
            for server in agent.mcp_servers
            if not any(g is self.input_guardrail for g in getattr(server, "tool_input_guardrails", None) or [])
        ]
        if ungoverned:
            message = (
                f"Agent '{agent.name}' has tools that ACGS cannot check: {', '.join(ungoverned)}. "
                "Their calls do not run through the tool guardrail pipeline."
            )
            if not allow_ungoverned:
                msg = f"{message} Remove them, or pass allow_ungoverned=True to accept this."
                raise UngovernedToolsError(msg)
            logger.warning("%s", message)
        agent.tools = [self.govern_tool(tool) if isinstance(tool, FunctionTool) else tool for tool in agent.tools]
        return agent

    def _approval_policy(self, tool_name: str, *, own: bool | ApprovalPolicy) -> ApprovalPolicy:
        async def needs_approval(run_context: RunContextWrapper[Any], params: dict[str, Any], call_id: str) -> bool:
            verdict = self.evaluate(tool_name, params).verdict
            if verdict is Verdict.DENY:
                return False  # the input guardrail rejects it before the tool runs; nobody is asked
            if verdict is Verdict.REVIEW:
                return self.on_review == "approve"  # with "deny" the input guardrail rejects it
            if isinstance(own, bool):
                return own
            decision = own(run_context, params, call_id)
            return bool(await decision) if inspect.isawaitable(decision) else bool(decision)

        needs_approval.__acgs_guard__ = self  # type: ignore[attr-defined]
        return needs_approval

    def _check(self, data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        tool_name = "unknown"
        try:
            context = data.context
            tool_name = context.tool_name
            if not isinstance(tool_name, str) or not tool_name:
                msg = f"Tool name must be a non-empty string, got {tool_name!r}."
                raise TypeError(msg)
            if context.tool_namespace:  # the name the SDK records approvals under, as FunctionTool.qualified_name
                tool_name = f"{context.tool_namespace}.{tool_name}"
            arguments = json.loads(context.tool_arguments or "{}")
            if not isinstance(arguments, dict):
                msg = f"Tool arguments must be a JSON object, got {type(arguments).__name__}."
                raise TypeError(msg)
            decision = self.evaluate(tool_name, arguments)
            approved = (
                context.is_tool_approved(tool_name, context.tool_call_id)
                if decision.verdict is Verdict.REVIEW
                else None
            )
        except Exception as error:  # fail-closed: the SDK would otherwise abort the whole run
            logger.warning("ACGS check failed for tool %r; rejecting the call (fail-closed).", tool_name, exc_info=True)
            return self._deny(
                str(tool_name),
                _EVALUATION_FAILED,
                {"verdict": Verdict.DENY.value, "reason": _EVALUATION_FAILED, "rule_ids": [], "error": repr(error)},
            )
        info = _info(decision)
        if decision.verdict is Verdict.ALLOW:
            return ToolGuardrailFunctionOutput.allow(output_info=info)
        if decision.verdict is Verdict.DENY:
            return self._deny(tool_name, decision.reason, info)
        if self.on_review == "deny":
            return self._deny(tool_name, f"Human review did not approve the call. {decision.reason}", info)
        if approved is True:
            logger.info("ACGS review of tool '%s' was approved.", tool_name)
            return ToolGuardrailFunctionOutput.allow(output_info=info)
        return self._deny(tool_name, f"The call needs human approval and was not approved. {decision.reason}", info)

    def _deny(self, tool_name: str, reason: str, info: dict[str, Any]) -> ToolGuardrailFunctionOutput:
        if self.on_deny == "raise":
            return ToolGuardrailFunctionOutput.raise_exception(output_info=info)
        return ToolGuardrailFunctionOutput.reject_content(f"Tool call '{tool_name}' was not executed. {reason}", info)


def _info(decision: GovernanceDecision) -> dict[str, Any]:
    return {
        "verdict": decision.verdict.value,
        "reason": decision.reason,
        "rule_ids": list(decision.rule_ids),
        "error": decision.error,
    }
