# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tools that record whether they ran, an in-memory MCP server, and helpers around ``agents.testing``."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agents import Agent, FunctionTool, RunConfig, function_tool
from agents.mcp import MCPServer
from agents.testing import ScriptedModel
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

T = TypeVar("T")

LEAK = {"to": "ops@example.com", "body": "The admin password is hunter2"}
HELLO = {"to": "team@example.com", "body": "Meeting moved to 3pm."}
REFUND = {"order_id": "A-17", "amount": 20.0}

RUN_CONFIG = RunConfig(tracing_disabled=True)

REFUND_REVIEW = Rule(
    id="REFUND-REVIEW",
    text="Refunds need human approval",
    severity=Severity.HIGH,
    keywords=["refund"],
    workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
)


def review_rules(*, with_defaults: bool = True) -> Constitution:
    """The refund review rule, by default alongside the default rules."""
    rules = [REFUND_REVIEW, *(Constitution.default().rules if with_defaults else [])]
    return Constitution.from_rules(rules, name="refunds")


@dataclass
class Tools:
    """``send_email`` (sync), ``issue_refund`` (async) and ``delete_row`` (``needs_approval=True``)."""

    ran: list[str] = field(default_factory=list)

    def send_email(self, *, needs_approval: Any = False) -> FunctionTool:
        @function_tool(needs_approval=needs_approval)
        def send_email(to: str, body: str) -> str:
            """Send an email."""
            self.ran.append("send_email")
            return f"sent to {to}"

        return send_email

    def issue_refund(self, *, needs_approval: Any = False) -> FunctionTool:
        @function_tool(needs_approval=needs_approval)
        async def issue_refund(order_id: str, amount: float) -> str:
            """Refund an order."""
            self.ran.append("issue_refund")
            return f"refunded {amount} for {order_id}"

        return issue_refund

    def delete_row(self) -> FunctionTool:
        @function_tool(needs_approval=True)
        def delete_row(row: int) -> str:
            """Delete a row."""
            self.ran.append("delete_row")
            return f"deleted {row}"

        return delete_row

    def all(self) -> list[FunctionTool]:
        return [self.send_email(), self.issue_refund(), self.delete_row()]


def agent(model: ScriptedModel, tools: list[Any], **kwargs: Any) -> Agent[Any]:
    return Agent(name="assistant", model=model, tools=tools, **kwargs)


def outputs(model: ScriptedModel) -> dict[str, Any]:
    """The tool results the model received in its last call, by call ID."""
    call = model.last_call
    assert call is not None
    return {
        item["call_id"]: item["output"]
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }


class MemoryMCPServer(MCPServer):
    """A local MCP server with one ``post_note`` tool, kept in memory."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any] | None] = []

    async def connect(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "notes"

    async def cleanup(self) -> None:
        return None

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[MCPTool]:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        return [MCPTool.model_validate({"name": "post_note", "inputSchema": schema})]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None
    ) -> CallToolResult:
        self.calls.append(arguments)
        return CallToolResult(content=[TextContent(type="text", text="posted")])

    async def list_prompts(self) -> Any:
        raise NotImplementedError

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError


def in_own_thread(function: Callable[[], T]) -> T:
    """Run synchronous agent code (``Runner.run_sync``) on a fresh thread with its own event loop, then close it.

    ``run_sync`` drives an event loop on the calling thread. A separate thread keeps that loop away from
    pytest-asyncio's loops and lets the test close it.
    """
    results: list[T] = []
    errors: list[BaseException] = []

    def target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results.append(function())
        except BaseException as error:  # noqa: BLE001  # re-raised in the calling thread
            errors.append(error)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return results[0]
