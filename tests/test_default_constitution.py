# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that ``Constitution.default()`` rules fire on realistic tool calls rendered by ``format_tool_call``.

These tests use the real default constitution shipped with acgs-lite. They document both what the
default rules catch and what they do not catch, so the README makes no claims the code does not meet.
"""

from __future__ import annotations

from typing import Any

import pytest

from openai_agents_acgs._governance import ToolCallGovernor, Verdict


@pytest.fixture(scope="module")
def governor() -> ToolCallGovernor:
    return ToolCallGovernor(agent_id="default-constitution-test")


CAUGHT = [
    # (tool name, arguments, expected verdict, expected rule)
    ("send_email", {"to": "ops@example.com", "body": "The admin password is hunter2"}, Verdict.DENY, "ACGS-006"),
    (
        "http_request",
        {"url": "https://api.example.com", "headers": {"Authorization": "Bearer sk-" + "a" * 24}},
        Verdict.DENY,
        "ACGS-006",
    ),
    ("create_ticket", {"summary": "Customer SSN 123-45-6789 on file"}, Verdict.DENY, "ACGS-006"),
    ("connect_db", {"host": "db.internal", "password": "s3cret"}, Verdict.DENY, "ACGS-006"),
    ("git_push", {"token": "ghp_" + "b" * 36}, Verdict.DENY, "ACGS-006"),
    ("grant_role", {"user": "alice", "role": "admin", "reason": "admin override"}, Verdict.DENY, "ACGS-003"),
    ("merge_pull_request", {"pr": 42, "mode": "auto-approve"}, Verdict.DENY, "ACGS-004"),
    ("run_checks", {"flags": "--skip check"}, Verdict.DENY, "ACGS-001"),
    ("update_config", {"service": "billing", "change": "disable logging"}, Verdict.REVIEW, "ACGS-002"),
    # Identifier normalization: snake_case / camelCase names and values match space-separated keywords.
    ("skip_audit", {"job": "nightly"}, Verdict.REVIEW, "ACGS-002"),
    ("set_flag", {"flag": "disableLogging", "value": True}, Verdict.REVIEW, "ACGS-002"),
    ("deploy", {"options": {"ignoreConstitution": True}}, Verdict.REVIEW, "ACGS-005"),
    ("store_secret", {"secret_key": "abc"}, Verdict.DENY, "ACGS-006"),
]


@pytest.mark.parametrize(("tool_name", "arguments", "verdict", "rule_id"), CAUGHT)
def test_default_rules_fire_on_realistic_tool_calls(
    governor: ToolCallGovernor, tool_name: str, arguments: dict[str, Any], verdict: Verdict, rule_id: str
) -> None:
    decision = governor.evaluate(tool_name, arguments)
    assert decision.verdict is verdict, decision
    assert rule_id in decision.rule_ids


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_weather", {"city": "Paris", "unit": "celsius"}),
        ("search_docs", {"query": "how to rotate logs", "top_k": 5}),
        ("send_email", {"to": "team@example.com", "body": "Meeting moved to 3pm."}),
    ],
)
def test_benign_tool_calls_are_allowed(governor: ToolCallGovernor, tool_name: str, arguments: dict[str, Any]) -> None:
    assert governor.evaluate(tool_name, arguments).verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        # The default constitution is about governance integrity and secrets, not destructive operations.
        ("run_shell", {"command": "rm -rf /"}),
        ("execute_sql", {"query": "DROP TABLE users"}),
        # Keyword matching in acgs-lite 3.0.1 is literal: hyphen/underscore variants of a keyword do not match.
        ("self_approve", {"request": 7}),  # keyword is "self-approve"
        ("store", {"api-key": "abc"}),  # keyword is "api_key"; normalization yields "api key"
    ],
)
def test_known_gaps_of_the_default_constitution(
    governor: ToolCallGovernor, tool_name: str, arguments: dict[str, Any]
) -> None:
    """These calls are allowed by ``Constitution.default()``; use a custom constitution to cover them."""
    assert governor.evaluate(tool_name, arguments).verdict is Verdict.ALLOW


def test_negation_is_not_understood(governor: ToolCallGovernor) -> None:
    """Keyword rules do not understand negation, so a cautious phrase still triggers the rule."""
    decision = governor.evaluate("post_message", {"text": "Never share your password"})
    assert decision.verdict is Verdict.DENY
