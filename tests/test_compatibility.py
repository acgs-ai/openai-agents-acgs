"""Regression checks for the adapter boundary with acgs-lite 4.x and the supported 3.x floor."""

from typing import Any

import pytest

from openai_agents_acgs import ACGSToolGuard, Verdict, format_tool_call


@pytest.mark.parametrize("binary_type", [bytes, bytearray])
@pytest.mark.parametrize("body", [b"Meeting at noon", b"password"])
def test_binary_arguments_remain_scalar_and_match_rules(binary_type: Any, body: bytes) -> None:
    arguments = {"items": [{"body": binary_type(body)}]}
    rendered = format_tool_call("record_note", arguments)
    assert f"items[0].body: {binary_type(body)}" in rendered
    assert "body[0]" not in rendered
    decision = ACGSToolGuard().evaluate("record_note", arguments)
    assert decision.verdict is (Verdict.DENY if body == b"password" else Verdict.ALLOW)
    assert decision.error is None
    assert decision.rule_ids == (("ACGS-006",) if body == b"password" else ())


@pytest.mark.parametrize("body", ["Meeting at noon", "password"])
def test_nested_json_arguments_with_default_unsigned_configuration(body: str) -> None:
    arguments: dict[str, Any] = {"items": [{"body": body, "values": [None, True, False, 7, 1.5, {}, []]}]}
    rendered = format_tool_call("record_note", arguments)
    for index, value in enumerate(["null", "true", "false", "7", "1.5", "{}", "[]"]):
        assert f"items[0].values[{index}]: {value}" in rendered
    decision = ACGSToolGuard().evaluate("record_note", arguments)
    assert decision.verdict is (Verdict.DENY if body == "password" else Verdict.ALLOW)
    assert decision.error is None
    assert decision.rule_ids == (("ACGS-006",) if body == "password" else ())
