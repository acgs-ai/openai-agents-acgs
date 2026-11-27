# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for hidden control characters and slow identifier matching in tool-call arguments."""

from __future__ import annotations

import time

import pytest
from acgs_lite import Constitution, Rule, Severity

from openai_agents_acgs._governance import _IDENTIFIER_VALUE, ToolCallGovernor, Verdict, format_tool_call

BLOCK = Rule(id="T-BLOCK", text="no deletes", severity=Severity.CRITICAL, keywords=["delete everything"])
SKIP = Rule(id="T-SKIP", text="no skipped audits", severity=Severity.CRITICAL, keywords=["skip audit"])

# Control characters that ``str.split()`` treats as whitespace.
WHITESPACE_CONTROLS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85"]
# Control characters that ``str.split()`` keeps.
OTHER_CONTROLS = ["\x00", "\x08", "\x7f", "\x9f"]


@pytest.fixture
def governor() -> ToolCallGovernor:
    return ToolCallGovernor(Constitution.from_rules([BLOCK, SKIP], name="test"), agent_id="agent")


class _Stringly:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


class TestHiddenControlCharacters:
    @pytest.mark.parametrize("hidden", WHITESPACE_CONTROLS + OTHER_CONTROLS)
    def test_control_inside_a_keyword_does_not_hide_it(self, governor: ToolCallGovernor, hidden: str) -> None:
        decision = governor.evaluate("execute", {"value": f"del{hidden}ete everything"})
        assert decision.verdict is Verdict.DENY
        assert decision.rule_ids == ("T-BLOCK",)

    @pytest.mark.parametrize("hidden", WHITESPACE_CONTROLS)
    def test_control_used_as_a_word_separator_is_still_denied(self, governor: ToolCallGovernor, hidden: str) -> None:
        assert governor.evaluate("execute", {"value": f"delete{hidden}everything"}).verdict is Verdict.DENY

    @pytest.mark.parametrize(
        "value",
        [
            "\x85delete\x85 e\x7fvery\x00thing",
            "del\x1cete\x0b\x0c every\x1fthing",
            "d\x00e\x0bl\x0ce\x1ct\x1de\n\x1eeverything\x1f",
            # One kind of control inside the word and another kind between the words.
            "del\x00ete\x0beverything",
            "del\x7fete\x0ceverything",
            "\x9fdel\x1bete\x85every\x08thing",
            # The whitespace control inside a word and another control between the words.
            "del\x0bete\x00everything",
            "delete\x00every\x0bthing",
            "del\x85ete\x1b\x7feverything",
        ],
    )
    def test_mixed_controls_do_not_hide_a_keyword(self, governor: ToolCallGovernor, value: str) -> None:
        assert governor.evaluate("execute", {"value": value}).verdict is Verdict.DENY

    def test_control_in_a_nested_value_does_not_hide_a_keyword(self, governor: ToolCallGovernor) -> None:
        decision = governor.evaluate("execute", {"steps": [{"command": "del\x0bete everything"}]})
        assert decision.verdict is Verdict.DENY

    def test_control_in_a_non_string_value_does_not_hide_a_keyword(self, governor: ToolCallGovernor) -> None:
        decision = governor.evaluate("execute", {"value": _Stringly("del\x0bete everything")})
        assert decision.verdict is Verdict.DENY

    def test_value_is_rendered_a_second_time_without_the_controls(self) -> None:
        text = format_tool_call("execute", {"value": "del\x0bete everything"})
        assert text.splitlines() == [
            "tool: execute",
            "arguments:",
            "value: del ete everything",
            "value: delete everything",
        ]

    def test_each_distinct_reading_is_rendered_once(self) -> None:
        text = format_tool_call("execute", {"value": "del\x00ete\x0beverything"})
        assert text.splitlines() == [
            "tool: execute",
            "arguments:",
            "value: del\x00ete everything",
            "value: delete everything",
            "value: deleteeverything",
            "value: del ete everything",
            "value: del eteeverything",
        ]
        text = format_tool_call("execute", {"value": "del\x00ete everything"})
        assert text.splitlines()[2:] == [
            "value: del\x00ete everything",
            "value: delete everything",
            "value: del ete everything",
        ]

    @pytest.mark.parametrize("separator", OTHER_CONTROLS)
    def test_non_whitespace_control_used_as_a_word_separator_is_denied(
        self, governor: ToolCallGovernor, separator: str
    ) -> None:
        assert governor.evaluate("execute", {"value": f"delete{separator}everything"}).verdict is Verdict.DENY
        assert governor.evaluate("configure", {"mode": f"skip{separator}audit"}).verdict is Verdict.DENY

    @pytest.mark.parametrize("value", ["skip_\x0baudit", "skip\x00_audit", "skip\x00Audit", "skip\x1f-\x7faudit"])
    def test_control_inside_an_identifier_value_does_not_hide_it(self, governor: ToolCallGovernor, value: str) -> None:
        decision = governor.evaluate("configure", {"mode": value})
        assert decision.verdict is Verdict.DENY
        assert decision.rule_ids == ("T-SKIP",)

    def test_identifier_reading_is_added_to_the_normalized_line(self) -> None:
        text = format_tool_call("configure", {"mode": "auto\x00_approve"})
        assert text.splitlines()[-1] == "normalized: auto approve"

    @pytest.mark.parametrize(
        "value",
        [
            "del\x0bete\x0ceverything",
            "del\x0bete\x0beverything",
            "del\x1cete\x85every\x1fthing",
            "page one\x0cpage two\x0cpage three",
        ],
    )
    def test_value_with_several_whitespace_controls_is_denied(self, governor: ToolCallGovernor, value: str) -> None:
        # Each such character could be a separator or sit inside a word, so no rendering is trustworthy.
        decision = governor.evaluate("execute", {"steps": [{"command": value}]})
        assert decision.verdict is Verdict.DENY
        assert decision.error is not None
        with pytest.raises(ValueError, match="more than one whitespace control character"):
            format_tool_call("execute", {"value": value})

    def test_value_with_one_whitespace_control_and_no_rule_match_is_allowed(self, governor: ToolCallGovernor) -> None:
        decision = governor.evaluate("write_file", {"content": "page one\x0cpage two"})
        assert decision.verdict is Verdict.ALLOW

    def test_values_without_controls_are_rendered_once(self, governor: ToolCallGovernor) -> None:
        arguments = {"value": "delete\n  nothing\tnow", "n": 3, "flag": True, "none": None}
        assert format_tool_call("execute", arguments).splitlines() == [
            "tool: execute",
            "arguments:",
            "flag: true",
            "n: 3",
            "none: null",
            "value: delete nothing now",
        ]
        assert governor.evaluate("execute", arguments).verdict is Verdict.ALLOW


class TestIdentifierPattern:
    @pytest.mark.parametrize("value", ["auto_approve", "skip-audit", "a_b_c", "disableLogging", "x1_y2", "aBc"])
    def test_accepts_single_token_identifiers(self, value: str) -> None:
        assert _IDENTIFIER_VALUE.match(value)

    @pytest.mark.parametrize(
        "value", ["", "plain", "two words", "_leading", "trailing_", "a__b", "1abc_def", "a.b", "aB"]
    )
    def test_rejects_everything_else(self, value: str) -> None:
        assert _IDENTIFIER_VALUE.match(value) is None

    def test_does_not_backtrack_exponentially(self) -> None:
        # The previous pattern roughly doubled its time with every repetition of this input.
        hostile = "A-" + "0A0-" * 25
        started = time.perf_counter()
        assert _IDENTIFIER_VALUE.match(hostile) is None
        assert time.perf_counter() - started < 1.0
