# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import openai_agents_acgs


def test_public_api() -> None:
    assert sorted(openai_agents_acgs.__all__) == sorted(
        [
            "GUARDRAIL_NAME",
            "ACGSToolGuard",
            "ApprovalPolicy",
            "GovernanceDecision",
            "ToolCallFormatter",
            "UngovernedToolsError",
            "Verdict",
            "format_tool_call",
        ]
    )
    for name in openai_agents_acgs.__all__:
        assert getattr(openai_agents_acgs, name) is not None
