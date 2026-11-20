# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The README only shows code that runs, and the quickstart output it shows is the real output."""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from agents import Agent
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_acgs import ACGSToolGuard
from tests.fakes import REFUND, Tools, in_own_thread, review_rules

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(block) for block in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    env = {**os.environ, "OPENAI_AGENTS_DISABLE_TRACING": "1"}
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(QUICKSTART)], capture_output=True, text=True, check=True, cwd=ROOT, env=env, timeout=300
    )
    [expected] = [block for block in _blocks("text") if block.startswith("email-1:")]
    assert completed.stdout == expected


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_deny_and_audit_runs_offline_without_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in list(os.environ):
        if key.endswith("API_KEY"):
            monkeypatch.delenv(key)
    # Register the example's import-time change so monkeypatch restores the original environment.
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    # pytest-socket disables network access for this in-process execution.
    in_own_thread(lambda: runpy.run_path(str(ROOT / "examples" / "deny_and_audit.py"), run_name="__main__"))
    output = capsys.readouterr().out
    execution, audit = output.split("Audit trail (application decisions):\n")
    assert execution == "Executed record_note: The meeting starts at noon.\n"
    decisions = [json.loads(line) for line in audit.splitlines()]
    assert [decision["verdict"] for decision in decisions] == ["allow", "deny"]
    assert [decision["tool_name"] for decision in decisions] == ["record_note", "record_note"]
    assert decisions[0]["rule_ids"] == []
    assert decisions[1]["rule_ids"] == ["ACGS-006"]
    assert "Blocked by" in decisions[1]["reason"]
    assert all(decision["error"] is None for decision in decisions)
    assert "demo-secret" not in output


def test_readme_states_the_supported_versions() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    [requirement] = re.findall(r'"(openai-agents>=[^"]+)"', pyproject)
    assert f"`{requirement}`" in README
    [floor] = re.findall(r'"openai-agents==([^"]+)"', pyproject)
    assert requirement.startswith(f"openai-agents>={floor},")
    assert README.count(f"openai-agents {floor}") == 2


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 6
    namespaces: list[dict[str, Any]] = []

    def run_blocks() -> None:
        for block in blocks:
            namespace: dict[str, Any] = {"__name__": "readme"}
            exec(compile(block, "<README>", "exec", dont_inherit=True), namespace)  # noqa: S102
            namespaces.append(namespace)

    in_own_thread(run_blocks)
    output = capsys.readouterr().out
    assert "normalized: update config | dry run" in output
    assert "paused: issue_refund: Human review required by ACGS rule REFUND-REVIEW" in output
    usage, mcp = namespaces[0], namespaces[4]
    assert usage["guard"].is_governing(usage["agent"].tools[0])
    assert mcp["agent"].mcp_servers == [mcp["server"]]


async def test_readme_review_loop_approves_and_resumes() -> None:
    [block] = [block for block in _blocks("python") if "async def run_with_review" in block]
    namespace: dict[str, Any] = {"__name__": "readme"}
    exec(compile(block, "<README>", "exec", dont_inherit=True), namespace)  # noqa: S102
    tools = Tools()
    model = ScriptedModel([[function_call("issue_refund", REFUND, call_id="r1")], [assistant_message("Done.")]])
    agent = ACGSToolGuard(constitution=review_rules()).govern_agent(
        Agent(name="Support", model=model, tools=[tools.issue_refund()])
    )
    assert await namespace["run_with_review"](agent, "Refund order A-17") == "Done."
    assert tools.ran == ["issue_refund"]
