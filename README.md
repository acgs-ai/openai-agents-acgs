# openai-agents-acgs

[![CI](https://github.com/acgs-ai/openai-agents-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/openai-agents-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openai-agents-acgs.svg)](https://pypi.org/project/openai-agents-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/openai-agents-acgs.svg)](https://pypi.org/project/openai-agents-acgs/)

`openai-agents-acgs` checks [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) function tool calls
against [ACGS](https://pypi.org/project/acgs-lite/) constitution rules before the tool runs. It uses two documented
SDK mechanisms on each `FunctionTool`:

- a [tool input guardrail](https://openai.github.io/openai-agents-python/guardrails/#tool-guardrails), which the
  SDK runs right before the tool executes, including after a call was approved;
- a [`needs_approval`](https://openai.github.io/openai-agents-python/human_in_the_loop/) policy, so calls that need
  review pause the run as `ToolApprovalItem` interruptions that you approve or reject on the `RunState`.

What happens to a call:

- Calls that match no blocking rule run unchanged.
- Calls that match a blocking rule do not run. The guardrail rejects them
  (`ToolGuardrailFunctionOutput.reject_content`), so the model gets the reason as the tool result and can
  continue, or ends the run with `ToolInputGuardrailTripwireTriggered` if you pass `on_deny="raise"`.
- Calls that only match review rules pause the run for approval and run only once approved. They can also be
  rejected outright.
- A blocking rule always wins. Nobody is asked to approve a blocked call, and an approved call is checked again
  and rejected if it is blocked.
- Any error while evaluating a call rejects it (fail-closed).

The package is a thin adapter over the public `acgs-lite` API.

## Installation

```bash
pip install openai-agents-acgs
```

Requires Python 3.10 or later, `openai-agents>=0.22.1,<1` and `acgs-lite>=3.0.1,<5`.

acgs-lite 4 is a major release. Its production profile of `GovernedCallable` requires an Ed25519
receipt signer, governed payloads are JSON-only, and `GovernedAgent` no longer passes per-call keyword arguments to
rule conditions (see the [acgs-lite changelog](https://github.com/acgs-ai/acgs-lite/blob/main/CHANGELOG.md)). This
adapter works with both majors, but a fresh environment installs the newest version allowed. To stay on 3.x, also
require `"acgs-lite<4"`, for example `pip install openai-agents-acgs "acgs-lite<4"`.

## Usage

The SDK has no run-wide hook that can stop a tool call, so the guard is applied to each tool.
`govern_agent` applies it to every function tool of an agent:

```python
from agents import Agent, function_tool

from openai_agents_acgs import ACGSToolGuard


@function_tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


guard = ACGSToolGuard()
agent = guard.govern_agent(Agent(name="Assistant", tools=[send_email]))
```

`govern_agent` replaces `agent.tools` in place with governed copies (`govern_tool(tool)` does the same for one
tool and leaves the original unchanged), so handoffs that point at the agent see the governed tools. It raises
`UngovernedToolsError`, and leaves the agent unchanged, if the agent has tools the guard cannot check (see
[What is covered](#what-is-covered)); pass `allow_ungoverned=True` to accept them with a logged warning. Tools
added to `agent.tools` later are not governed until you call `govern_agent` again.

## Quickstart

`examples/quickstart.py` runs offline. The SDK's own
[`ScriptedModel`](https://openai.github.io/openai-agents-python/testing/) stands in for an LLM:

```python
"""Gate OpenAI Agents SDK tool calls with ACGS. Runs offline: the SDK's ScriptedModel stands in for an LLM."""

import json

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agents import Agent, RunConfig, Runner, RunResult, ToolCallOutputItem, function_tool
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_acgs import ACGSToolGuard

config = RunConfig(tracing_disabled=True)


@function_tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@function_tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} for {order_id}"


def print_tool_results(result: RunResult) -> None:
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem):
            print(f"{item.call_id}: {item.output}")


# 1. The default constitution blocks the email that leaks a password; the weather lookup runs.
model = ScriptedModel(
    [
        [
            function_call(
                "send_email", {"to": "ops@example.com", "body": "The admin password is hunter2"}, call_id="email-1"
            ),
            function_call("get_weather", {"city": "Paris"}, call_id="weather-1"),
        ],
        [assistant_message("Done.")],
    ]
)
guard = ACGSToolGuard()
agent = guard.govern_agent(Agent(name="Assistant", model=model, tools=[send_email, get_weather]))
print_tool_results(Runner.run_sync(agent, "Hi", run_config=config))

# 2. A review rule: refunds pause the run for approval, then run once approved.
refunds = Constitution.from_rules(
    [
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need human approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        )
    ],
    name="refunds",
)
model = ScriptedModel(
    [
        [function_call("issue_refund", {"order_id": "A-17", "amount": 20.0}, call_id="refund-1")],
        [assistant_message("Done.")],
    ]
)
guard = ACGSToolGuard(constitution=refunds)
agent = guard.govern_agent(Agent(name="Support", model=model, tools=[issue_refund]))
result = Runner.run_sync(agent, "Refund order A-17", run_config=config)
state = result.to_state()
for interruption in result.interruptions:
    decision = guard.evaluate(str(interruption.tool_name), json.loads(interruption.arguments or "{}"))
    print(f"paused: {decision.tool_name}: {decision.reason}")
    state.approve(interruption)  # a person approves the call

print_tool_results(Runner.run_sync(agent, state, run_config=config))
```

```text
email-1: Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
weather-1: Sunny in Paris
paused: issue_refund: Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval).
refund-1: refunded 20.0 for A-17
```

## How a tool call is evaluated

1. The tool name (`namespace.name` for namespaced tools) and the call's JSON arguments are rendered as text by
   `format_tool_call`:

   ```python
   from openai_agents_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The tool runs |
   | Only `require_human_review` / `escalate_to_senior` rules matched | Review (see below) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading or evaluating the tool call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.
   The denial reason then lists only the blocking rules after "Blocked by" and names the matched review
   rules separately.

A rejected call returns `"Tool call '<name>' was not executed. <reason>"` to the model as the tool result. The
reason names the matched rules but never repeats argument values. Every check is also recorded in
`result.tool_input_guardrail_results` under the guardrail name `acgs`, with the decision (`verdict`, `reason`,
`rule_ids`, `error`) in `output.output_info`.

`ACGSToolGuard.evaluate(tool_name, args)` returns the underlying `GovernanceDecision` without running anything.

### When the check runs

For a governed tool, the approval policy runs first, when the model emits the call:

| ACGS verdict | `needs_approval` returns |
|---|---|
| Blocked | `False`. Nobody is asked; the input guardrail rejects the call |
| Review (`on_review="approve"`) | `True`. The run pauses with a `ToolApprovalItem` |
| Review (`on_review="deny"`) | `False`. The input guardrail rejects the call |
| Allowed | The tool's own `needs_approval` (`bool` or callable), unchanged |

The input guardrail then runs immediately before the tool executes: in the same run for calls that need no
approval, and in the resumed run for approved calls. It evaluates the call again, so a call that is blocked
when it is about to run is rejected even if it was approved. A call that needs review runs only if the SDK has
recorded an approval for it (`RunContextWrapper.is_tool_approved`); a rejected call never reaches the guardrail,
and the model gets the SDK's rejection message. The guard runs before the tool's own input guardrails, and
`RunHooks.on_tool_start` / `AgentHooks.on_tool_start` only run for calls the guardrails let through.

## Review

```python
from typing import Any

from agents import Agent, Runner


async def run_with_review(agent: Agent[Any], prompt: str) -> Any:
    result = await Runner.run(agent, prompt)
    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            state.approve(interruption)  # or state.reject(interruption, rejection_message="...")
        result = await Runner.run(agent, state)
    return result.final_output
```

This is the SDK's own [approval flow](https://openai.github.io/openai-agents-python/human_in_the_loop/): it works
the same for streamed runs (`Runner.run_streamed`) and for states saved with `state.to_string()` and restored with
`RunState.from_string(agent, ...)`. `state.approve(interruption, always_approve=True)` approves later review calls
of the same tool in that run without pausing again; blocked calls are still rejected. Interruptions do not carry
the ACGS decision; call `guard.evaluate(interruption.tool_name, json.loads(interruption.arguments))` to show the
reason to a reviewer, as the quickstart does.

Use `on_review="deny"` for runs nobody can approve. Leave `RunConfig.tool_execution.pre_approval_tool_input_guardrails`
off: when it is on, the SDK runs the input guardrail before the approval is requested, and the guard rejects calls
that need review because they are not approved yet. Blocked calls are never sent for approval without it.

## What is covered

| Tool | Checked | How |
|---|---|---|
| `FunctionTool` (`@function_tool`, `FunctionTool(...)`, `tool_namespace(...)`) | Yes | `govern_agent` / `govern_tool` |
| `Agent.as_tool(...)` | The call to the agent tool, yes | It is a `FunctionTool`; a blocked call never starts the inner agent |
| Tools of the agent behind `Agent.as_tool` | Only if governed | Call `govern_agent` on that agent too |
| Handoffs | The handoff itself, no | Handoffs do not use the tool pipeline. Call `govern_agent` on every agent a run can hand off to |
| Local MCP servers (`MCPServerStdio`, `MCPServerSse`, `MCPServerStreamableHttp`) | Yes, if given the guardrail | `tool_input_guardrails=[guard.input_guardrail]` on the server |
| Hosted tools (`WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `HostedMCPTool`, `ToolSearchTool`) | No | The provider runs them; `govern_agent` refuses them |
| `ComputerTool`, `ShellTool`, `LocalShellTool`, `ApplyPatchTool`, `CustomTool`, `ProgrammaticToolCallingTool` | No | They do not use the tool guardrail pipeline; `govern_agent` refuses them |

For local MCP servers, pass the guardrail to the server; `govern_agent` refuses an agent whose MCP servers do not
have it:

```python
from agents import Agent
from agents.mcp import MCPServerStdio

from openai_agents_acgs import ACGSToolGuard

guard = ACGSToolGuard()
server = MCPServerStdio(
    params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
    tool_input_guardrails=[guard.input_guardrail],
    require_approval="always",  # optional: lets review calls run once approved
)
agent = guard.govern_agent(Agent(name="Files", mcp_servers=[server]))
```

MCP tools get the guardrail but not the approval policy: the SDK's `require_approval` callable for MCP servers
receives the tool, not the call's arguments. Without `require_approval`, MCP calls that need review are rejected.
With `require_approval="always"`, every call is sent for approval, review calls run once approved, and blocked
calls are rejected after approval.

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from openai_agents_acgs import ACGSToolGuard

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
guard = ACGSToolGuard(constitution=constitution, agent_id="payments-app")
```

`Constitution.from_yaml("rules.yaml")` works as well. To enforce two constitutions, govern the tools with two
guards; a call runs only if both allow it.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"openai-agents"` | Agent identifier recorded by ACGS |
| `on_review` | `"approve"` | `"approve"` (pause for approval) or `"deny"` |
| `on_deny` | `"reject"` | `"reject"` (tool result with the reason) or `"raise"` (`ToolInputGuardrailTripwireTriggered`) |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |

## Limitations

- Only tools listed as checked in [What is covered](#what-is-covered). User input, model output, and tool results
  are not checked; the SDK's agent-level input and output guardrails and tool output guardrails remain available
  for that.
- The guard is applied per tool and per agent. A tool or agent it was not applied to (a handoff target, the agent
  behind `Agent.as_tool`, a tool added after `govern_agent`) is not checked.
- Tested with `Runner.run`, `Runner.run_sync`, `Runner.run_streamed`, parallel tool calls in one model turn,
  approval, rejection and resume (including serialized `RunState`), `always_approve`, tools with their own
  `needs_approval`, namespaced tools, `Agent.as_tool`, handoffs, an in-memory local MCP server, and `RunHooks`, against
  openai-agents 0.22.1 and the latest release. Realtime and voice sessions, sandbox agents, and durable
  execution integrations are not tested.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied under
  the default constitution whenever it is called.
- The OpenAI Agents SDK is pre-1.0 and its minor releases can contain breaking changes. The adapter allows
  `openai-agents<1` and is tested against the oldest supported and the latest release.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against openai-agents 0.22.1 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
