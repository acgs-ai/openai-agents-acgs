# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.1.0 (unreleased)

### Added

- `ACGSToolGuard`, which checks OpenAI Agents SDK function tool calls against an ACGS constitution with two
  documented per-tool mechanisms: a tool input guardrail (`input_guardrail`, named `acgs`) that runs right before
  the tool executes, and a `needs_approval` policy that pauses the run with a `ToolApprovalItem` when ACGS requires
  review.
- `govern_agent(agent)`, which applies the guard to every `FunctionTool` of an agent in place, and raises
  `UngovernedToolsError` for hosted tools, built-in execution tools and local MCP servers without the guardrail
  (`allow_ungoverned=True` logs a warning instead). `govern_tool(tool)` returns a governed copy of one tool.
- Blocked calls are never sent for approval and are rejected with `reject_content`, or end the run with
  `on_deny="raise"`. Approved calls are checked again before they run, and a blocking rule still rejects them.
- `on_review` (`"approve"` or `"deny"`) for calls that only match review rules. A tool's own `needs_approval`
  still applies to allowed calls.
- Local MCP server support through `tool_input_guardrails=[guard.input_guardrail]` (openai-agents 0.22.1 and later).
- Fail-closed handling: any error while evaluating a call rejects it instead of ending the run.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
