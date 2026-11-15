"""Run one allowed and one denied tool call offline, then print their governance decisions."""

import json
import os

# Disable tracing before importing the SDK; ScriptedModel makes no model service requests.
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

from agents import Agent, RunConfig, Runner, function_tool
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_acgs import GUARDRAIL_NAME, ACGSToolGuard


@function_tool
def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


def main() -> None:
    model = ScriptedModel(
        [
            [function_call("record_note", {"body": "The meeting starts at noon."}, call_id="note-1")],
            [function_call("record_note", {"body": "The admin password is demo-secret."}, call_id="note-2")],
            [assistant_message("Done.")],
        ]
    )
    guard = ACGSToolGuard()
    agent = guard.govern_agent(Agent(name="Note taker", model=model, tools=[record_note]))
    result = Runner.run_sync(agent, "Record the two notes.", run_config=RunConfig(tracing_disabled=True))
    print("Audit trail (application decisions):")
    # This agent has just one tool. Print its actual input guardrail results, without the arguments.
    for check in result.tool_input_guardrail_results:
        if check.guardrail.get_name() == GUARDRAIL_NAME:
            print(json.dumps({"tool_name": record_note.name, **check.output.output_info}, sort_keys=True))


if __name__ == "__main__":
    main()
