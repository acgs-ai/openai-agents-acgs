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
