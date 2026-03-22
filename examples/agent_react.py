"""ReAct Agent — Thought → Action → Observation → Answer reasoning loop."""

from zerollm import Agent

# ReAct mode: the agent thinks step-by-step before acting
agent = Agent("Qwen/Qwen3.5-4B", react=True)


@agent.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


@agent.tool
def get_price(item: str) -> str:
    """Get the price of an item."""
    prices = {"laptop": 999, "phone": 699, "headphones": 149}
    return f"${prices.get(item, 'unknown')}"


# The agent will:
# 1. Thought: I need to find the price first
# 2. Action: get_price("laptop")
# 3. Observation: $999
# 4. Thought: Now calculate 15% discount
# 5. Action: calculate("999 * 0.85")
# 6. Observation: 849.15
# 7. Answer: The laptop with 15% discount costs $849.15
print(agent.ask("What is the price of a laptop with 15% discount?"))
