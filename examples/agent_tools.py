"""Agent example — tool-calling with local LLMs."""

from zerollm import Agent

agent = Agent("Qwen/Qwen3-0.6B")


@agent.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"


@agent.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # noqa: S307 — example only
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Agent auto-picks and calls the right tool
print(agent.ask("What is the weather in Auckland?"))
print(agent.ask("What is 42 * 17?"))
