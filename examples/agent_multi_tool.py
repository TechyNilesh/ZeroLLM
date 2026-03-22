"""Multi-tool agent — give the agent multiple tools and let it decide."""

from zerollm import Agent

agent = Agent(
    "Qwen/Qwen3.5-4B",
    system_prompt="You are a helpful assistant with access to weather, math, and note tools.",
)


@agent.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "Auckland": "18°C, partly cloudy",
        "London": "12°C, rainy",
        "Tokyo": "25°C, sunny",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@agent.tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 3 * 4'"""
    try:
        result = eval(expression)  # noqa: S307 — example only
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@agent.tool
def save_note(text: str) -> str:
    """Save a note for later."""
    print(f"  [Note saved: {text}]")
    return f"Note saved: {text}"


# Let the agent decide which tool to use
print(agent.ask("What's the weather like in Tokyo?"))
print(agent.ask("What is 15 * 7 + 23?"))
print(agent.ask("Save a note: buy milk tomorrow"))

# Interactive chat with all tools available
# agent.chat()
