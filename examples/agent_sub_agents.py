"""Sub-agents with shared context — delegate tasks to specialized agents."""

from zerollm import Agent, SharedContext

# All agents share one context
ctx = SharedContext()

# Research agent
researcher = Agent(
    "Qwen/Qwen3.5-4B",
    name="researcher",
    context=ctx,
    system_prompt="You are a research assistant. Find and summarize information.",
)

@researcher.tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles."

# Writer agent
writer = Agent(
    "Qwen/Qwen3.5-4B",
    name="writer",
    context=ctx,
    system_prompt="You are a writer. Write clear, engaging content.",
)

# Main orchestrator
main = Agent(
    "Qwen/Qwen3.5-4B",
    name="orchestrator",
    context=ctx,
    system_prompt=(
        "You are a project manager. You have a 'researcher' for finding "
        "information and a 'writer' for composing text. Delegate tasks."
    ),
)

main.add_agent("researcher", researcher, "Research any topic")
main.add_agent("writer", writer, "Write content")

# Main agent delegates — sub-agents see each other's results via shared context
print(main.ask("Research local LLMs and write a summary"))
