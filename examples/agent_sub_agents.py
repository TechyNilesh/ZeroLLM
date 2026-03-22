"""Sub-agents — delegate tasks to specialized agents.

Main agent orchestrates, sub-agents handle specific domains.
Each sub-agent can have its own model, tools, and system prompt.
"""

from zerollm import Agent

# ── Create specialized sub-agents ──

# Research agent — good at finding information
researcher = Agent(
    "Qwen/Qwen3-1.7B",
    name="researcher",
    system_prompt="You are a research assistant. Find and summarize information accurately.",
)

@researcher.tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles about {query}."

@researcher.tool
def read_url(url: str) -> str:
    """Read the contents of a URL."""
    return f"Contents of {url}: [article text here]"


# Writer agent — good at composing text
writer = Agent(
    "Qwen/Qwen3.5-4B",
    name="writer",
    system_prompt="You are a skilled writer. Write clear, engaging content.",
)

@writer.tool
def save_draft(title: str, content: str) -> str:
    """Save a draft document."""
    print(f"  [Draft saved: {title}]")
    return f"Draft '{title}' saved successfully."


# ── Create the main orchestrator agent ──

main = Agent(
    "Qwen/Qwen3.5-4B",
    name="orchestrator",
    system_prompt=(
        "You are a project manager agent. You have two sub-agents: "
        "'researcher' for finding information and 'writer' for composing text. "
        "Delegate tasks to the appropriate sub-agent."
    ),
)

# Register sub-agents — they become tools the main agent can call
main.add_agent("researcher", researcher, "Research any topic — searches the web and reads URLs")
main.add_agent("writer", writer, "Write content — composes text and saves drafts")

# The main agent can also have its own tools
@main.tool
def get_date() -> str:
    """Get today's date."""
    from datetime import date
    return str(date.today())


# ── Use the orchestrator ──

# Main agent decides which sub-agent to delegate to
print(main.ask("Research the latest trends in local LLMs"))
print(main.ask("Write a blog post about what you found"))
print(main.ask("What is today's date?"))  # uses its own tool

# Interactive mode
# main.chat()
