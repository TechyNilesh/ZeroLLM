"""Human-in-the-loop — ask for confirmation before dangerous tool calls."""

from zerollm import Agent

agent = Agent("Qwen/Qwen3.5-4B")


# Safe tool — runs without asking
@agent.tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


# Dangerous tool — asks for human confirmation before running
@agent.tool(confirm=True)
def delete_file(path: str) -> str:
    """Delete a file from disk."""
    # This will prompt: "Confirm: Call delete_file({"path": "/tmp/old.txt"})? [y/N]"
    import os
    os.remove(path)
    return f"Deleted {path}"


@agent.tool(confirm=True)
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    # User must confirm before the email is sent
    return f"Email sent to {to}: {subject}"


# Safe tools run automatically, dangerous ones ask first
agent.ask("Search for Python tutorials")        # runs immediately
agent.ask("Delete /tmp/old.txt")                 # asks: Confirm? [y/N]
agent.ask("Send an email to boss@company.com")   # asks: Confirm? [y/N]
