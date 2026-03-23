"""Tests for memory management."""

from zerollm.memory import Memory


def test_session_memory():
    mem = Memory()
    mem.add("user", "Hello")
    mem.add("assistant", "Hi there")

    assert mem.turn_count == 1
    assert len(mem.get_context()) == 2


def test_system_prompt():
    mem = Memory()
    mem.add_system("You are helpful.")
    mem.add("user", "Hello")

    context = mem.get_context()
    assert context[0]["role"] == "system"
    assert context[1]["role"] == "user"


def test_context_limit():
    mem = Memory()
    for i in range(50):
        mem.add("user", f"Message {i}")
        mem.add("assistant", f"Reply {i}")

    context = mem.get_context(max_messages=10)
    assert len(context) == 10


def test_context_limit_preserves_system():
    mem = Memory()
    mem.add_system("System prompt")
    for i in range(50):
        mem.add("user", f"Message {i}")
        mem.add("assistant", f"Reply {i}")

    context = mem.get_context(max_messages=10)
    assert context[0]["role"] == "system"
    assert len(context) == 11  # system + 10 recent


def test_clear_preserves_system():
    mem = Memory()
    mem.add_system("System prompt")
    mem.add("user", "Hello")
    mem.clear()

    assert len(mem.messages) == 1
    assert mem.messages[0]["role"] == "system"


def test_clear_all():
    mem = Memory()
    mem.add_system("System prompt")
    mem.add("user", "Hello")
    mem.clear_all()

    assert len(mem.messages) == 0


# ── Auto-summarization tests ──

def test_maybe_summarize_short_history():
    """No summarization when history is short."""
    mem = Memory(summarize_after=10)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi")
    assert mem.maybe_summarize() is False


def test_maybe_summarize_no_backend():
    """No summarization without a backend."""
    mem = Memory(summarize_after=4)
    for i in range(10):
        mem.add("user", f"Message {i}")
        mem.add("assistant", f"Reply {i}")
    assert mem.maybe_summarize(backend=None) is False


def test_has_summaries_false():
    mem = Memory()
    assert mem.has_summaries is False


def test_context_includes_summaries():
    mem = Memory()
    mem._summaries = ["User's name is Nilesh. Works on AI."]
    mem.add_system("You are helpful.")
    mem.add("user", "What is my name?")

    context = mem.get_context()
    # Should have: system + summary + user message
    assert len(context) == 3
    assert "Summary" in context[1]["content"]
    assert "Nilesh" in context[1]["content"]


def test_turn_count():
    mem = Memory()
    mem.add("user", "Q1")
    mem.add("assistant", "A1")
    mem.add("user", "Q2")
    assert mem.turn_count == 2
