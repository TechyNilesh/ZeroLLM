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
