"""Tests for the backend module — think tag stripping (no real model needed)."""

from zerollm.backend import _strip_think_tags


def test_strip_think_tags():
    assert _strip_think_tags("<think>reasoning</think>Answer") == "Answer"


def test_strip_reasoning_tags():
    assert _strip_think_tags("<reasoning>step 1</reasoning>42") == "42"


def test_strip_thinking_tags():
    assert _strip_think_tags("<thinking>hmm</thinking>yes") == "yes"


def test_strip_thought_tags():
    assert _strip_think_tags("<thought>deep</thought>result") == "result"


def test_strip_reflection_tags():
    assert _strip_think_tags("<reflection>check</reflection>done") == "done"


def test_strip_multiline_think():
    text = "<think>\nLet me think about this...\nStep 1\nStep 2\n</think>\n\n4"
    assert _strip_think_tags(text) == "4"


def test_strip_case_insensitive():
    assert _strip_think_tags("<THINK>CAPS</THINK>ok") == "ok"


def test_no_tags_passthrough():
    assert _strip_think_tags("Just a normal response.") == "Just a normal response."


def test_empty_string():
    assert _strip_think_tags("") == ""


def test_only_think_tags():
    assert _strip_think_tags("<think>all thinking no answer</think>") == ""


def test_multiple_think_blocks():
    text = "<think>first</think>Hello <think>second</think>world"
    assert _strip_think_tags(text) == "Hello world"
