"""Tests for the Chat class (mocked backend)."""

from unittest.mock import patch, MagicMock

from zerollm.chat import Chat


def _mock_chat(**kwargs):
    """Create a Chat instance with mocked backend and resolver."""
    from zerollm.resolver import ResolvedModel

    mock_resolved = ResolvedModel(
        name=kwargs.get("model", "Qwen/Qwen3.5-4B"),
        model_id="Qwen/Qwen3.5-4B",
        context_length=8192,
        source="huggingface",
        supports_tools=True,
    )

    with patch("zerollm.chat.resolve", return_value=mock_resolved), \
         patch("zerollm.chat.HFBackend") as mock_backend, \
         patch("zerollm.chat.console"):

        mock_instance = MagicMock()
        mock_backend.return_value = mock_instance

        bot = Chat(**kwargs)
        bot._mock_backend = mock_instance
        return bot


def test_chat_init_default():
    bot = _mock_chat()
    assert bot.model_name == "Qwen/Qwen3.5-4B"


def test_chat_init_custom_model():
    bot = _mock_chat(model="microsoft/Phi-3-mini-4k-instruct")
    assert bot.model_name == "microsoft/Phi-3-mini-4k-instruct"


def test_chat_ask():
    bot = _mock_chat()
    bot._mock_backend.generate.return_value = "Paris"

    result = bot.ask("What is the capital of France?")
    assert result == "Paris"
    assert bot.memory.turn_count == 1


def test_chat_ask_multiple():
    bot = _mock_chat()
    bot._mock_backend.generate.side_effect = ["Paris", "Berlin"]

    bot.ask("Capital of France?")
    bot.ask("Capital of Germany?")
    assert bot.memory.turn_count == 2


def test_chat_stream():
    bot = _mock_chat()
    bot._mock_backend.generate.return_value = iter(["Hello", " ", "world"])

    tokens = list(bot.stream("Hi"))
    assert tokens == ["Hello", " ", "world"]
    assert bot.memory.turn_count == 1


def test_chat_system_prompt():
    bot = _mock_chat(system_prompt="You are a pirate.")
    context = bot.memory.get_context()
    assert context[0]["role"] == "system"
    assert "pirate" in context[0]["content"]


def test_chat_reset():
    bot = _mock_chat()
    bot._mock_backend.generate.return_value = "Hi"
    bot.ask("Hello")
    assert bot.memory.turn_count == 1

    bot.reset()
    assert bot.memory.turn_count == 0


def test_chat_history():
    bot = _mock_chat()
    bot._mock_backend.generate.return_value = "Hi there"
    bot.ask("Hello")

    history = bot.history
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_with_memory():
    bot = _mock_chat(memory=True)
    assert bot.memory.persist is True


def test_chat_power_setting():
    bot = _mock_chat(power=0.5)
    assert bot.power == 0.5


def test_chat_temperature():
    bot = _mock_chat(temperature=0.1)
    assert bot.temperature == 0.1
