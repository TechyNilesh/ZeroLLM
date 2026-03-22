"""Tests for the Agent class (mocked backend)."""

from unittest.mock import patch, MagicMock

from zerollm.agent import Agent, _build_tool_schema


# ── Tool schema tests (no mocking needed) ──

def test_build_tool_schema_simple():
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello {name}"

    schema = _build_tool_schema(greet)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "greet"
    assert schema["function"]["description"] == "Say hello to someone."
    assert "name" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["name"]["type"] == "string"
    assert "name" in schema["function"]["parameters"]["required"]


def test_build_tool_schema_multiple_params():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    schema = _build_tool_schema(add)
    props = schema["function"]["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert len(schema["function"]["parameters"]["required"]) == 2


def test_build_tool_schema_optional_param():
    def search(query: str, limit: int = 10) -> str:
        """Search for something."""
        return query

    schema = _build_tool_schema(search)
    required = schema["function"]["parameters"]["required"]
    assert "query" in required
    assert "limit" not in required


def test_build_tool_schema_bool_param():
    def toggle(enabled: bool) -> str:
        """Toggle a feature."""
        return str(enabled)

    schema = _build_tool_schema(toggle)
    assert schema["function"]["parameters"]["properties"]["enabled"]["type"] == "boolean"


# ── Agent tests (mocked backend) ──

def _mock_agent(**kwargs):
    """Create an Agent with mocked backend."""
    from zerollm.resolver import ResolvedModel

    mock_resolved = ResolvedModel(
        name=kwargs.get("model", "Qwen/Qwen3-0.6B"),
        path="/fake/model.gguf",
        context_length=8192,
        source="registry",
        supports_tools=True,
    )

    with patch("zerollm.agent.resolve", return_value=mock_resolved), \
         patch("zerollm.agent.LlamaBackend") as mock_backend, \
         patch("zerollm.agent.console"):

        mock_instance = MagicMock()
        mock_backend.return_value = mock_instance

        agent = Agent(**kwargs)
        agent._mock_backend = mock_instance
        return agent


def test_agent_init():
    agent = _mock_agent()
    assert agent.model_name == "Qwen/Qwen3-0.6B"
    assert len(agent._tools) == 0


def test_agent_register_tool():
    agent = _mock_agent()

    @agent.tool
    def hello(name: str) -> str:
        """Say hello."""
        return f"Hello {name}"

    assert "hello" in agent._tools
    assert len(agent._tool_schemas) == 1


def test_agent_ask_no_tools():
    agent = _mock_agent()
    agent._mock_backend.generate.return_value = "Just a text response"

    result = agent.ask("Hello")
    assert result == "Just a text response"


def test_agent_ask_with_tool_call():
    agent = _mock_agent()

    @agent.tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    # First call returns tool_call, second returns text
    agent._mock_backend.generate_with_tools.side_effect = [
        {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Auckland"}},
        {"type": "text", "content": "It's sunny in Auckland!"},
    ]

    result = agent.ask("Weather in Auckland?")
    assert "sunny" in result.lower() or "Auckland" in result


def test_agent_ask_unknown_tool():
    agent = _mock_agent()

    agent._mock_backend.generate_with_tools.return_value = {
        "type": "tool_call",
        "name": "nonexistent_tool",
        "arguments": {},
    }

    result = agent.ask("Do something")
    assert "Unknown tool" in result


def test_agent_max_tool_rounds():
    agent = _mock_agent(max_tool_rounds=2)

    @agent.tool
    def loop_tool() -> str:
        """A tool that keeps getting called."""
        return "result"

    # Always return tool calls (never text)
    agent._mock_backend.generate_with_tools.return_value = {
        "type": "tool_call",
        "name": "loop_tool",
        "arguments": {},
    }

    result = agent.ask("Keep calling tools")
    assert "maximum" in result.lower()


def test_agent_multiple_tools():
    agent = _mock_agent()

    @agent.tool
    def tool_a(x: str) -> str:
        """Tool A."""
        return f"A: {x}"

    @agent.tool
    def tool_b(y: int) -> str:
        """Tool B."""
        return f"B: {y}"

    assert len(agent._tools) == 2
    assert len(agent._tool_schemas) == 2
    assert "tool_a" in agent._tools
    assert "tool_b" in agent._tools


# ── Sub-agent tests ──

def test_agent_add_sub_agent():
    main = _mock_agent(name="main")
    sub = _mock_agent(name="researcher")

    @sub.tool
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    main.add_agent("researcher", sub, description="Research any topic")

    assert "researcher" in main._tools
    assert "researcher" in main._sub_agents
    assert len(main._tool_schemas) == 1  # sub-agent registered as tool


def test_agent_sub_agent_in_schemas():
    main = _mock_agent()
    sub = _mock_agent()

    main.add_agent("helper", sub, description="A helpful sub-agent")

    # Check the tool schema was created correctly
    schema = main._tool_schemas[0]
    assert schema["function"]["name"] == "helper"
    assert "sub-agent" in schema["function"]["description"]
    assert "task" in schema["function"]["parameters"]["properties"]


def test_agent_sub_agent_delegation():
    main = _mock_agent()
    sub = _mock_agent()

    # Sub-agent returns text directly
    sub._mock_backend.generate.return_value = "Sub-agent result: found 3 articles"

    main.add_agent("researcher", sub, description="Research topics")

    # When main agent calls the sub-agent tool
    result = main._tools["researcher"](task="latest AI news")
    assert "Sub-agent result" in result


def test_agent_multiple_sub_agents():
    main = _mock_agent()
    researcher = _mock_agent(name="researcher")
    writer = _mock_agent(name="writer")

    main.add_agent("researcher", researcher, "Research topics")
    main.add_agent("writer", writer, "Write content")

    assert len(main._sub_agents) == 2
    assert len(main._tool_schemas) == 2
    assert "researcher" in main._tools
    assert "writer" in main._tools


def test_agent_mixed_tools_and_sub_agents():
    main = _mock_agent()
    sub = _mock_agent()

    @main.tool
    def calculate(expr: str) -> str:
        """Calculate math."""
        return str(eval(expr))

    main.add_agent("helper", sub, "Help with tasks")

    assert len(main._tools) == 2  # calculate + helper
    assert len(main._sub_agents) == 1  # only helper
    assert "calculate" in main._tools
    assert "helper" in main._tools


def test_agent_name_parameter():
    agent = _mock_agent(name="my-custom-agent")
    assert agent.name == "my-custom-agent"
