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
        name=kwargs.get("model", "Qwen/Qwen3.5-4B"),
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
    assert agent.model_name == "Qwen/Qwen3.5-4B"
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

    # Register a tool so the agent uses generate_with_tools path
    @agent.tool
    def real_tool(x: str) -> str:
        """A real tool."""
        return x

    # But the LLM calls a tool that doesn't exist
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


# ── SharedContext tests ──

def test_shared_context_set_get():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    ctx.set("key1", "value1")
    assert ctx.get("key1") == "value1"
    assert ctx.get("missing") is None
    assert ctx.get("missing", "default") == "default"


def test_shared_context_append():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    ctx.append("results", "item1")
    ctx.append("results", "item2")
    assert ctx.get("results") == ["item1", "item2"]


def test_shared_context_keys():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    ctx.set("a", 1)
    ctx.set("b", 2)
    assert sorted(ctx.keys()) == ["a", "b"]


def test_shared_context_summary():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    ctx.set("research", "Found 3 articles")
    ctx.set("status", "done")
    summary = ctx.summary()
    assert "research" in summary
    assert "Found 3 articles" in summary
    assert "status" in summary


def test_shared_context_clear():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    ctx.set("key", "value")
    ctx.clear()
    assert ctx.get("key") is None
    assert ctx.keys() == []


def test_shared_context_empty_summary():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    assert ctx.summary() == ""


def test_agents_share_context():
    from zerollm.agent import SharedContext
    ctx = SharedContext()
    main = _mock_agent(name="main", context=ctx)
    sub = _mock_agent(name="sub", context=ctx)

    # Both agents see the same context
    assert main.context is sub.context
    main.context.set("shared_data", "hello")
    assert sub.context.get("shared_data") == "hello"


def test_add_agent_shares_context():
    main = _mock_agent(name="main")
    sub = _mock_agent(name="sub")

    # Initially different contexts
    assert main.context is not sub.context

    main.add_agent("sub", sub, "A sub-agent")

    # After add_agent, they share the same context
    assert sub.context is main.context


def test_sub_agent_stores_result_in_context():
    main = _mock_agent(name="main")
    sub = _mock_agent(name="researcher")
    sub._mock_backend.generate.return_value = "Found important data"

    main.add_agent("researcher", sub, "Research topics")

    # Call the sub-agent
    main._tools["researcher"](task="find data")

    # Result should be in shared context
    assert main.context.get("researcher_result") == "Found important data"


# ── Pipeline tests ──

def test_pipeline_creation():
    from zerollm.agent import Pipeline
    a1 = _mock_agent(name="step1")
    a2 = _mock_agent(name="step2")

    pipe = Pipeline([("research", a1), ("write", a2)])
    assert len(pipe.steps) == 2
    # All agents share the same context
    assert a1.context is a2.context
    assert a1.context is pipe.context


def test_pipeline_run():
    from zerollm.agent import Pipeline
    a1 = _mock_agent(name="researcher")
    a2 = _mock_agent(name="writer")

    a1._mock_backend.generate.return_value = "Research results about AI"
    a2._mock_backend.generate.return_value = "Blog post based on research"

    pipe = Pipeline([("research", a1), ("write", a2)])
    result = pipe.run("Write a blog about AI")

    assert result == "Blog post based on research"
    assert pipe.context.get("initial_prompt") == "Write a blog about AI"
    assert pipe.context.get("research_result") == "Research results about AI"
    assert pipe.context.get("write_result") == "Blog post based on research"
