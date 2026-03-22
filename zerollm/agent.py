"""Agent class — tool-calling LLM with sub-agents, shared context, and pipelines."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints

from rich.console import Console

from zerollm.backend import LlamaBackend
from zerollm.hardware import detect
from zerollm.memory import Memory
from zerollm.resolver import resolve

console = Console()

# Python type → JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _build_tool_schema(func: Callable) -> dict:
    """Build an OpenAI-compatible tool schema from a Python function."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(param_type, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class SharedContext:
    """Shared memory space for multi-agent communication.

    All agents in a group can read and write to the same context.
    This solves the problem of sub-agents not knowing what other agents found.

    Usage:
        ctx = SharedContext()
        main = Agent("model", context=ctx)
        researcher = Agent("model", context=ctx)

        # researcher writes
        ctx.set("research_results", "Found 3 articles about AI")

        # writer reads
        results = ctx.get("research_results")
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._log: list[dict[str, str]] = []

    def set(self, key: str, value: Any) -> None:
        """Store a value in shared context."""
        self._data[key] = value
        self._log.append({"action": "set", "key": key, "value": str(value)[:200]})

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from shared context."""
        return self._data.get(key, default)

    def append(self, key: str, value: Any) -> None:
        """Append to a list in shared context. Creates list if key doesn't exist."""
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(value)
        self._log.append({"action": "append", "key": key, "value": str(value)[:200]})

    def keys(self) -> list[str]:
        """List all keys in shared context."""
        return list(self._data.keys())

    def summary(self) -> str:
        """Get a text summary of everything in shared context (for injecting into prompts)."""
        if not self._data:
            return ""
        parts = []
        for key, value in self._data.items():
            val_str = str(value)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            parts.append(f"[{key}]: {val_str}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all shared context."""
        self._data.clear()
        self._log.clear()


class Agent:
    """Build tool-calling agents with local LLMs.

    Supports tools, sub-agents with shared context, and pipelines.

    Usage:
        agent = Agent("Qwen/Qwen3.5-4B")

        @agent.tool
        def get_weather(city: str) -> str:
            return f"22C and sunny in {city}"

        agent.ask("What is the weather in Auckland?")

    Sub-agents with shared context:
        ctx = SharedContext()
        researcher = Agent("model", name="researcher", context=ctx)
        writer = Agent("model", name="writer", context=ctx)

        main = Agent("model", context=ctx)
        main.add_agent("researcher", researcher, "Research topics")
        main.add_agent("writer", writer, "Write content")

        # All agents see what others wrote to ctx

    Pipeline:
        pipe = Pipeline([
            ("research", researcher),
            ("write", writer),
        ])
        result = pipe.run("Write a blog post about local LLMs")
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-4B",
        power: float = 1.0,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        max_tool_rounds: int = 5,
        name: str | None = None,
        context: SharedContext | None = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds

        # Tool registry
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []

        # Sub-agent registry
        self._sub_agents: dict[str, Agent] = {}

        # Shared context (for multi-agent communication)
        self.context = context or SharedContext()

        # Resolve model
        console.print(f"[dim]Loading {model} for agent...[/dim]")
        resolved = resolve(model)
        self.model_name = resolved.name
        self.name = name or resolved.name
        hw = detect()

        if not resolved.supports_tools:
            console.print(
                f"[yellow]Warning:[/yellow] {model} may not support tool calling well. "
                f"Consider using a model with supports_tools=True."
            )

        self.backend = LlamaBackend(
            model_path=resolved.path,
            context_length=resolved.context_length,
            power=power,
            hw=hw,
        )

        self.memory = Memory()

        default_system = (
            "You are a helpful assistant with access to tools. "
            "When a user asks something that can be answered using a tool, "
            "call the appropriate tool. Otherwise, respond directly."
        )
        self._system_prompt = system_prompt or default_system
        self.memory.add_system(self._system_prompt)

        console.print(f"[green]✓[/green] Agent ready ({self.name})")

    def tool(self, func: Callable) -> Callable:
        """Register a function as a tool the agent can call."""
        schema = _build_tool_schema(func)
        self._tools[func.__name__] = func
        self._tool_schemas.append(schema)
        return func

    def add_agent(
        self,
        name: str,
        agent: Agent,
        description: str = "",
    ) -> None:
        """Register a sub-agent that this agent can delegate to.

        The sub-agent receives:
        - The task string from the parent agent
        - Shared context summary (so it knows what others have found)
        - The parent's conversation context for background

        Args:
            name: Name to reference this sub-agent (used as tool name).
            agent: An Agent instance with its own tools and model.
            description: What this sub-agent does (shown to the LLM).
        """
        self._sub_agents[name] = agent

        # Share context if not already shared
        if agent.context is not self.context:
            agent.context = self.context

        parent_memory = self.memory

        def _delegate(task: str) -> str:
            console.print(f"[dim]  → Delegating to sub-agent '{name}': {task}[/dim]")

            # Build context-rich prompt for the sub-agent
            parts = [task]

            # Include shared context summary
            ctx_summary = self.context.summary()
            if ctx_summary:
                parts.append(f"\nShared context from other agents:\n{ctx_summary}")

            # Include parent conversation for background
            parent_history = parent_memory.get_full_history()
            if len(parent_history) > 1:
                recent = parent_history[-4:]  # last 2 turns
                history_text = "\n".join(
                    f"{m['role']}: {m['content'][:200]}" for m in recent
                    if m["role"] != "system"
                )
                if history_text:
                    parts.append(f"\nParent conversation context:\n{history_text}")

            full_prompt = "\n".join(parts)
            result = agent.ask(full_prompt)

            # Store result in shared context
            self.context.set(f"{name}_result", result)

            return result

        _delegate.__name__ = name
        _delegate.__doc__ = description or f"Delegate a task to the {name} sub-agent."
        _delegate.__annotations__ = {"task": str, "return": str}

        schema = _build_tool_schema(_delegate)
        self._tools[name] = _delegate
        self._tool_schemas.append(schema)

    def ask(self, prompt: str) -> str:
        """Send a prompt and let the agent use tools to answer.

        The agent will:
        1. Inject shared context into the system prompt
        2. Send prompt + tool schemas to the LLM
        3. If LLM returns a tool call → execute the function
        4. Feed the result back to the LLM
        5. Repeat until the LLM gives a final text answer
        """
        # Inject shared context into system prompt if available
        ctx_summary = self.context.summary()
        if ctx_summary:
            enriched_system = (
                f"{self._system_prompt}\n\n"
                f"Shared context from other agents:\n{ctx_summary}"
            )
            self.memory.add_system(enriched_system)

        self.memory.add("user", prompt)

        for _ in range(self.max_tool_rounds):
            messages = self.memory.get_context()

            if not self._tool_schemas:
                response = self.backend.generate(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.memory.add("assistant", response)
                return response

            result = self.backend.generate_with_tools(
                messages=messages,
                tools=self._tool_schemas,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if result["type"] == "text":
                self.memory.add("assistant", result["content"])
                return result["content"]

            # Execute tool call
            tool_name = result["name"]
            tool_args = result["arguments"]

            if tool_name not in self._tools:
                error_msg = f"Unknown tool: {tool_name}"
                self.memory.add("assistant", error_msg)
                return error_msg

            console.print(
                f"[dim]  → Calling {tool_name}({json.dumps(tool_args)})[/dim]"
            )

            try:
                tool_result = self._tools[tool_name](**tool_args)
            except Exception as e:
                tool_result = f"Error calling {tool_name}: {e}"

            self.memory.add(
                "assistant",
                f"I'll call the {tool_name} tool with {json.dumps(tool_args)}",
            )
            self.memory.add("tool", str(tool_result))

        fallback = "I've reached the maximum number of tool calls. Here's what I found so far."
        self.memory.add("assistant", fallback)
        return fallback

    def chat(self) -> None:
        """Start an interactive agent chat in the terminal."""
        tools_list = [t for t in self._tools.keys() if t not in self._sub_agents]
        agents_list = list(self._sub_agents.keys())

        info_parts = [f"\n[bold]ZeroLLM Agent[/bold] — {self.name}"]
        if tools_list:
            info_parts.append(f"Tools: {', '.join(tools_list)}")
        if agents_list:
            info_parts.append(f"Sub-agents: {', '.join(agents_list)}")
        info_parts.append("Type your message. Type [bold]/quit[/bold] to exit.\n")

        console.print("\n".join(info_parts))

        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/reset":
                self.memory.clear()
                self.context.clear()
                console.print("[dim]Conversation and context reset.[/dim]\n")
                continue

            if user_input.lower() == "/context":
                summary = self.context.summary()
                if summary:
                    console.print(f"[dim]{summary}[/dim]\n")
                else:
                    console.print("[dim]Shared context is empty.[/dim]\n")
                continue

            response = self.ask(user_input)
            console.print(f"[bold green]Agent:[/bold green] {response}\n")

    def serve(self, port: int = 8081, host: str = "0.0.0.0") -> None:
        """Serve the agent as a REST API."""
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI(title=f"ZeroLLM Agent — {self.name}")

        class AgentRequest(BaseModel):
            prompt: str
            max_tokens: int = self.max_tokens

        @app.post("/ask")
        async def agent_ask(req: AgentRequest):
            response = self.ask(req.prompt)
            return {"response": response, "model": self.model_name}

        @app.get("/tools")
        async def list_tools():
            return {"tools": self._tool_schemas}

        @app.get("/context")
        async def get_context():
            return {"context": self.context._data}

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": self.model_name, "agent": self.name}

        console.print(
            f"\n[bold]ZeroLLM Agent API[/bold]\n"
            f"  Agent: {self.name}\n"
            f"  Model: {self.model_name}\n"
            f"  Tools: {', '.join(self._tools.keys())}\n"
            f"  URL:   http://{host}:{port}\n"
        )

        uvicorn.run(app, host=host, port=port, log_level="info")


class Pipeline:
    """Run multiple agents in sequence, passing output from one to the next.

    Each agent's result is stored in SharedContext and passed as input to the next.

    Usage:
        researcher = Agent("model", name="researcher")
        writer = Agent("model", name="writer")

        pipe = Pipeline([
            ("research", researcher),
            ("write", writer),
        ])
        result = pipe.run("Write a blog post about local LLMs")
    """

    def __init__(self, steps: list[tuple[str, Agent]]):
        """Initialize pipeline.

        Args:
            steps: List of (step_name, agent) tuples. Agents run in order.
        """
        self.steps = steps

        # All agents share the same context
        self.context = SharedContext()
        for _, agent in self.steps:
            agent.context = self.context

    def run(self, initial_prompt: str) -> str:
        """Run the pipeline.

        Args:
            initial_prompt: Starting prompt for the first agent.

        Returns:
            Final agent's response.
        """
        self.context.set("initial_prompt", initial_prompt)
        current_input = initial_prompt
        last_result = ""

        for step_name, agent in self.steps:
            console.print(f"\n[bold]Pipeline step: {step_name}[/bold] ({agent.name})")

            # Build prompt with context from previous steps
            parts = [current_input]
            ctx_summary = self.context.summary()
            if ctx_summary:
                parts.append(f"\nContext from previous steps:\n{ctx_summary}")

            prompt = "\n".join(parts)
            last_result = agent.ask(prompt)

            # Store result in context for next step
            self.context.set(f"{step_name}_result", last_result)
            current_input = last_result

            console.print(f"[green]✓[/green] {step_name} complete")

        return last_result
