"""Agent class — tool-calling LLM with multi-turn support."""

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

        # If no default value, it's required
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


class Agent:
    """Build tool-calling agents with local LLMs.

    Supports tools, sub-agents, and multi-turn conversations.

    Usage:
        agent = Agent("Qwen/Qwen3-0.6B")

        @agent.tool
        def get_weather(city: str) -> str:
            return f"22C and sunny in {city}"

        agent.ask("What is the weather in Auckland?")

    Sub-agents:
        researcher = Agent("Qwen/Qwen3-1.7B")

        @researcher.tool
        def search(query: str) -> str:
            return f"Results for {query}"

        main = Agent("Qwen/Qwen3-0.6B")
        main.add_agent("researcher", researcher, "Research any topic on the web")

        main.ask("Research the latest AI news")  # delegates to researcher
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-0.6B",
        power: float = 1.0,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        max_tool_rounds: int = 5,
        name: str | None = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds

        # Tool registry
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []

        # Sub-agent registry
        self._sub_agents: dict[str, Agent] = {}

        # Resolve model — handles registry, local GGUF, and fine-tuned models
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
        self.memory.add_system(system_prompt or default_system)

        console.print(f"[green]✓[/green] Agent ready ({self.name})")

    def tool(self, func: Callable) -> Callable:
        """Register a function as a tool the agent can call.

        Usage:
            @agent.tool
            def search(query: str) -> str:
                '''Search the web for information.'''
                return f"Results for: {query}"
        """
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

        The sub-agent is exposed as a tool. When the main agent calls it,
        the sub-agent receives the prompt and runs its own tool loop.

        Args:
            name: Name to reference this sub-agent (used as tool name).
            agent: An Agent instance with its own tools and model.
            description: What this sub-agent does (shown to the LLM).
        """
        self._sub_agents[name] = agent

        # Create a wrapper function that delegates to the sub-agent
        def _delegate(task: str) -> str:
            console.print(f"[dim]  → Delegating to sub-agent '{name}': {task}[/dim]")
            return agent.ask(task)

        _delegate.__name__ = name
        _delegate.__doc__ = description or f"Delegate a task to the {name} sub-agent."
        _delegate.__annotations__ = {"task": str, "return": str}

        # Register the wrapper as a tool
        schema = _build_tool_schema(_delegate)
        self._tools[name] = _delegate
        self._tool_schemas.append(schema)

    def ask(self, prompt: str) -> str:
        """Send a prompt and let the agent use tools to answer.

        The agent will:
        1. Send prompt + tool schemas to the LLM
        2. If LLM returns a tool call → execute the function
        3. Feed the result back to the LLM
        4. Repeat until the LLM gives a final text answer
        """
        self.memory.add("user", prompt)

        for _ in range(self.max_tool_rounds):
            messages = self.memory.get_context()

            if not self._tool_schemas:
                # No tools registered, just generate normally
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

            # Add tool call and result to memory
            self.memory.add(
                "assistant",
                f"I'll call the {tool_name} tool with {json.dumps(tool_args)}",
            )
            self.memory.add("tool", str(tool_result))

        # Max rounds exceeded
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
                console.print("[dim]Conversation reset.[/dim]\n")
                continue

            response = self.ask(user_input)
            console.print(f"[bold green]Agent:[/bold green] {response}\n")

    def serve(self, port: int = 8081, host: str = "0.0.0.0") -> None:
        """Serve the agent as a REST API."""
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI(title=f"ZeroLLM Agent — {self.model_name}")

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

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": self.model_name}

        console.print(
            f"\n[bold]ZeroLLM Agent API[/bold]\n"
            f"  Model: {self.model_name}\n"
            f"  Tools: {', '.join(self._tools.keys())}\n"
            f"  URL:   http://{host}:{port}\n"
        )

        uvicorn.run(app, host=host, port=port, log_level="info")
