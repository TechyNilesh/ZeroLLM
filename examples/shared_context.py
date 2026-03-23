"""SharedContext — agents share memory across a group."""

from zerollm import Agent, SharedContext

# All agents share one context
ctx = SharedContext()

analyst = Agent("Qwen/Qwen3.5-4B", name="analyst", context=ctx)
advisor = Agent("Qwen/Qwen3.5-4B", name="advisor", context=ctx)

@analyst.tool
def get_metrics(company: str) -> str:
    """Get company metrics."""
    return f"{company}: revenue $10M, growth 25%, employees 50"

# Analyst runs first, stores findings in shared context
result = analyst.ask("Analyze the metrics for Acme Corp")
ctx.set("analysis", result)

# Advisor sees what analyst found via shared context
print(advisor.ask("Based on the analysis, what should Acme Corp focus on?"))

# Inspect shared context
print("\n--- Shared Context ---")
for key in ctx.keys():
    print(f"  {key}: {str(ctx.get(key))[:80]}...")
