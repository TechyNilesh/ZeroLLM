"""Agentic RAG — agent decides when to search documents."""

from zerollm import Agent, RAG

# Set up RAG with your documents
rag = RAG("Qwen/Qwen3.5-4B", rerank=True)
rag.add("company_docs.pdf")
rag.add("faq.txt")

# Create an agent and connect RAG as a tool
agent = Agent("Qwen/Qwen3.5-4B")
agent.add_rag(rag, "Search company documents for policies, FAQs, and procedures")

# The agent decides when to search
# For factual questions about documents — it searches
print(agent.ask("What is our refund policy?"))

# For general questions — it answers directly without searching
print(agent.ask("What is 2+2?"))

# You can also add other tools alongside RAG
@agent.tool
def send_email(to: str, subject: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to} with subject: {subject}"

# Agent can combine RAG search with other tools
print(agent.ask("Find our refund policy and email it to john@example.com"))
