"""Conversation-aware RAG — follow-up questions just work."""

from zerollm import RAG

rag = RAG("Qwen/Qwen3.5-4B")
rag.add("company_docs.pdf")
rag.add("faq.txt")

# First question — searches normally
print(rag.chat("What is your refund policy?"))

# Follow-up — "how long" is vague on its own
# ZeroLLM rewrites it to: "How long is the refund window for the refund policy?"
print(rag.chat("How long do I have?"))

# Another follow-up
print(rag.chat("What if the item is damaged?"))

# Each question is automatically rewritten using the conversation history
# so the retrieval always finds the right documents
