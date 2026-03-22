"""RAG example — ask questions about your documents."""

from zerollm import RAG

rag = RAG("Qwen/Qwen3-0.6B")

# Add documents
rag.add("company_docs.pdf")
rag.add("faq.txt")

# Ask questions
print(rag.ask("What is the return policy?"))
print(rag.ask("How do I contact support?"))

# List indexed documents
for doc in rag.list_documents():
    print(f"  {doc['path']} — {doc['chunks']} chunks")
