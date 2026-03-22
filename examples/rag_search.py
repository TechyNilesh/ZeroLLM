"""RAG search — search your documents without generating an answer."""

from zerollm import RAG

rag = RAG("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Add documents
rag.add("company_docs.pdf")
rag.add("faq.txt")

# Search only — returns matching chunks with scores (no LLM generation)
results = rag.search("refund policy", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | Source: {r['doc_path']}")
    print(f"  {r['content'][:100]}...")
    print()

# Ask a question — uses LLM to generate answer from retrieved chunks
answer = rag.ask("What is the refund policy?")
print(f"Answer: {answer}")

# Remove a document
rag.remove("faq.txt")

# List all indexed documents
print("\nIndexed documents:")
for doc in rag.list_documents():
    print(f"  {doc['path']} — {doc['chunks']} chunks, added {doc['added']}")
