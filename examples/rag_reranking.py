"""RAG with cross-encoder reranking — better retrieval quality."""

from zerollm import RAG

# Without reranking (default — fast)
rag_fast = RAG("Qwen/Qwen3.5-4B")

# With reranking (slower but much better results)
rag_quality = RAG(
    "Qwen/Qwen3.5-4B",
    rerank=True,  # enables cross-encoder reranking
    rerank_model="BAAI/bge-reranker-v2-m3",  # default reranker
)

# How reranking works:
# 1. Hybrid search retrieves top-20 candidates (vector + BM25)
# 2. Cross-encoder reranks all 20 candidates jointly
# 3. Returns top-5 with much better relevance

rag_quality.add("company_docs.pdf")

# Same API — just better results
print(rag_quality.ask("What is the refund policy?"))

# You can also use a different reranker
rag_custom = RAG(
    "Qwen/Qwen3.5-4B",
    rerank=True,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # lighter, faster
)
