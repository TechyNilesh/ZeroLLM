"""RAG with custom embedding models.

The default embedding model is 'all-MiniLM-L6-v2' (fast, small, good enough).
You can use any sentence-transformers model from HuggingFace for better quality.
"""

from zerollm import RAG

# ── Default embedding (fast, 384 dimensions) ──

rag_fast = RAG("HuggingFaceTB/SmolLM2-1.7B-Instruct")
# Uses: all-MiniLM-L6-v2 (default)
# Size: ~80MB, fast, good for most use cases


# ── Better quality embedding (slower, 384 dimensions) ──

rag_quality = RAG(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    embedding_model="BAAI/bge-small-en-v1.5",
    db_path="./rag_quality.db",  # separate DB for different embeddings
)
# Better retrieval quality, slightly slower


# ── Multilingual embedding (supports 100+ languages) ──

rag_multi = RAG(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    db_path="./rag_multilingual.db",
)
# Works with non-English documents


# ── Large embedding for best quality (slower, 1024 dimensions) ──

rag_best = RAG(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    embedding_model="BAAI/bge-large-en-v1.5",
    db_path="./rag_best.db",
)
# Best retrieval quality, but ~1.3GB model and slower


# ── Use any of them the same way ──

rag_fast.add("company_docs.pdf")
print(rag_fast.ask("What is the return policy?"))

# ── Custom chunk sizes for different document types ──

# Small chunks for Q&A / FAQ documents
rag_faq = RAG(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    chunk_size=200,       # smaller chunks = more precise retrieval
    chunk_overlap=40,
    top_k=3,              # fewer results needed
)

# Large chunks for long-form documents (reports, books)
rag_reports = RAG(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    chunk_size=800,       # bigger chunks = more context per result
    chunk_overlap=150,
    top_k=5,
)
