"""RAG — Retrieval-Augmented Generation with SQLite + sqlite-vec.

Inspired by OpenClaw's memory architecture:
- Vector search via sqlite-vec (cosine similarity)
- Keyword search via SQLite FTS5 (BM25)
- Hybrid scoring: 70% vector + 30% BM25
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

from rich.console import Console

from zerollm.backend import LlamaBackend
from zerollm.dataloader import chunk, extract_text
from zerollm.hardware import detect
from zerollm.resolver import resolve

console = Console()

RAG_DB = Path.home() / ".cache" / "zerollm" / "rag.db"

# Weights for hybrid scoring
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def _serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class RAG:
    """Retrieval-Augmented Generation — ask questions about your documents.

    Usage:
        rag = RAG("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        rag.add("company_docs.pdf")
        rag.add("faq.txt")
        answer = rag.ask("What is the return policy?")
    """

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        power: float = 1.0,
        db_path: str | Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 5,
    ):
        """Initialize RAG.

        Args:
            model: LLM model name for answering questions.
            power: Resource usage 0.0-1.0.
            db_path: Path to SQLite database (default: ~/.cache/zerollm/rag.db).
            embedding_model: Sentence transformer model for embeddings.
            chunk_size: Words per chunk when ingesting documents.
            chunk_overlap: Overlap words between chunks.
            top_k: Number of chunks to retrieve per query.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Resolve model — handles registry, local GGUF, and fine-tuned models
        resolved = resolve(model)
        self.model_name = resolved.name
        hw = detect()

        self.backend = LlamaBackend(
            model_path=resolved.path,
            context_length=resolved.context_length,
            power=power,
            hw=hw,
        )

        # Load embedding model
        console.print(f"[dim]Loading embedding model ({embedding_model})...[/dim]")
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(embedding_model)
        self._embedding_dim = self._embedder.get_sentence_embedding_dimension()

        # Initialize database
        self._db_path = Path(db_path) if db_path else RAG_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = self._init_db()

        console.print(f"[green]✓[/green] RAG ready ({model}, {self._doc_count()} documents)")

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database with FTS5 and sqlite-vec."""
        db = sqlite3.connect(str(self._db_path))

        # Enable sqlite-vec extension
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)

        # Chunks table — stores text and metadata
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_hash TEXT NOT NULL,
                doc_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # FTS5 virtual table for keyword search (BM25)
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(content, content_rowid='id')
        """)

        # Vector table for semantic search
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
            USING vec0(embedding float[{self._embedding_dim}])
        """)

        db.commit()
        return db

    def add(self, source: str | Path) -> int:
        """Add a document to the RAG database.

        Args:
            source: Path to PDF, TXT, DOCX, or directory.

        Returns:
            Number of chunks added.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Generate document hash to avoid duplicates
        doc_hash = self._hash_file(path)
        if self._doc_exists(doc_hash):
            console.print(f"[yellow]![/yellow] {path.name} already added (skipping)")
            return 0

        console.print(f"[dim]Processing {path.name}...[/dim]")

        # Chunk the document
        chunks = chunk(source, self.chunk_size, self.chunk_overlap)
        if not chunks:
            console.print(f"[yellow]![/yellow] No text found in {path.name}")
            return 0

        # Generate embeddings
        embeddings = self._embedder.encode(chunks, show_progress_bar=False)

        # Insert into database
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            # Insert chunk text
            cursor = self._db.execute(
                "INSERT INTO chunks (doc_hash, doc_path, chunk_index, content) VALUES (?, ?, ?, ?)",
                (doc_hash, str(path), i, text),
            )
            row_id = cursor.lastrowid

            # Insert into FTS5
            self._db.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                (row_id, text),
            )

            # Insert embedding vector
            self._db.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (row_id, _serialize_vector(emb.tolist())),
            )

        self._db.commit()
        console.print(f"[green]✓[/green] Added {path.name} ({len(chunks)} chunks)")
        return len(chunks)

    def ask(self, question: str) -> str:
        """Ask a question about your documents.

        Args:
            question: Your question.

        Returns:
            Answer generated by the LLM using retrieved context.
        """
        # Retrieve relevant chunks
        retrieved = self.search(question)

        if not retrieved:
            return "I don't have enough information to answer that question. Try adding more documents."

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join(
            f"[Source: {r['doc_path']}, Chunk {r['chunk_index']}]\n{r['content']}"
            for r in retrieved
        )

        # Generate answer
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based ONLY on "
                    "the provided context. If the context doesn't contain enough information, "
                    "say so. Be concise and accurate."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        response = self.backend.generate(messages=messages, max_tokens=1024, temperature=0.3)
        return response

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for relevant chunks using hybrid search.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with content, score, doc_path, chunk_index.
        """
        k = top_k or self.top_k

        # Vector search
        query_embedding = self._embedder.encode([query])[0]
        vector_results = self._vector_search(query_embedding, k * 4)

        # Keyword search (BM25)
        bm25_results = self._bm25_search(query, k * 4)

        # Hybrid scoring
        scored = self._hybrid_score(vector_results, bm25_results)

        # Return top-k
        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for row_id, score in top:
            row = self._db.execute(
                "SELECT content, doc_path, chunk_index FROM chunks WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row:
                results.append({
                    "content": row[0],
                    "doc_path": row[1],
                    "chunk_index": row[2],
                    "score": score,
                })

        return results

    def _vector_search(self, query_vec, limit: int) -> dict[int, float]:
        """Cosine similarity search via sqlite-vec."""
        results = {}
        rows = self._db.execute(
            """
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_serialize_vector(query_vec.tolist()), limit),
        ).fetchall()

        for row_id, distance in rows:
            # sqlite-vec returns distance; convert to similarity (1 - distance for cosine)
            results[row_id] = max(0.0, 1.0 - distance)

        return results

    def _bm25_search(self, query: str, limit: int) -> dict[int, float]:
        """BM25 keyword search via FTS5."""
        results = {}
        try:
            rows = self._db.execute(
                """
                SELECT rowid, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

            if not rows:
                return results

            # Normalize BM25 scores to 0-1 range
            ranks = [abs(r[1]) for r in rows]
            max_rank = max(ranks) if ranks else 1.0

            for row_id, rank in rows:
                results[row_id] = abs(rank) / max_rank if max_rank > 0 else 0.0

        except sqlite3.OperationalError:
            # FTS5 query syntax error — fall back to empty results
            pass

        return results

    def _hybrid_score(
        self,
        vector_results: dict[int, float],
        bm25_results: dict[int, float],
    ) -> dict[int, float]:
        """Combine vector and BM25 scores with weighted blend."""
        all_ids = set(vector_results.keys()) | set(bm25_results.keys())
        scored = {}

        for row_id in all_ids:
            vec_score = vector_results.get(row_id, 0.0)
            bm25_score = bm25_results.get(row_id, 0.0)
            scored[row_id] = VECTOR_WEIGHT * vec_score + BM25_WEIGHT * bm25_score

        return scored

    def remove(self, source: str | Path) -> bool:
        """Remove a document from the RAG database."""
        path = Path(source)
        doc_hash = self._hash_file(path)

        # Get chunk IDs
        rows = self._db.execute(
            "SELECT id FROM chunks WHERE doc_hash = ?", (doc_hash,)
        ).fetchall()

        if not rows:
            console.print(f"[yellow]![/yellow] {path.name} not found in database")
            return False

        ids = [r[0] for r in rows]

        for row_id in ids:
            self._db.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row_id,))
            self._db.execute("DELETE FROM chunks_vec WHERE rowid = ?", (row_id,))

        self._db.execute("DELETE FROM chunks WHERE doc_hash = ?", (doc_hash,))
        self._db.commit()

        console.print(f"[green]✓[/green] Removed {path.name} ({len(ids)} chunks)")
        return True

    def list_documents(self) -> list[dict]:
        """List all documents in the RAG database."""
        rows = self._db.execute(
            """
            SELECT doc_path, doc_hash, COUNT(*) as chunks, MIN(created_at) as added
            FROM chunks
            GROUP BY doc_hash
            ORDER BY added
            """
        ).fetchall()

        return [
            {"path": r[0], "hash": r[1], "chunks": r[2], "added": r[3]}
            for r in rows
        ]

    def _hash_file(self, path: Path) -> str:
        """Generate a hash for a file to detect duplicates."""
        if path.is_dir():
            return hashlib.md5(str(path).encode()).hexdigest()
        content = path.read_bytes()
        return hashlib.md5(content).hexdigest()

    def _doc_exists(self, doc_hash: str) -> bool:
        """Check if a document is already in the database."""
        row = self._db.execute(
            "SELECT 1 FROM chunks WHERE doc_hash = ? LIMIT 1", (doc_hash,)
        ).fetchone()
        return row is not None

    def _doc_count(self) -> int:
        """Count unique documents in the database."""
        row = self._db.execute(
            "SELECT COUNT(DISTINCT doc_hash) FROM chunks"
        ).fetchone()
        return row[0] if row else 0

    def __del__(self):
        if hasattr(self, "_db") and self._db:
            self._db.close()
