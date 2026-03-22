"""Data loader — shared file reader for FineTuner and RAG.

Supports: CSV, JSONL, TXT, PDF, DOCX, and directories of these files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load(source: str | Path | list[dict]) -> list[dict[str, str]]:
    """Load training data as prompt/response pairs.

    Args:
        source: Path to CSV/JSONL file, directory, or list of dicts.

    Returns:
        List of {"prompt": "...", "response": "..."} dicts.
    """
    # Already a list of dicts
    if isinstance(source, list):
        return _validate_pairs(source)

    path = Path(source)

    if path.is_dir():
        return _load_directory_pairs(path)

    ext = path.suffix.lower()
    loaders = {
        ".csv": _load_csv,
        ".jsonl": _load_jsonl,
        ".json": _load_jsonl,  # also accept .json as line-delimited
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported format '{ext}' for training data. "
            f"Supported: .csv, .jsonl, .json"
        )

    return loader(path)


def chunk(
    source: str | Path,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[str]:
    """Load and chunk text for RAG ingestion.

    Args:
        source: Path to PDF/TXT/DOCX file or directory.
        chunk_size: Target tokens per chunk (approximate, uses words as proxy).
        overlap: Number of words to overlap between chunks.

    Returns:
        List of text chunks.
    """
    path = Path(source)

    if path.is_dir():
        return _chunk_directory(path, chunk_size, overlap)

    text = extract_text(path)
    return _split_text(text, chunk_size, overlap)


def extract_text(path: str | Path) -> str:
    """Extract raw text from a file.

    Args:
        path: Path to PDF, TXT, or DOCX file.

    Returns:
        Extracted text as a string.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    extractors = {
        ".txt": _read_txt,
        ".md": _read_txt,
        ".pdf": _read_pdf,
        ".docx": _read_docx,
    }

    extractor = extractors.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported format '{ext}' for text extraction. "
            f"Supported: .txt, .md, .pdf, .docx"
        )

    return extractor(path)


# ──────────────────────────────────────────────────────────────
# Training data loaders (prompt/response pairs)
# ──────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load CSV with prompt/response columns."""
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Find the right column names (flexible matching)
        fieldnames = reader.fieldnames or []
        prompt_col = _find_column(fieldnames, ["prompt", "question", "input", "text"])
        response_col = _find_column(fieldnames, ["response", "answer", "output", "completion"])

        if prompt_col is None:
            raise ValueError(
                f"CSV must have a prompt column. "
                f"Found columns: {fieldnames}. "
                f"Expected one of: prompt, question, input, text"
            )

        for row in reader:
            prompt = row.get(prompt_col, "").strip()
            response = row.get(response_col, "").strip() if response_col else ""
            if prompt:
                pairs.append({"prompt": prompt, "response": response})

    return pairs


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load JSONL with prompt/response keys."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            # Flexible key matching
            prompt = (
                obj.get("prompt")
                or obj.get("question")
                or obj.get("input")
                or obj.get("text")
                or ""
            )
            response = (
                obj.get("response")
                or obj.get("answer")
                or obj.get("output")
                or obj.get("completion")
                or ""
            )
            if prompt:
                pairs.append({"prompt": str(prompt), "response": str(response)})

    return pairs


def _load_directory_pairs(directory: Path) -> list[dict[str, str]]:
    """Load all CSV/JSONL files from a directory."""
    pairs = []
    for ext in (".csv", ".jsonl", ".json"):
        for file in sorted(directory.glob(f"*{ext}")):
            pairs.extend(load(file))
    return pairs


def _validate_pairs(data: list[dict]) -> list[dict[str, str]]:
    """Validate and normalize a list of training pairs."""
    pairs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dict: {type(item)}")
        prompt = item.get("prompt") or item.get("question") or item.get("input") or ""
        response = item.get("response") or item.get("answer") or item.get("output") or ""
        if prompt:
            pairs.append({"prompt": str(prompt), "response": str(response)})
    return pairs


def _find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Find a column name from a list of candidates (case-insensitive)."""
    lower_fields = {f.lower().strip(): f for f in fieldnames}
    for candidate in candidates:
        if candidate in lower_fields:
            return lower_fields[candidate]
    return None


# ──────────────────────────────────────────────────────────────
# Text extractors
# ──────────────────────────────────────────────────────────────

def _read_txt(path: Path) -> str:
    """Read plain text file."""
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import pymupdf

    text_parts = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _read_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document

    doc = Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


# ──────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by word count.

    Uses words as a proxy for tokens (~0.75 tokens per word).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())

        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap
        if start <= 0:
            start = chunk_size  # safety: avoid infinite loop

    return chunks


def _chunk_directory(
    directory: Path, chunk_size: int = 400, overlap: int = 80
) -> list[str]:
    """Chunk all supported files in a directory."""
    all_chunks = []
    for ext in (".txt", ".md", ".pdf", ".docx"):
        for file in sorted(directory.glob(f"*{ext}")):
            all_chunks.extend(chunk(file, chunk_size, overlap))
    return all_chunks
