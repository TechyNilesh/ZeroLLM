"""Data loader — read and parse files for training or RAG."""

from zerollm.dataloader import load, chunk, extract_text

# ── Load training data (prompt/response pairs) ──

# From a CSV file
pairs = load("training_data.csv")
print(f"Loaded {len(pairs)} pairs from CSV")
for p in pairs[:2]:
    print(f"  Q: {p['prompt'][:50]}...")
    print(f"  A: {p['response'][:50]}...")

# From a JSONL file
pairs = load("training_data.jsonl")
print(f"Loaded {len(pairs)} pairs from JSONL")

# From a Python list
pairs = load([
    {"prompt": "What is AI?", "response": "Artificial Intelligence"},
    {"prompt": "What is ML?", "response": "Machine Learning"},
])
print(f"Loaded {len(pairs)} pairs from list")

# From a directory (reads all CSV/JSONL files)
pairs = load("training_data/")
print(f"Loaded {len(pairs)} pairs from directory")


# ── Chunk documents for RAG ──

# Chunk a PDF
chunks = chunk("report.pdf", chunk_size=400, overlap=80)
print(f"\nChunked PDF into {len(chunks)} chunks")

# Chunk a text file
chunks = chunk("notes.txt", chunk_size=200, overlap=40)
print(f"Chunked TXT into {len(chunks)} chunks")

# Chunk a DOCX
chunks = chunk("proposal.docx", chunk_size=400, overlap=80)
print(f"Chunked DOCX into {len(chunks)} chunks")

# Chunk all files in a directory
chunks = chunk("documents/", chunk_size=400, overlap=80)
print(f"Chunked directory into {len(chunks)} chunks")


# ── Extract raw text ──

text = extract_text("report.pdf")
print(f"\nExtracted {len(text)} characters from PDF")
