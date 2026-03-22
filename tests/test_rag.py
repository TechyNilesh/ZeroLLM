"""Tests for the RAG module (unit tests for search/scoring logic)."""

from zerollm.rag import _serialize_vector


def test_serialize_vector():
    vec = [1.0, 2.0, 3.0]
    result = _serialize_vector(vec)
    assert isinstance(result, bytes)
    # 3 floats * 4 bytes each = 12 bytes
    assert len(result) == 12


def test_serialize_vector_empty():
    vec = []
    result = _serialize_vector(vec)
    assert len(result) == 0


def test_hybrid_scoring():
    """Test hybrid scoring logic directly."""
    from zerollm.rag import VECTOR_WEIGHT, BM25_WEIGHT

    # Simulate scores
    vector_results = {1: 0.9, 2: 0.5, 3: 0.3}
    bm25_results = {1: 0.2, 2: 0.8, 4: 1.0}

    # Manual hybrid calculation
    all_ids = set(vector_results.keys()) | set(bm25_results.keys())
    scored = {}
    for row_id in all_ids:
        vec_score = vector_results.get(row_id, 0.0)
        bm25_score = bm25_results.get(row_id, 0.0)
        scored[row_id] = VECTOR_WEIGHT * vec_score + BM25_WEIGHT * bm25_score

    # Doc 1: 0.7*0.9 + 0.3*0.2 = 0.69
    assert abs(scored[1] - 0.69) < 0.01
    # Doc 2: 0.7*0.5 + 0.3*0.8 = 0.59
    assert abs(scored[2] - 0.59) < 0.01
    # Doc 3: 0.7*0.3 + 0.3*0.0 = 0.21
    assert abs(scored[3] - 0.21) < 0.01
    # Doc 4: 0.7*0.0 + 0.3*1.0 = 0.30
    assert abs(scored[4] - 0.30) < 0.01

    # Doc 1 should rank highest
    best = max(scored, key=scored.get)
    assert best == 1


def test_weights_sum_to_one():
    from zerollm.rag import VECTOR_WEIGHT, BM25_WEIGHT
    assert abs(VECTOR_WEIGHT + BM25_WEIGHT - 1.0) < 0.001
