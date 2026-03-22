"""Tests for the data loader."""

import json
import tempfile
from pathlib import Path

import pytest

from zerollm.dataloader import load, chunk, _split_text


def test_load_list_of_dicts():
    data = [
        {"prompt": "Hello", "response": "Hi there"},
        {"prompt": "Bye", "response": "Goodbye"},
    ]
    pairs = load(data)
    assert len(pairs) == 2
    assert pairs[0]["prompt"] == "Hello"
    assert pairs[0]["response"] == "Hi there"


def test_load_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("prompt,response\nWhat is AI?,Artificial Intelligence\nHello,Hi\n")

    pairs = load(str(csv_file))
    assert len(pairs) == 2
    assert pairs[0]["prompt"] == "What is AI?"


def test_load_jsonl(tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"prompt": "Q1", "response": "A1"}),
        json.dumps({"prompt": "Q2", "response": "A2"}),
    ]
    jsonl_file.write_text("\n".join(lines))

    pairs = load(str(jsonl_file))
    assert len(pairs) == 2


def test_load_flexible_column_names(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("question,answer\nWhat?,This\nWhy?,Because\n")

    pairs = load(str(csv_file))
    assert len(pairs) == 2
    assert pairs[0]["prompt"] == "What?"


def test_split_text():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = _split_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)


def test_chunk_txt_file(tmp_path):
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text(" ".join(["hello"] * 500))

    chunks = chunk(str(txt_file), chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_load_directory(tmp_path):
    # Create multiple files
    (tmp_path / "a.csv").write_text("prompt,response\nQ1,A1\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"prompt": "Q2", "response": "A2"}))

    pairs = load(str(tmp_path))
    assert len(pairs) == 2
