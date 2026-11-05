import pyarrow as pa
import pyarrow.parquet as pq

from nusalens.retrieval.chunker import chunk_documents


def _write_docs(tmp_path, rows):
    table = pa.Table.from_pylist(rows)
    path = tmp_path / "documents.parquet"
    pq.write_table(table, path)
    return path


def test_short_document_below_min_tokens_produces_no_chunk(tmp_path):
    path = _write_docs(tmp_path, [{"doc_id": "d1", "title": "T", "text": "cuma beberapa kata saja"}])
    out = chunk_documents(path)
    chunks = pq.read_table(out).to_pylist()
    assert chunks == []


def test_document_within_target_produces_one_chunk(tmp_path):
    text = " ".join(f"kata{i}" for i in range(100))
    path = _write_docs(tmp_path, [{"doc_id": "d1", "title": "T", "text": text}])
    out = chunk_documents(path, target_token=450, overlap=80)
    chunks = pq.read_table(out).to_pylist()
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "d1_0"
    assert chunks[0]["doc_id"] == "d1"
    assert chunks[0]["token_count"] == 100


def test_long_document_splits_with_overlap(tmp_path):
    text = " ".join(f"kata{i}" for i in range(1000))
    path = _write_docs(tmp_path, [{"doc_id": "d1", "title": "T", "text": text}])
    out = chunk_documents(path, target_token=450, overlap=80)
    chunks = pq.read_table(out).to_pylist()
    assert len(chunks) >= 2
    # consecutive chunks advance by (target_token - overlap) tokens, so the tail of one
    # chunk and the head of the next share exactly `overlap` tokens.
    first_tokens = chunks[0]["text"].split()
    second_tokens = chunks[1]["text"].split()
    assert first_tokens[-80:] == second_tokens[:80]
    assert [c["chunk_id"] for c in chunks] == [f"d1_{c['ord']}" for c in chunks]


def test_multiple_documents_are_chunked_independently(tmp_path):
    text = " ".join(f"kata{i}" for i in range(100))
    path = _write_docs(tmp_path, [
        {"doc_id": "a", "title": "A", "text": text},
        {"doc_id": "b", "title": "B", "text": text},
    ])
    out = chunk_documents(path)
    chunks = pq.read_table(out).to_pylist()
    assert {c["doc_id"] for c in chunks} == {"a", "b"}
