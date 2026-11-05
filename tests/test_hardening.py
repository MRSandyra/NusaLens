"""Phase 13: input validation regression tests (D30). The DB-down/LLM-down graceful-degradation
checks in the same decision are not re-simulated here (they temporarily repoint live settings at
an unreachable port, which is safe but not worth running on every pytest invocation); see
docs/DECISIONS.md D30 for that verification.
"""
from fastapi.testclient import TestClient

from nusalens.api.app import app

client = TestClient(app)


def test_ask_rejects_empty_question():
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_rejects_oversized_question():
    resp = client.post("/ask", json={"question": "x" * 1001})
    assert resp.status_code == 422


def test_search_rejects_empty_query():
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_sql_rejects_oversized_statement():
    resp = client.post("/sql", json={"sql": "SELECT 1 " * 300})
    assert resp.status_code == 422


def test_search_k_out_of_range_rejected():
    resp = client.post("/search", json={"query": "test", "k": 100})
    assert resp.status_code == 422
