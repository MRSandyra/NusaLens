"""Offline HTTP tests for startup/readiness failures and the evidence abstain contract."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import nusalens.api.app as api_module


pytestmark = pytest.mark.safe_integration


@pytest.fixture(autouse=True)
def clear_rate_limit():
    api_module._request_log.clear()


def test_health_reports_degraded_dependencies(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: {
        "database": "ok", "migrations": "pending (7/8)", "index": "empty", "ollama": "unavailable",
    })
    response = TestClient(api_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded", "database": "ok", "migrations": "pending (7/8)",
        "index": "empty", "ollama": "unavailable",
    }


def test_ask_fails_closed_when_ollama_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: {
        "database": "ok", "migrations": "ok", "index": "ok", "ollama": "unavailable",
    })
    response = TestClient(api_module.app).post("/ask", json={"question": "Apa itu kemiskinan?"})

    assert response.status_code == 503
    assert "ollama=unavailable" in response.json()["detail"]


def test_ask_fails_closed_when_migrations_are_pending(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: {
        "database": "ok", "migrations": "pending (7/8)", "index": "unavailable", "ollama": "ok",
    })
    response = TestClient(api_module.app).post("/ask", json={"question": "Apa itu kemiskinan?"})

    assert response.status_code == 503
    assert "migrations=pending (7/8)" in response.json()["detail"]


def test_search_fails_closed_when_index_is_empty(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: {
        "database": "ok", "migrations": "ok", "index": "empty", "ollama": "ok",
    })
    response = TestClient(api_module.app).post("/search", json={"query": "sejarah Indonesia"})

    assert response.status_code == 503
    assert "index=empty" in response.json()["detail"]


def test_malformed_request_is_rejected_before_dependency_work(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: pytest.fail("readiness must not run for malformed input"))
    response = TestClient(api_module.app).post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_insufficient_evidence_is_returned_as_an_abstention(monkeypatch):
    monkeypatch.setattr(api_module, "readiness", lambda: {
        "database": "ok", "migrations": "ok", "index": "ok", "ollama": "ok",
    })
    fake_result = SimpleNamespace(
        route="DOC", route_source="rule", summary="Bukti tidak cukup.", sources=[], sql=None,
        confidence=0.0, abstained=True, per_stage_latency_ms={"retrieve_ms": 1.0},
    )
    monkeypatch.setattr(api_module, "get_pipeline", lambda: SimpleNamespace(answer=lambda _: fake_result))
    response = TestClient(api_module.app).post("/ask", json={"question": "Topik di luar index?"})

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["answer"] == "Bukti tidak cukup."
