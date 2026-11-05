from fastapi.testclient import TestClient

from nusalens.api.app import app

client = TestClient(app)


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_indicators_returns_catalog():
    resp = client.get("/indicators")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] > 0
    assert "code" in body["indicators"][0]


def test_sql_rejects_write_statement():
    resp = client.post("/sql", json={"sql": "DROP TABLE documents"})
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_sql_accepts_safe_select():
    resp = client.post("/sql", json={
        "sql": "SELECT value FROM v_indicator_series WHERE country_name = 'Indonesia' "
               "AND indicator_name = 'Population, total' AND year = 2022",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"]


def test_openapi_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp2 = client.get("/openapi.json")
    assert resp2.status_code == 200


def test_metrics_returns_structured_counts():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "query_count" in body
    assert "route_distribution" in body


def test_rate_limit_returns_429_after_threshold():
    for _ in range(31):
        resp = client.get("/health")
    assert resp.status_code == 429
    assert resp.json()["error_type"] == "RateLimited"
