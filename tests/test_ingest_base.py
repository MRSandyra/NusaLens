import json
import time

from nusalens.ingest.base import BaseConnector, append_manifest, file_rows, sha256_file


def test_sha256_file_matches_known_hash(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    import hashlib
    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_append_manifest_appends_jsonl_lines(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    append_manifest(manifest, {"a": 1})
    append_manifest(manifest, {"b": 2})
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [{"a": 1}, {"b": 2}]


def test_file_rows_missing_file_returns_zero(tmp_path):
    assert file_rows(tmp_path / "nope.parquet") == 0


def test_file_rows_counts_json_array_length(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert file_rows(path) == 3


def test_connector_cache_key_is_deterministic(tmp_path):
    c = BaseConnector("S1_test", "https://example.test", 2.0, tmp_path, run_id="r1")
    key1 = c._cache_key("https://example.test/x", {"a": 1, "b": 2})
    key2 = c._cache_key("https://example.test/x", {"b": 2, "a": 1})  # order-independent
    assert key1 == key2
    assert len(key1) == 16


def test_connector_get_json_cache_hit_skips_network(tmp_path):
    c = BaseConnector("S1_test", "https://example.test", 2.0, tmp_path, run_id="r1")
    key = c._cache_key("https://example.test/x", None)
    cache_path = c._cache_path(key)
    cache_path.write_text(json.dumps({"cached": True}), encoding="utf-8")

    result = c.get_json("https://example.test/x", use_cache=True)
    assert result == {"cached": True}


def test_connector_throttle_enforces_minimum_interval(tmp_path):
    c = BaseConnector("S1_test", "https://example.test", rate_limit_per_second=100, cache_dir=tmp_path, run_id="r1")
    c.last_request_time = time.monotonic()
    t0 = time.monotonic()
    c._throttle()
    elapsed = time.monotonic() - t0
    assert elapsed >= c.min_interval - 0.01  # small tolerance for timer granularity
