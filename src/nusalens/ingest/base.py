import time
import structlog
import hashlib
import json
import re
import httpx
from datetime import datetime, timezone
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = structlog.get_logger()


class SourceError(Exception):
    pass


class BaseConnector:
    """HTTP connector with retry, rate limiting, and raw response caching."""

    def __init__(
        self,
        source_id: str,
        base_url: str,
        rate_limit_per_second: float,
        cache_dir: Path,
        run_id: str,
        user_agent: str = "NusaLens/0.1 (portfolio project)",
    ):
        self.source_id = source_id
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / max(rate_limit_per_second, 0.01)
        self.cache_dir = Path(cache_dir) / source_id / run_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time: float = 0.0
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self.run_id = run_id

    def _throttle(self):
        elapsed = time.monotonic() - self.last_request_time
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _cache_key(self, url: str, params: dict | None) -> str:
        payload = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str, suffix: str = "json") -> Path:
        return self.cache_dir / f"{key}.{suffix}"

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True):
        key = self._cache_key(url, params)
        path = self._cache_path(key)
        if use_cache and path.exists():
            log.info("cache_hit", source=self.source_id, key=key)
            return json.loads(path.read_text(encoding="utf-8"))
        self._throttle()
        self.last_request_time = time.monotonic()
        try:
            with httpx.Client(timeout=30.0, headers=self.headers) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise SourceError(f"{type(exc).__name__}: {exc}") from exc
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.info("cache_store", source=self.source_id, key=key, url=url)
        return data

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(SourceError),
        reraise=True,
    )
    def fetch_with_retry(self, url: str, params: dict | None = None):
        return self.get_json(url, params=params, use_cache=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_manifest(manifest_path: Path, entry: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def file_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    elif path.suffix == ".json":
        return len(json.loads(path.read_text(encoding="utf-8")))
    return 0