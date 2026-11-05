import json
import re
import time

import httpx

from nusalens.config import settings

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def generate(model: str, prompt: str, temperature: float = 0.0, num_predict: int = 500,
             endpoint: str | None = None) -> str:
    return _generate_raw(model, prompt, temperature, num_predict, endpoint)["response"]


def generate_with_meta(model: str, prompt: str, temperature: float = 0.0, num_predict: int = 500,
                        endpoint: str | None = None) -> tuple[str, int, int]:
    """Same as generate(), but also returns (prompt_tokens, completion_tokens) from Ollama's own eval counters."""
    data = _generate_raw(model, prompt, temperature, num_predict, endpoint)
    return data["response"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)


def _generate_raw(model: str, prompt: str, temperature: float, num_predict: int,
                   endpoint: str | None) -> dict:
    last_exc = None
    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{endpoint or settings.llm_endpoint}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": temperature, "num_predict": num_predict}},
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            # Ollama occasionally returns a transient 500 under sustained load (observed during
            # the Phase 9 120-item e2e run); retry a couple of times before giving up.
            last_exc = exc
            if exc.response.status_code < 500 or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise last_exc


def extract_json(text: str) -> dict | None:
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
