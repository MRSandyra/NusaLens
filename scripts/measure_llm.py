import json
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3:4b"
ENDPOINT = "http://localhost:11434/api/chat"
PROMPT = "Sebutkan tiga sektor utama ekonomi Indonesia dan jelaskan masing-masing dalam satu kalimat."


def vram_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    vram_idle = vram_used_mb()

    # Warm load so the measured generation excludes model load time.
    post({"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "stream": False})
    vram_loaded = vram_used_mb()

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    start = time.perf_counter()
    data = post(payload)
    wall_seconds = time.perf_counter() - start
    vram_peak = vram_used_mb()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    tps_eval = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
    tps_wall = eval_count / wall_seconds if wall_seconds else 0.0

    result = {
        "runtime": "ollama",
        "runtime_version": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip(),
        "model": MODEL,
        "prompt": PROMPT,
        "response": data.get("message", {}).get("content", ""),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "wall_seconds": round(wall_seconds, 3),
        "tokens_per_second_eval": round(tps_eval, 2),
        "tokens_per_second_wall": round(tps_wall, 2),
        "vram_idle_mb": vram_idle,
        "vram_after_load_mb": vram_loaded,
        "vram_peak_mb": vram_peak,
        "vram_model_footprint_mb": vram_loaded - vram_idle,
        "vram_budget_mb": 6144,
    }
    result["within_vram_budget"] = result["vram_peak_mb"] <= result["vram_budget_mb"]

    out = Path("reports/raw/00_llm_run.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
