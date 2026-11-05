# Phase 0: Preflight and Foundation

Machine capability, runtime availability, and project skeleton are verified. All figures below come from executed commands with raw output under `reports/raw/`.

## Machine specification

| Item | Value | Source |
|---|---|---|
| OS | Windows 11 (build 10.0.26200) | `reports/raw/00_hardware.json` |
| CPU | Intel64 Family 6 Model 141 Stepping 1 | `reports/raw/00_hardware.json` |
| Logical cores | 16 | `reports/raw/00_hardware.json` |
| RAM total | 16105 MB | `reports/raw/00_hardware.json` |
| RAM available at probe | 3555 MB | `reports/raw/00_hardware.json` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | `nvidia-smi` |
| VRAM total | 6144 MB | `nvidia-smi` |
| NVIDIA driver | 616.56 | `nvidia-smi` |
| Disk free (project volume) | 100.24 GB | `reports/raw/00_hardware.json` |
| Disk threshold (25 GB) | pass | `reports/raw/00_hardware.json` |

The 6144 MB VRAM figure confirms the R3 budget is the real physical limit on this machine, not a self-imposed cap.

## Component versions

| Component | Version | Source |
|---|---|---|
| Python (project venv) | 3.11.16 | `reports/raw/00_hardware.json` |
| uv | 0.12.13 | `uv --version` |
| torch | 2.6.0+cu124 | `reports/raw/00_hardware.json` |
| CUDA available to torch | true | `reports/raw/00_hardware.json` |
| CUDA device seen by torch | NVIDIA GeForce RTX 3060 Laptop GPU (6143 MB) | `reports/raw/00_hardware.json` |
| Docker Engine | 29.6.2 | `docker --version` |
| Docker Compose | v5.3.1 | `docker compose version` |
| PostgreSQL | 17.11 (Debian 17.11-1.pgdg12+2) | `reports/raw/00_db_version.txt` |
| pgvector extension | 0.8.6 | `reports/raw/00_db_vector_ext.txt` |
| Ollama | 0.34.1 | `reports/raw/00_llm_run.json` |
| sentence-transformers | 6.1.0 | `uv sync` resolution |
| sqlglot | 30.20.0 | `uv sync` resolution |

## LLM runtime measurement

Generator model `qwen3:4b` (GGUF Q4_K_M, 2.5 GB on disk), served by Ollama with CUDA offload, prompted in Indonesian. Raw record: `reports/raw/00_llm_run.json`.

| Metric | Value |
|---|---|
| Model | qwen3:4b |
| Prompt tokens | 35 |
| Generated tokens | 1907 |
| Generation duration (runtime-reported) | 172.21 s |
| Wall clock duration | 174.74 s |
| Tokens per second (runtime-reported) | 11.07 |
| Tokens per second (wall clock) | 10.91 |
| VRAM occupied with model resident | 4047 MB |
| VRAM budget | 6144 MB |
| Headroom | 2097 MB |
| Within budget | yes |

VRAM was sampled while the model was resident and serving, so 4047 MB is the steady-state footprint including KV cache at the default 4096-token context. The idle, post-load, and peak samples are identical because the measurement script warms the model before timing, which means load cost is excluded from the token rate.

Throughput of 11.07 tokens per second on a 4 B model at Q4_K_M is the practical ceiling for this laptop GPU. This figure sets the latency expectation for Phase 9: any synthesis prompt producing 300 tokens costs roughly 27 seconds of generation, so prompt and answer length are latency levers, not cosmetic choices.

## Deviations from specification

| Item | Specified | Actual | Reason |
|---|---|---|---|
| Generator model tag | Qwen3-4B-Instruct GGUF Q4_K_M | `qwen3:4b` | The registry has no `qwen3-4b-instruct` or `qwen2.5-3b-instruct` manifest. `qwen3:4b` is the same 4 B Qwen3 weights at Q4_K_M. Evidence of the failed pulls: `reports/raw/00_ollama_pull.log`, `reports/raw/00_ollama_pull_qwen25_3b.log`. |
| Alternative runtime | `llama-cpp-python[server]` | installed, GPU offload unavailable | The published wheel reports `llama_supports_gpu_offload() == False`, so it runs CPU-only and cannot honour the VRAM policy. Ollama is the CUDA-capable runtime and becomes primary. Kept as an installed dependency for CPU fallback. |
| Pre-existing local model | not specified | `qwen2.5-coder:7b` present | Attempting it crashed the runtime (`exit status 0xc0000409`, stack buffer overrun) because 7.6 B at Q4_K_M exceeds 6 GB VRAM with context. Recorded as the measured justification for staying at 4 B, consistent with R3 (never move to a larger model). |

## Observation carried into later phases

The Indonesian prompt produced an English answer. The model has the capability but not the instruction adherence by default, so language pinning belongs in the prompt templates rather than being assumed from the model. This is a Phase 8 concern and is recorded here because it was observed under measurement, not inferred.

## Limitations

RAM available at probe time was 3555 MB of 16105 MB total, measured with other desktop processes running. Phase 6 training and Phase 3 indexing should be run without a browser session resident, since the reranker alone reserves roughly 1.1 GB of host RAM.
