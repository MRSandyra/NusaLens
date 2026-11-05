from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 55433
    db_name: str = "nusalens"
    db_user: str = "nusalens"
    db_password: str = "nusalens"
    db_ro_user: str = "nusalens_ro"
    db_ro_password: str = "readonly_pass"

    llm_endpoint: str = "http://localhost:11434"
    generator_model: str = "qwen3:4b"
    # qwen3:4b spends its token budget on chain-of-thought and never emits a
    # clean SQL statement or short classification label at any practical
    # budget (see docs/DECISIONS.md D13, D14); these roles use a non-reasoning
    # instruct model instead. generator_model (qwen3:4b) stays the synthesizer.
    text2sql_model: str = "qwen2.5:3b-instruct"
    router_model: str = "qwen2.5:3b-instruct"
    judge_model: str = "qwen2.5:3b-instruct"
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-base"

    retrieval_mode: str = "rrf_rerank_exp"
    rrf_k: int = 60
    top_k_dense: int = 50
    top_k_sparse: int = 50
    rerank_top_n: int = 15
    final_top_k: int = 8

    # Phase 9 abstain gate / fast-router thresholds. Years measured directly from
    # fact_indicator (SELECT min(year), max(year)), not assumed; reranker threshold measured
    # from BAAI/bge-reranker-base scores on known relevant (~0.2-1.0) vs irrelevant (~3.7e-5)
    # pairs, see docs/DECISIONS.md D17.
    data_min_year: int = 1960
    data_max_year: int = 2024
    abstain_rerank_threshold: float = 0.01

    log_level: str = "INFO"

    # Applied via the DSN itself (a libpq connection parameter) rather than passed to every one
    # of the ~25 call sites that do `psycopg.connect(settings.db_dsn)`: without it, an unreachable
    # database blocks on the OS-level TCP timeout (often minutes), not a graceful failure (Phase 13
    # hardening, docs/DECISIONS.md).
    db_connect_timeout_s: int = 5

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?connect_timeout={self.db_connect_timeout_s}"
        )

    @property
    def db_ro_dsn(self) -> str:
        return (
            f"postgresql://{self.db_ro_user}:{self.db_ro_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?connect_timeout={self.db_connect_timeout_s}"
        )

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def config_dir(self) -> Path:
        return PROJECT_ROOT / "config"


settings = Settings()
