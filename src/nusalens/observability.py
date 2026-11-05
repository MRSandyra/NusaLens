# Mirrors the same fields query_log persists to Postgres into a flat JSONL file, so /metrics and a
# log-shipper can both read them without needing the database.
import structlog

from nusalens.config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "nusalens.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(file=LOG_PATH.open("a", encoding="utf-8")),
)

log = structlog.get_logger("nusalens.pipeline")
