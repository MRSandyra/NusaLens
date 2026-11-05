import json
from pathlib import Path

import httpx
import typer
import yaml

from .config import settings

app = typer.Typer(add_completion=False, help="NusaLens command line interface")
data_app = typer.Typer(add_completion=False, help="Data pipeline subcommands")
app.add_typer(data_app, name="data")

# data_cli.ingest() is the real S1/S2 connector-based ingestion that produced data/gold/*.parquet
# (confirmed against data/manifest.jsonl's own ingested_at records) -- it lived in its own never-
# mounted module, so `nusalens.cli data ingest` previously had no real command behind it at all.
# Phase 13 wires it in (docs/DECISIONS.md D27) so a fresh checkout can actually reproduce ingestion.
from . import data_cli as _data_cli  # noqa: E402

data_app.command(name="ingest")(_data_cli.ingest)


@app.callback()
def _root() -> None:
    return


def _check_database() -> dict:
    try:
        import psycopg
    except ImportError as exc:
        return {"status": "ERROR", "detail": f"psycopg not installed: {exc}"}
    try:
        with psycopg.connect(settings.db_dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                row = cur.fetchone()
        if row is None:
            return {"status": "ERROR", "detail": "vector extension not installed"}
        return {"status": "OK", "server_version": version.split(" on ")[0], "pgvector": row[0]}
    except Exception as exc:
        return {"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}


def _check_llm() -> dict:
    try:
        resp = httpx.get(f"{settings.llm_endpoint}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        return {"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}
    if settings.generator_model not in names:
        return {"status": "ERROR", "detail": f"{settings.generator_model} not pulled", "available": names}
    return {"status": "OK", "model": settings.generator_model, "available": names}


def _check_embedding() -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return {"status": "ERROR", "detail": f"sentence-transformers not installed: {exc}"}
    try:
        model = SentenceTransformer(settings.embedding_model, device="cpu")
        vector = model.encode("query: uji koneksi", normalize_embeddings=True)
        return {
            "status": "OK",
            "model": settings.embedding_model,
            "dimension": int(vector.shape[0]),
            "device": "cpu",
        }
    except Exception as exc:
        return {"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}


@app.command()
def health(as_json: bool = typer.Option(False, "--json", help="Emit machine readable JSON")) -> None:
    """Check database, LLM runtime, and embedding model availability."""
    checks = {
        "database": _check_database(),
        "llm": _check_llm(),
        "embedding": _check_embedding(),
    }
    overall = "OK" if all(c["status"] == "OK" for c in checks.values()) else "ERROR"
    report = {"overall": overall, "checks": checks}

    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        for name, result in checks.items():
            detail = result.get("detail", "")
            extra = f" ({detail})" if detail else ""
            typer.echo(f"{name}: {result['status']}{extra}")
        typer.echo(f"overall: {overall}")

    if overall != "OK":
        raise typer.Exit(code=1)


@app.command()
def serve(api_port: int = 8000, ui_port: int = 8501) -> None:
    """Start the FastAPI server in the background and the Streamlit UI in the foreground."""
    import os
    import subprocess
    import sys
    import threading

    import uvicorn

    def _run_api() -> None:
        uvicorn.run("nusalens.api.app:app", host="0.0.0.0", port=api_port, log_level="info")

    threading.Thread(target=_run_api, daemon=True).start()
    typer.echo(f"API starting on http://localhost:{api_port} (docs at /docs)")

    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    ui_env = {**os.environ, "NUSALENS_API_BASE": f"http://localhost:{api_port}"}
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_path),
        "--server.port", str(ui_port),
    ], env=ui_env)


@app.command()
def ask(question: str) -> None:
    """Route, answer, and print one question through the full pipeline."""
    from .pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.answer(question)
    typer.echo(f"[route: {result.route} via {result.route_source}]")
    typer.echo(result.formatted())


def _load_thresholds() -> dict:
    cfg_path = Path(settings.config_dir) / "settings.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("data_quality", {})


@data_app.command(name="stats")
def data_stats() -> None:
    """Print row counts for every gold Parquet artifact."""
    import pyarrow.parquet as pq

    gold_dir = Path(settings.data_dir) / "gold"
    if not gold_dir.exists():
        typer.echo("no gold directory")
        raise typer.Exit(code=1)
    found = False
    for f in sorted(gold_dir.glob("*.parquet")):
        found = True
        meta = pq.ParquetFile(f).metadata
        typer.echo(f"{f.name}: {meta.num_rows} rows")
    if not found:
        typer.echo("no gold parquet files")


@data_app.command(name="validate")
def data_validate(
    strict: bool = typer.Option(False, "--strict", help="Exit non zero on any failure"),
) -> None:
    """Check null ratio, duplicate primary key, year range, and empty text on every gold Parquet artifact."""
    import pyarrow.parquet as pq

    thresholds = _load_thresholds()
    max_null_ratio = float(thresholds.get("max_null_ratio", 0.05))
    min_year = int(thresholds.get("min_year", 1960))
    min_document_chars = int(thresholds.get("min_document_chars", 300))
    current_year = 2030  # Phase 1 DoD ceiling is "1960 to the current year".

    gold_dir = Path(settings.data_dir) / "gold"
    if not gold_dir.exists():
        typer.echo("no gold directory")
        raise typer.Exit(code=1)

    failures: list[str] = []
    for path in sorted(gold_dir.glob("*.parquet")):
        typer.echo(f"--- {path.name} ---")
        table = pq.read_table(path)
        df = table.to_pandas()
        rows = len(df)
        typer.echo(f"rows: {rows}")
        if "value" in df.columns:
            null_ratio = float(df["value"].isna().mean())
            typer.echo(f"null ratio on value: {null_ratio:.4f} (max {max_null_ratio})")
            if null_ratio > max_null_ratio:
                failures.append(f"{path.name}: null ratio {null_ratio:.4f} > {max_null_ratio}")
        if {"country_iso3", "indicator_code", "year"}.issubset(df.columns):
            pk_cols = ["country_iso3", "indicator_code", "year"]
            dupes = int(df.duplicated(subset=pk_cols).sum())
            typer.echo(f"duplicate pk (country,indicator,year): {dupes}")
            if dupes > 0:
                failures.append(f"{path.name}: {dupes} duplicate primary keys")
            outside = int(((df["year"] < min_year) | (df["year"] > current_year)).sum())
            typer.echo(f"rows with year outside [{min_year},{current_year}]: {outside}")
            if outside > 0:
                failures.append(f"{path.name}: {outside} rows outside year range")
        if "doc_id" in df.columns and path.name != "chunks.parquet":
            dupes = int(df.duplicated(subset=["doc_id"]).sum())
            typer.echo(f"duplicate doc_id: {dupes}")
            if dupes > 0:
                failures.append(f"{path.name}: {dupes} duplicate doc_ids")
            if "text" in df.columns:
                lens = df["text"].fillna("").str.len()
                short = int((lens < min_document_chars).sum())
                empty = int((lens == 0).sum())
                typer.echo(
                    f"documents under {min_document_chars} chars: {short} (empty: {empty})"
                )
                if short > 0:
                    failures.append(f"{path.name}: {short} documents shorter than {min_document_chars} chars")

    typer.echo("")
    if failures:
        typer.echo("FAIL:")
        for f in failures:
            typer.echo(f"  - {f}")
        if strict:
            raise typer.Exit(code=1)
    else:
        typer.echo("OK: all data quality checks passed")


if __name__ == "__main__":
    app()