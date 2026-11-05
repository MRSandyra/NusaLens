import json
import time
from pathlib import Path
from datetime import datetime, timezone

import typer

from .config import settings

app = typer.Typer(add_completion=False, help="NusaLens data subcommands")


@app.command()
def ingest(
    sources: str = typer.Option("S1,S2", help="Comma separated source ids"),
    run_id: str | None = typer.Option(None, help="Override run identifier"),
) -> None:
    """Run bronze ingestion for the requested sources and emit silver/gold Parquet."""
    from .ingest.base import BaseConnector, append_manifest, sha256_file
    from .ingest import worldbank, wikipedia

    requested = [s.strip() for s in sources.split(",") if s.strip()]
    bronze_root = Path(settings.data_dir) / "bronze"
    silver_root = Path(settings.data_dir) / "silver"
    gold_root = Path(settings.data_dir) / "gold"
    for d in (bronze_root, silver_root, gold_root):
        d.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = Path(settings.data_dir) / "manifest.jsonl"
    summary_rows = []

    if "S1" in requested:
        connector = BaseConnector(
            source_id="S1_worldbank",
            base_url="https://api.worldbank.org/v2",
            rate_limit_per_second=2.0,
            cache_dir=bronze_root,
            # Stable namespace, not run_id: a per-run cache dir can never be hit
            # on a second run, which defeats idempotency.
            run_id="cache",
        )
        gold_path, count, sha = worldbank.run_worldbank(
            connector=connector,
            run_id=run_id,
            bronze_root=bronze_root,
            gold_root=gold_root,
            manifest_path=manifest,
            per_page=1000,
        )
        summary_rows.append(("S1_worldbank", "data/gold/indicators.parquet", count))

    if "S2" in requested:
        target_max = 15000
        gold_path, count, sha, scanned = wikipedia.run_wikipedia(
            bronze_root=bronze_root,
            gold_root=gold_root,
            target_max=target_max,
        )
        append_manifest(manifest, {
            "path": "data/gold/documents.parquet",
            "source": "S2_wikipedia_id",
            "rows": count,
            "sha256": sha,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "tool_version": "0.1.0",
            "scanned": scanned,
        })
        summary_rows.append(("S2_wikipedia_id", "data/gold/documents.parquet", count))

    for source, path, rows in summary_rows:
        typer.echo(f"{source} -> {path}: {rows} rows")


@app.command()
def stats() -> None:
    """Print row counts for every gold artifact."""
    import pyarrow.parquet as pq

    gold_dir = Path(settings.data_dir) / "gold"
    if not gold_dir.exists():
        typer.echo("no gold directory")
        return
    for f in sorted(gold_dir.glob("*.parquet")):
        meta = pq.ParquetFile(f).metadata
        typer.echo(f"{f.name}: {meta.num_rows} rows")


if __name__ == "__main__":
    app()