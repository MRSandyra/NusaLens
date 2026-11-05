import os
from pathlib import Path
import psycopg
import pyarrow.parquet as pq


def load_documents(parquet_path: Path, dsn: str):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    records = list(
        df[["doc_id", "source", "title", "url", "lang", "license", "published_at", "text"]]
        .itertuples(index=False, name=None)
    )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_documents (LIKE documents) ON COMMIT DROP")
            with cur.copy(
                "COPY tmp_documents (doc_id, source, title, url, lang, license, published_at, text) FROM STDIN"
            ) as copy:
                for rec in records:
                    copy.write_row(rec)
            cur.execute("""
                INSERT INTO documents (doc_id, source, title, url, lang, license, published_at, text)
                SELECT doc_id, source, title, url, lang, license, published_at, text FROM tmp_documents
                ON CONFLICT (doc_id) DO NOTHING
            """)
        conn.commit()
    print(f"Upserted documents ({len(records)} rows)")


def seed_dim_country(dsn: str):
    countries = [
        ("IDN", "Indonesia", "Indonesia", "Asia"),
        ("MYS", "Malaysia", "Malaysia", "Asia"),
        ("THA", "Thailand", "Thailand", "Asia"),
        ("VNM", "Vietnam", "Vietnam", "Asia"),
        ("PHL", "Philippines", "Philippines", "Asia"),
        ("SGP", "Singapore", "Singapore", "Asia"),
        ("KHM", "Kamboja", "Cambodia", "Asia"),
        ("LAO", "Laos", "Laos", "Asia"),
        ("MMR", "Myanmar", "Myanmar", "Asia"),
        ("BRN", "Brunei Darussalam", "Brunei Darussalam", "Asia"),
        ("CHN", "Tiongkok", "China", "Asia"),
        ("JPN", "Jepang", "Japan", "Asia"),
        ("KOR", "Korea Selatan", "Korea, Rep.", "Asia"),
        ("IND", "India", "India", "Asia"),
        ("AUS", "Australia", "Australia", "Oceania"),
        ("NZL", "Selandia Baru", "New Zealand", "Oceania"),
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO dim_country (iso3, name_id, name_en, region) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                countries,
            )
        conn.commit()
    print("Seeded dim_country")


def seed_dim_indicator(parquet_path: Path, dsn: str):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()[["indicator_code", "indicator_name", "unit"]].drop_duplicates(
        subset=["indicator_code"]
    )
    rows = [
        (r.indicator_code, r.indicator_name, r.indicator_name, r.unit or "", "", "World Bank")
        for r in df.itertuples(index=False)
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO dim_indicator (code, name_id, name_en, unit, topic, source) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )
        conn.commit()
    print(f"Seeded dim_indicator ({len(rows)} distinct indicators)")


def seed_fact_indicator(parquet_path: Path, dsn: str):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()[["country_iso3", "indicator_code", "year", "value"]]
    records = list(df.itertuples(index=False, name=None))
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_indicators (LIKE fact_indicator) ON COMMIT DROP")
            with cur.copy("COPY tmp_indicators (iso3, indicator_code, year, value) FROM STDIN") as copy:
                for rec in records:
                    copy.write_row(rec)
            cur.execute("""
                INSERT INTO fact_indicator (iso3, indicator_code, year, value)
                SELECT iso3, indicator_code, year, value FROM tmp_indicators
                ON CONFLICT (iso3, indicator_code, year) DO UPDATE SET value = EXCLUDED.value
            """)
        conn.commit()
    print(f"Upserted fact_indicator ({len(records)} rows)")


if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL", "postgresql://nusalens:nusalens@localhost:55433/nusalens")
    indicators_parquet = Path("data/gold/indicators.parquet")
    documents_parquet = Path("data/gold/documents.parquet")
    seed_dim_country(dsn)
    if indicators_parquet.exists():
        seed_dim_indicator(indicators_parquet, dsn)
        seed_fact_indicator(indicators_parquet, dsn)
    else:
        print("No indicators.parquet found; skipping fact seed")
    if documents_parquet.exists():
        load_documents(documents_parquet, dsn)
    else:
        print("No documents.parquet found; skipping document load")
