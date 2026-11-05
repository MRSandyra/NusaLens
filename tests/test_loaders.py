import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
import pytest

from nusalens.config import settings
from nusalens.warehouse.seed import load_documents, seed_dim_country


def test_seed_dim_country_is_idempotent():
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM dim_country")
        before = cur.fetchone()[0]

    seed_dim_country(settings.db_dsn)  # ON CONFLICT DO NOTHING -- rerunning must not duplicate rows

    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM dim_country")
        after = cur.fetchone()[0]
        cur.execute("SELECT name_id FROM dim_country WHERE iso3 = 'IDN'")
        name = cur.fetchone()[0]

    assert after == before
    assert name == "Indonesia"


@pytest.fixture
def fixture_doc_parquet(tmp_path):
    table = pa.Table.from_pylist([{
        "doc_id": "test_loader_fixture:doc1",
        "source": "unit_test",
        "title": "Unit Test Fixture Document",
        "url": "https://example.test/fixture",
        "lang": "id",
        "license": "test",
        "published_at": None,
        "text": "Ini adalah dokumen fiktif untuk pengujian loader seed.py.",
    }])
    path = tmp_path / "documents.parquet"
    pq.write_table(table, path)
    return path


def test_load_documents_inserts_then_upserts_without_duplicating(fixture_doc_parquet):
    try:
        load_documents(fixture_doc_parquet, settings.db_dsn)
        load_documents(fixture_doc_parquet, settings.db_dsn)  # rerun: ON CONFLICT DO NOTHING

        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), title FROM documents WHERE doc_id = %s GROUP BY title",
                        ("test_loader_fixture:doc1",))
            rows = cur.fetchall()

        assert rows == [(1, "Unit Test Fixture Document")]
    finally:
        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", ("test_loader_fixture:doc1",))
            conn.commit()
