"""Phase 13: ingest -> index -> ask over a self-contained 50-document fixture.

Runs the real pipeline code (chunker, embed_chunks, HybridSearcher, Pipeline) against freshly
generated fixture content inserted into the live documents/chunks tables under a distinct
"integration_fixture:" id namespace, so it never collides with the real ~15k/41k corpus and is
fully cleaned up afterward regardless of outcome.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
import pytest

from nusalens.config import settings
from nusalens.retrieval.chunker import chunk_documents
from nusalens.retrieval.embed_chunks import embed_chunks
from nusalens.retrieval.hybrid_search import HybridSearcher
from nusalens.warehouse.seed import load_documents

N_DOCS = 50
FIXTURE_PREFIX = "integration_fixture:"

# 50 lexically distinct fictional place/mineral/animal name triples, not a shared template with
# only a numeric suffix changed -- near-duplicate text defeats embedding-based rank-1 precision
# (confirmed: with only a number differing, dense search returned a neighboring doc, not the
# queried one), which is a property of this test's own fixture design, not a retrieval bug.
_PLACES = [
    "Zubrovka", "Kallisto", "Wyrmhaven", "Thornmoor", "Caldera", "Ebonmere", "Frostholt",
    "Suncrest", "Ravenwick", "Brindlewood", "Starfall", "Mossgate", "Duskfen", "Silverglen",
    "Ashbourne", "Mirewick", "Goldspire", "Hollowmere", "Stormwatch", "Palewind",
    "Nightshade", "Emberfall", "Glimmerhold", "Thistledown", "Crowhaven", "Wintermere",
    "Amberfield", "Ironvale", "Whisperwood", "Sablecrest", "Larkspur", "Driftmoor",
    "Cinderpeak", "Foxglove", "Marrowgate", "Pinehollow", "Quillfen", "Ravencross",
    "Saltmarsh", "Thorncliff", "Umberlane", "Vesperhold", "Wraithmoor", "Yewbridge",
    "Zephyrholt", "Brambleford", "Copperdell", "Dovecrest", "Elmswick", "Foghaven",
]
_MINERALS = [
    "kersenium", "valtrite", "obsidion", "mornstone", "glacerite", "pyrolith", "aurvex",
    "crysanite", "duskamber", "feralite", "grimstone", "halbrite", "ironmoss", "jadewyrm",
    "kelvinite", "luminore", "mythralite", "nocturite", "opaline", "petravin", "quartzen",
    "ravenstone", "saphrite", "terravex", "umbrastone", "verdite", "windstone", "xylite",
    "yttrion", "zephyron", "ashstone", "briarite", "coralith", "drakite", "emberstone",
    "frostvein", "gloamite", "hexstone", "ivorine", "jetstone", "krakenite", "loamite",
    "mossvein", "nightstone", "orethorn", "prismite", "quillstone", "rimestone", "shardite",
    "thornite",
]


def _fixture_rows() -> list[dict]:
    rows = []
    for i in range(N_DOCS):
        place = _PLACES[i]
        mineral = _MINERALS[i]
        rows.append({
            "doc_id": f"{FIXTURE_PREFIX}doc{i:02d}",
            "source": "integration_test",
            "title": f"Kabupaten {place}",
            "url": f"https://example.test/{place.lower()}",
            "lang": "id",
            "license": "test",
            "published_at": None,
            # chunker.py drops anything under 80 tokens (min viable chunk) and splits anything
            # over target_token=450, so this needs to land reliably between the two -- padded
            # with varied filler sentences, not a naive repeat, to land around 110-120 words.
            "text": (
                f"Kabupaten {place} adalah sebuah wilayah fiktif dengan ibu kota "
                f"Kota Mendelsohn{i:02d}. Populasi kabupaten ini pada sensus fiktif adalah "
                f"{100000 + i * 137} jiwa, tersebar di beberapa kecamatan kecil di sekitar "
                f"Mendelsohn{i:02d}. Komoditas utama daerah ini adalah {mineral}, "
                f"sebuah mineral fiktif yang hanya ditambang di {place} dan tidak "
                f"ditemukan di wilayah fiktif lainnya. Pertambangan {mineral} menjadi "
                f"sumber mata pencaharian utama penduduk setempat sejak wilayah ini didirikan "
                f"dalam cerita fiktif ini, dengan hasil tambang diekspor melalui pelabuhan "
                f"kecil di selatan Mendelsohn{i:02d} menuju wilayah fiktif tetangga, yang "
                f"juga turut membeli hasil pertanian dan kerajinan tangan dari {place}."
            ),
        })
    return rows


@pytest.fixture(scope="module")
def fixture_corpus(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("integration_fixture")
    docs_path = tmp_path / "documents.parquet"
    pq.write_table(pa.Table.from_pylist(_fixture_rows()), docs_path)

    load_documents(docs_path, settings.db_dsn)
    chunks_path = chunk_documents(docs_path, target_token=450, overlap=80)
    embed_chunks(chunks_path, settings.db_dsn)

    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE chunk_id LIKE %s", (f"{FIXTURE_PREFIX}%",))
        n_chunks = cur.fetchone()[0]

    yield {"n_chunks": n_chunks}

    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE chunk_id LIKE %s", (f"{FIXTURE_PREFIX}%",))
        cur.execute("DELETE FROM documents WHERE doc_id LIKE %s", (f"{FIXTURE_PREFIX}%",))
        conn.commit()


def test_ingest_produced_one_chunk_per_fixture_document(fixture_corpus):
    # each fixture text is ~150 tokens, well under the 450-token chunking target, so chunking
    # should yield exactly one chunk per document.
    assert fixture_corpus["n_chunks"] == N_DOCS


def test_index_dense_search_finds_the_right_fixture_chunk():
    place, mineral = _PLACES[23], _MINERALS[23]
    searcher = HybridSearcher(settings.db_dsn, device="cpu")
    hits = searcher.search_rrf_rerank_exp(f"Apa komoditas utama {place}?", k=5)
    assert hits, "expected at least one hit"
    assert hits[0].chunk_id == f"{FIXTURE_PREFIX}doc23_0"
    assert mineral in hits[0].text


def test_index_sparse_component_is_populated_not_null():
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE chunk_id LIKE %s AND tsv IS NULL", (f"{FIXTURE_PREFIX}%",)
        )
        n_null = cur.fetchone()[0]
    assert n_null == 0  # embed_chunks() populates tsv directly now (Phase 13 fix, D24)


def test_ask_answers_from_the_fixture_corpus():
    from nusalens.pipeline import Pipeline

    place, mineral = _PLACES[7], _MINERALS[7]
    pipeline = Pipeline(device="cpu")
    result = pipeline.answer(
        f"Dalam artikel Kabupaten {place}, bagaimana {mineral} dijelaskan kaitannya dengan "
        "Mendelsohn dan populasi?"
    )
    assert result.route == "DOC"
    assert not result.abstained
    assert any(c.startswith(f"{FIXTURE_PREFIX}doc07") for c in result.retrieved_chunk_ids)
