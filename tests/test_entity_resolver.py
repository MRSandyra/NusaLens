import pytest
from sentence_transformers import SentenceTransformer

from nusalens.agents.entity_resolution import EntityResolver
from nusalens.config import settings


@pytest.fixture(scope="module")
def resolver():
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    return EntityResolver(settings.db_dsn, model)


def test_alias_table_covers_iso3_and_both_names(resolver):
    assert resolver.alias_to_canonical["idn"] == "Indonesia"
    assert resolver.alias_to_canonical["vietnam"] == "Vietnam"
    # name_en for Korea differs from name_id ("Korea, Rep." vs "Korea Selatan")
    assert resolver.alias_to_canonical["korea, rep."] == "Korea Selatan"


def test_resolve_exact_country_mention_ranks_first(resolver):
    candidates = resolver.resolve("Berapa PDB Vietnam pada tahun 2022?", top_k=5)
    assert candidates.countries[0] == "Vietnam"


def test_resolve_exact_indicator_mention_ranks_first(resolver):
    candidates = resolver.resolve("Berapa Gini index untuk Indonesia?", top_k=5)
    assert candidates.indicators[0] == "Gini index"


def test_resolve_paraphrased_indicator_still_surfaces_via_embedding(resolver):
    candidates = resolver.resolve("Berapa PDB per kapita Indonesia?", top_k=10)
    assert "GDP per capita (current US$)" in candidates.indicators


def test_resolve_unrelated_question_returns_top_k_candidates_not_empty(resolver):
    # No hard "doesn't exist" signal from resolve() itself (that's the router's job, D17) --
    # it always ranks and returns candidates, even for a nonsensical entity.
    candidates = resolver.resolve("Berapa PDB Wakanda?", top_k=5)
    assert len(candidates.countries) == 5
    assert len(candidates.indicators) == 5
