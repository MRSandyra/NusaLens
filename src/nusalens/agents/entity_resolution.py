# Country names in dim_country.name_id are already Indonesian ("Jepang", "Korea Selatan"), so an
# exact substring match against the question handles most cases. Indicator names are the
# untranslated World Bank English strings, so a paraphrased Indonesian mention ("PDB per kapita")
# needs the embedding fallback to reach "GDP per capita (current US$)".
import difflib
from dataclasses import dataclass

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer


@dataclass
class EntityCandidates:
    countries: list[str]
    indicators: list[str]


class EntityResolver:
    def __init__(self, dsn: str, model: SentenceTransformer):
        self.model = model
        self.alias_to_canonical: dict[str, str] = {}
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT iso3, name_id, name_en FROM dim_country ORDER BY name_id")
            country_rows = cur.fetchall()
            cur.execute("SELECT code, name_id, name_en FROM dim_indicator ORDER BY name_id")
            indicator_rows = cur.fetchall()

        self.country_names = [r[1] for r in country_rows]
        self.indicator_names = [r[1] for r in indicator_rows]
        for iso3, name_id, name_en in country_rows:
            for alias in (iso3, name_id, name_en):
                self.alias_to_canonical[alias.lower()] = name_id
        for code, name_id, name_en in indicator_rows:
            for alias in (code, name_id, name_en):
                self.alias_to_canonical[alias.lower()] = name_id

        self.country_emb = self._embed_passages(self.country_names)
        self.indicator_emb = self._embed_passages(self.indicator_names)

    def _embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.asarray(
            self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True),
            dtype=np.float32,
        )

    def _rank(self, question: str, names: list[str], emb: np.ndarray, top_k: int) -> list[str]:
        if not names:
            return []
        q_lower = question.lower()
        exact = [n for n in names if n.lower() in q_lower]

        fuzzy = difflib.get_close_matches(question, names, n=top_k, cutoff=0.5)

        qvec = self.model.encode(f"query: {question}", normalize_embeddings=True)
        sims = emb @ qvec
        order = np.argsort(-sims)[:top_k]
        semantic = [names[i] for i in order]

        ranked: list[str] = []
        for group in (exact, fuzzy, semantic):
            for n in group:
                if n not in ranked:
                    ranked.append(n)
        return ranked[:top_k]

    def resolve(self, question: str, top_k: int = 5) -> EntityCandidates:
        return EntityCandidates(
            countries=self._rank(question, self.country_names, self.country_emb, top_k),
            indicators=self._rank(question, self.indicator_names, self.indicator_emb, top_k),
        )
