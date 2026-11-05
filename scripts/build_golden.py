"""Phase 4 golden set builder.

Questions are generated deterministically from real gold-layer rows, not drafted
by an LLM: the local qwen3:4b emits inline reasoning that cannot be disabled on
this Ollama build, costing 113s for a 60-token call (reports/raw/04_llm_latency_probe.txt),
which makes 120 drafted items infeasible. Every DOC/HYBRID item is tethered to a
real chunk_id and must survive a BM25 round-trip filter; every SQL/HYBRID item
carries gold SQL that is executed against Postgres and required to return rows.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psycopg
import pyarrow.parquet as pq

from nusalens.config import PROJECT_ROOT, settings

RANDOM_SEED = 20260930
STOPWORDS = {
    "yang", "dan", "atau", "dengan", "untuk", "pada", "dari", "dalam", "adalah",
    "ini", "itu", "sebagai", "oleh", "juga", "tidak", "akan", "telah", "dapat",
    "ke", "di", "the", "of", "and", "in", "to", "a", "is", "as", "by", "an",
    "lebih", "sangat", "karena", "bahwa", "para", "suatu", "atas", "antara",
    "cukup", "sehingga", "hanya", "masih", "sudah", "bisa", "harus", "kemudian",
    "namun", "tetapi", "tersebut", "adanya", "yaitu", "yakni", "seperti",
    "banyak", "beberapa", "setiap", "sekitar", "hingga", "sampai", "melalui",
    "terhadap", "secara", "sering", "selalu", "belum", "pernah", "jumlahnya",
    "merupakan", "menjadi", "memiliki", "terdapat", "digunakan", "dilakukan",
    "disebut", "berupa", "antaranya", "misalnya", "sedangkan", "sementara",
    "bahkan", "apabila", "sebuah", "seorang", "tahun", "orang", "bagian",
    "dimana", "kepada", "dengan", "adapun", "walaupun", "meskipun", "selain",
    "for", "with", "that", "this", "from", "are", "was", "were", "have", "has",
}

COUNTRY_ID = {
    "IDN": "Indonesia", "MYS": "Malaysia", "THA": "Thailand", "VNM": "Vietnam",
    "PHL": "Filipina", "SGP": "Singapura", "KHM": "Kamboja", "LAO": "Laos",
    "MMR": "Myanmar", "BRN": "Brunei Darussalam", "CHN": "Tiongkok",
    "JPN": "Jepang", "KOR": "Korea Selatan", "IND": "India",
    "AUS": "Australia", "NZL": "Selandia Baru",
}


@dataclass
class GoldenItem:
    qid: str
    question: str
    type: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_sql: str | None = None
    gold_answer_facts: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""


def content_terms(text: str, limit: int = 6, idf: dict[str, float] | None = None,
                  min_idf: float = 3.0) -> list[str]:
    """Pick distinctive terms, preferring high-IDF (corpus-rare) tokens."""
    seen: list[str] = []
    for token in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text):
        low = token.lower()
        if low in STOPWORDS or low in seen:
            continue
        if idf is not None and idf.get(low, 0.0) < min_idf:
            continue
        seen.append(low)
        if len(seen) >= limit:
            break
    return seen


class Bm25Index:
    def __init__(self, chunk_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.doc_terms: list[Counter] = []
        self.doc_len: list[int] = []
        df: Counter = Counter()
        for text in texts:
            terms = Counter(re.findall(r"\w+", text.lower()))
            self.doc_terms.append(terms)
            self.doc_len.append(sum(terms.values()))
            df.update(terms.keys())
        self.n = len(texts)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for idx, terms in enumerate(self.doc_terms):
            for term in terms:
                self.postings.setdefault(term, []).append(idx)

    def top_k(self, query: str, k: int) -> list[str]:
        q_terms = [t for t in re.findall(r"\w+", query.lower()) if t in self.idf]
        if not q_terms:
            return []
        scores: dict[int, float] = {}
        for term in set(q_terms):
            idf = self.idf[term]
            for idx in self.postings.get(term, ()):
                tf = self.doc_terms[idx][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self.chunk_ids[idx] for idx, _ in ranked]


def fmt_value(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def build_doc_items(chunks, documents, rng, target: int, idf: dict[str, float]) -> list[GoldenItem]:
    title_by_doc = {row["doc_id"]: row["title"] for _, row in documents.iterrows()}
    candidates = chunks[chunks["token_count"] >= 200]
    order = list(candidates.index)
    rng.shuffle(order)
    items: list[GoldenItem] = []
    used_docs: set[str] = set()
    for idx in order:
        if len(items) >= target * 3:
            break
        row = candidates.loc[idx]
        doc_id = row["doc_id"]
        if doc_id in used_docs:
            continue
        title = title_by_doc.get(doc_id, row["heading_path"])
        terms = content_terms(row["text"], limit=5, idf=idf)
        if len(terms) < 3 or not isinstance(title, str) or len(title) < 4:
            continue
        used_docs.add(doc_id)
        question = (
            f"Dalam artikel {title}, bagaimana {terms[0]} dijelaskan "
            f"kaitannya dengan {terms[1]} dan {terms[2]}?"
        )
        items.append(GoldenItem(
            qid="",
            question=question,
            type="DOC",
            gold_chunk_ids=[row["chunk_id"]],
            gold_answer_facts=terms[:3],
            difficulty="easy" if row["token_count"] >= 400 else "medium",
            notes=f"grounded in {doc_id}",
        ))
    return items


def build_sql_items(indicators, rng, target: int) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    nonnull = indicators[indicators["value"].notna()]
    pairs = (
        nonnull.groupby(["indicator_code", "country_iso3"])
        .size()
        .reset_index(name="n")
    )
    pairs = pairs[pairs["n"] >= 10]
    order = list(pairs.index)
    rng.shuffle(order)
    templates = ("latest", "max_country", "avg_country", "trend", "count_years")
    for i, pos in enumerate(order):
        if len(items) >= target * 3:
            break
        code = pairs.loc[pos, "indicator_code"]
        iso = pairs.loc[pos, "country_iso3"]
        grp = nonnull[
            (nonnull["indicator_code"] == code) & (nonnull["country_iso3"] == iso)
        ]
        if grp.empty:
            continue
        name = grp.iloc[0]["indicator_name"]
        kind = templates[i % len(templates)]
        year_max = int(grp["year"].max())
        country = COUNTRY_ID.get(iso, iso)

        if kind == "latest":
            q = f"Berapa nilai {name} untuk {country} pada tahun {year_max}?"
            sql = (
                "SELECT f.value FROM fact_indicator f JOIN dim_indicator i "
                "ON f.indicator_code = i.code "
                f"WHERE f.indicator_code = '{code}' AND f.iso3 = '{iso}' "
                f"AND f.year = {year_max}"
            )
        elif kind == "max_country":
            q = f"Negara mana yang memiliki {name} tertinggi pada tahun {year_max}?"
            sql = (
                "SELECT f.iso3, f.value FROM fact_indicator f "
                f"WHERE f.indicator_code = '{code}' AND f.year = {year_max} "
                "AND f.value IS NOT NULL ORDER BY f.value DESC LIMIT 1"
            )
        elif kind == "avg_country":
            q = f"Berapa rata-rata {name} untuk {country} sepanjang periode 2000 sampai {year_max}?"
            sql = (
                "SELECT AVG(f.value) FROM fact_indicator f "
                f"WHERE f.indicator_code = '{code}' AND f.iso3 = '{iso}' "
                f"AND f.year BETWEEN 2000 AND {year_max}"
            )
        elif kind == "trend":
            q = f"Bagaimana perubahan {name} di {country} antara tahun 2010 dan {year_max}?"
            sql = (
                "SELECT f.year, f.value FROM fact_indicator f "
                f"WHERE f.indicator_code = '{code}' AND f.iso3 = '{iso}' "
                f"AND f.year IN (2010, {year_max}) ORDER BY f.year"
            )
        else:
            q = f"Ada berapa tahun data {name} yang tersedia untuk {country}?"
            sql = (
                "SELECT COUNT(DISTINCT f.year) FROM fact_indicator f "
                f"WHERE f.indicator_code = '{code}' AND f.iso3 = '{iso}'"
            )

        items.append(GoldenItem(
            qid="",
            question=q,
            type="SQL",
            gold_sql=sql,
            difficulty="medium" if kind in ("latest", "count_years") else "hard",
            notes=f"indicator {code}",
        ))
    return items


def build_hybrid_items(chunks, documents, indicators, rng, target: int,
                       idf: dict[str, float]) -> list[GoldenItem]:
    title_by_doc = {row["doc_id"]: row["title"] for _, row in documents.iterrows()}
    idn = indicators[(indicators["country_iso3"] == "IDN") & indicators["value"].notna()]
    codes = sorted(idn["indicator_code"].unique())
    rng.shuffle(codes)
    candidates = chunks[chunks["token_count"] >= 250]
    order = list(candidates.index)
    rng.shuffle(order)

    items: list[GoldenItem] = []
    used_docs: set[str] = set()
    ptr = 0
    for idx in order:
        if len(items) >= target * 3 or ptr >= len(codes) * 4:
            break
        row = candidates.loc[idx]
        doc_id = row["doc_id"]
        if doc_id in used_docs:
            continue
        title = title_by_doc.get(doc_id, row["heading_path"])
        terms = content_terms(row["text"], limit=4, idf=idf)
        if len(terms) < 2 or not isinstance(title, str) or len(title) < 4:
            continue
        code = codes[ptr % len(codes)]
        ptr += 1
        used_docs.add(doc_id)
        sub = idn[idn["indicator_code"] == code]
        name = sub.iloc[0]["indicator_name"]
        year_max = int(sub["year"].max())
        question = (
            f"Ringkas pembahasan {terms[0]} dalam artikel {title}, "
            f"lalu sebutkan nilai {name} Indonesia tahun {year_max}."
        )
        sql = (
            "SELECT f.value FROM fact_indicator f "
            f"WHERE f.indicator_code = '{code}' AND f.iso3 = 'IDN' "
            f"AND f.year = {year_max}"
        )
        items.append(GoldenItem(
            qid="",
            question=question,
            type="HYBRID",
            gold_chunk_ids=[row["chunk_id"]],
            gold_sql=sql,
            gold_answer_facts=terms[:2],
            difficulty="hard",
            notes=f"chunk {doc_id} + indicator {code}",
        ))
    return items


def build_abstain_items() -> list[GoldenItem]:
    specs = [
        ("Berapa PDB per kapita Wakanda pada tahun 2023?", "fictional country"),
        ("Berapa nilai PDB per kapita Indonesia pada tahun 2087?", "year beyond coverage"),
        ("Siapa Menteri Urusan Bawah Laut Indonesia tahun 2020?", "nonexistent ministry"),
        ("Berapa jumlah penduduk Gondor menurut data Bank Dunia?", "fictional country"),
        ("Berapa tingkat inflasi di planet Mars pada tahun 2022?", "out of scope entity"),
        ("Berapa nilai ekspor unicorn Thailand pada tahun 2023?", "false premise commodity"),
        ("Provinsi mana di Indonesia dengan populasi putri duyung terbanyak?", "false premise"),
        ("Berapa angka harapan hidup robot di Singapura tahun 2024?", "false premise subject"),
        ("Berapa PDB Atlantis pada tahun 2015?", "fictional country"),
        ("Berapa produksi emas Narnia per tahun?", "fictional country"),
        ("Kementerian Indonesia mana yang dibubarkan pada tahun 2099?", "year beyond coverage"),
        ("Berapa jumlah penutur bahasa Klingon di Indonesia?", "out of scope entity"),
        ("Sungai mana di Indonesia yang mengalir ke arah atas?", "false premise"),
        ("Berapa suhu inti matahari menurut catatan BMKG?", "out of scope for corpus"),
        ("Berapa nilai PDB per kapita Indonesia pada tahun 1850?", "year before coverage"),
    ]
    return [
        GoldenItem(
            qid="",
            question=q,
            type="ABSTAIN",
            difficulty="hard",
            notes=note,
        )
        for q, note in specs
    ]


def verify_sql(items: list[GoldenItem], dsn: str) -> tuple[list[GoldenItem], int]:
    kept: list[GoldenItem] = []
    rejected = 0
    with psycopg.connect(dsn) as conn:
        for item in items:
            if not item.gold_sql:
                kept.append(item)
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(item.gold_sql)
                    rows = cur.fetchall()
            except Exception:
                conn.rollback()
                rejected += 1
                continue
            if not rows or all(r[0] is None for r in rows):
                rejected += 1
                continue
            item.gold_answer_facts = list(item.gold_answer_facts) + [
                fmt_value(r[0]) if isinstance(r[0], (int, float)) else str(r[0])
                for r in rows[:2]
            ]
            kept.append(item)
    return kept, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-rank", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    data_dir = settings.data_dir
    chunks = pq.read_table(data_dir / "gold" / "chunks.parquet").to_pandas()
    indicators = pq.read_table(data_dir / "gold" / "indicators.parquet").to_pandas()
    documents = pq.read_table(data_dir / "gold" / "documents.parquet").to_pandas()
    print(f"loaded chunks={len(chunks)} indicators={len(indicators)} documents={len(documents)}")

    index = Bm25Index(chunks["chunk_id"].tolist(), chunks["text"].tolist())
    print(f"bm25 index built over {index.n} chunks, avgdl={index.avgdl:.1f}")

    quotas = {"DOC": 45, "SQL": 35, "HYBRID": 25, "ABSTAIN": 15}

    doc_pool = build_doc_items(chunks, documents, rng, quotas["DOC"], index.idf)
    hybrid_pool = build_hybrid_items(chunks, documents, indicators, rng, quotas["HYBRID"], index.idf)
    sql_pool = build_sql_items(indicators, rng, quotas["SQL"])
    abstain_pool = build_abstain_items()
    print(f"pools: doc={len(doc_pool)} sql={len(sql_pool)} hybrid={len(hybrid_pool)} abstain={len(abstain_pool)}")

    stats = {"doc_rt_pass": 0, "doc_rt_fail": 0, "hybrid_rt_pass": 0, "hybrid_rt_fail": 0}

    def round_trip(pool: list[GoldenItem], quota: int, key: str) -> list[GoldenItem]:
        kept: list[GoldenItem] = []
        for item in pool:
            if len(kept) >= quota:
                break
            top = index.top_k(item.question, args.bm25_rank)
            if any(cid in top for cid in item.gold_chunk_ids):
                stats[f"{key}_rt_pass"] += 1
                kept.append(item)
            else:
                stats[f"{key}_rt_fail"] += 1
        return kept

    doc_items = round_trip(doc_pool, quotas["DOC"], "doc")
    hybrid_rt = round_trip(hybrid_pool, quotas["HYBRID"] * 2, "hybrid")

    dsn = os.environ.get("NUSALENS_DSN") or settings.db_dsn
    sql_verified, sql_rejected = verify_sql(sql_pool, dsn)
    hybrid_verified, hybrid_sql_rejected = verify_sql(hybrid_rt, dsn)
    sql_items = sql_verified[: quotas["SQL"]]
    hybrid_items = hybrid_verified[: quotas["HYBRID"]]
    abstain_items = abstain_pool[: quotas["ABSTAIN"]]
    print(f"sql verified kept={len(sql_verified)} rejected={sql_rejected}")
    print(f"hybrid verified kept={len(hybrid_verified)} rejected={hybrid_sql_rejected}")

    final = doc_items + sql_items + hybrid_items + abstain_items
    for n, item in enumerate(final, start=1):
        item.qid = f"{item.type.lower()}-{n:03d}"

    counts = Counter(i.type for i in final)
    print(f"round-trip: {stats}")
    print(f"final distribution: {dict(counts)} total={len(final)}")
    for t, q in quotas.items():
        status = "OK" if counts.get(t, 0) == q else "SHORT"
        print(f"  {t}: {counts.get(t, 0)}/{q} {status}")

    eval_dir = PROJECT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    golden_path = eval_dir / "golden.jsonl"
    with golden_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in final:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    digest = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    (eval_dir / "golden.sha256").write_text(
        f"{digest}  golden.jsonl\n", encoding="utf-8"
    )
    print(f"wrote {golden_path} rows={len(final)}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
