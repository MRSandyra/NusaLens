import re
from typing import Iterable

KEYWORDS = [
    "ekonomi", "perekonomian", "ekonomi indonesia",
    "provinsi", "kabupaten", "kota", "daerah khusus",
    "kementerian", "menteri", "pemerintah", "politik", "pemilihan umum",
    "kebijakan", "regulasi", "undang-undang", "peraturan",
    "demografi", "penduduk", "kependudukan",
    "infrastruktur", "transportasi", "jalan tol", "bandara",
    "pertanian", "perkebunan", "padi", "sawit",
    "industri", "manufaktur", "tambang", "minyak",
    "pendidikan", "sekolah", "universitas",
    "kesehatan", "rumah sakit", "puskesmas",
    "bank", "bank indonesia", "fintech",
    "energi", "listrik", "minyak bumi", "gas alam",
    "pariwisata", "budaya", "bahasa", "suku bangsa",
    "geografi", "pulau", "gunung", "sungai", "danau",
    "sejarah", "kemerdekaan", "orde baru", "reformasi",
]

DISAMBIGUATION_PATTERNS = [
    r"\(disambiguasi\)$",
    r"\(perusahaan\)$",
    r"\(film\)$",
    r"\(album\)$",
    r"\(novel\)$",
    r"\(lagu\)$",
    r"^daftar\s",
]

MIN_CHARS = 300


def passes_filter(title: str, text: str) -> bool:
    if not title or not text:
        return False
    if len(text) < MIN_CHARS:
        return False
    t = title.lower()
    for pat in DISAMBIGUATION_PATTERNS:
        if re.search(pat, t):
            return False
    haystack = (title + " " + text[:2000]).lower()
    matches = sum(1 for kw in KEYWORDS if kw in haystack)
    return matches >= 1


def filter_articles(articles: Iterable[dict]) -> list[dict]:
    """Keep articles whose title or lead body matches a category keyword."""
    kept = []
    for art in articles:
        title = art.get("title", "")
        text = art.get("text") or art.get("summary") or ""
        if passes_filter(title, text):
            kept.append({"title": title, "text": text, "url": art.get("url", "")})
    return kept