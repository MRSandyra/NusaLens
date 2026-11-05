from nusalens.ingest.wikipedia_filters import filter_articles, passes_filter

LONG_ECONOMY_TEXT = "Ekonomi Indonesia " + ("tumbuh pesat pada dekade ini. " * 20)
LONG_UNRELATED_TEXT = "Kucing adalah hewan peliharaan yang populer. " * 20


def test_passes_filter_rejects_short_text():
    assert passes_filter("Ekonomi", "terlalu pendek") is False


def test_passes_filter_rejects_text_under_minimum_even_with_keyword():
    text = "ekonomi " + ("a" * 290)  # 298 chars total, under MIN_CHARS=300
    assert len(text) < 300
    assert passes_filter("Ekonomi", text) is False


def test_passes_filter_accepts_long_text_with_keyword():
    assert passes_filter("Provinsi Jawa Barat", LONG_ECONOMY_TEXT) is True


def test_passes_filter_rejects_long_text_without_keyword():
    assert passes_filter("Kucing Peliharaan", LONG_UNRELATED_TEXT) is False


def test_passes_filter_rejects_disambiguation_pages():
    assert passes_filter("Jakarta (disambiguasi)", LONG_ECONOMY_TEXT) is False


def test_passes_filter_rejects_list_pages():
    assert passes_filter("Daftar kabupaten di Indonesia", LONG_ECONOMY_TEXT) is False


def test_passes_filter_rejects_empty_title_or_text():
    assert passes_filter("", LONG_ECONOMY_TEXT) is False
    assert passes_filter("Judul", "") is False


def test_filter_articles_keeps_only_passing_entries():
    articles = [
        {"title": "Provinsi Jawa Barat", "text": LONG_ECONOMY_TEXT, "url": "https://x/1"},
        {"title": "Kucing Peliharaan", "text": LONG_UNRELATED_TEXT, "url": "https://x/2"},
    ]
    kept = filter_articles(articles)
    assert len(kept) == 1
    assert kept[0]["title"] == "Provinsi Jawa Barat"
    assert kept[0]["url"] == "https://x/1"


def test_filter_articles_handles_summary_fallback_field():
    articles = [{"title": "Provinsi Jawa Barat", "summary": LONG_ECONOMY_TEXT, "url": ""}]
    kept = filter_articles(articles)
    assert len(kept) == 1
