CREATE TABLE IF NOT EXISTS dim_country (
    iso3 CHAR(3) PRIMARY KEY,
    name_id TEXT NOT NULL,
    name_en TEXT NOT NULL,
    region TEXT
);

CREATE TABLE IF NOT EXISTS dim_indicator (
    code TEXT PRIMARY KEY,
    name_id TEXT NOT NULL,
    name_en TEXT NOT NULL,
    unit TEXT,
    topic TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS fact_indicator (
    iso3 CHAR(3) REFERENCES dim_country(iso3),
    indicator_code TEXT REFERENCES dim_indicator(code),
    year INT NOT NULL,
    value NUMERIC,
    PRIMARY KEY (iso3, indicator_code, year)
);

CREATE TABLE IF NOT EXISTS dim_dataset (
    dataset_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    publisher TEXT,
    license TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    lang TEXT,
    license TEXT,
    published_at DATE,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_indicator_code_year ON fact_indicator(indicator_code, year);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_iso3_year ON fact_indicator(iso3, year);
