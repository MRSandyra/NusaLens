# UI labels are Indonesian per convention O7; retrieved evidence stays as-is.
import os
import re

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("NUSALENS_API_BASE", "http://localhost:8000")

_COUNTRY_RE = re.compile(r"country_name\s*=\s*'([^']+)'")
_INDICATOR_RE = re.compile(r"indicator_name\s*=\s*'([^']+)'")

st.set_page_config(page_title="NusaLens", page_icon="\U0001F1EE\U0001F1E9", layout="wide")
st.title("NusaLens")
st.caption("Asisten analitik data publik Indonesia -- indikator Bank Dunia dan artikel Wikipedia Indonesia.")

if "history" not in st.session_state:
    st.session_state.history = []


def fetch_series(country: str, indicator: str) -> pd.DataFrame | None:
    sql = (
        f"SELECT year, value FROM v_indicator_series WHERE country_name = '{country}' "
        f"AND indicator_name = '{indicator}' ORDER BY year"
    )
    try:
        resp = httpx.post(f"{API_BASE}/sql", json={"sql": sql}, timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
        df = pd.DataFrame(body["rows"], columns=body["columns"])
        # /sql returns NUMERIC values as JSON strings (Decimal is not a native JSON type); cast
        # back to numeric or st.line_chart renders a garbled, out-of-range x axis.
        df["year"] = df["year"].astype(int)
        df["value"] = df["value"].astype(float)
        return df.set_index("year")
    except Exception:
        return None


question = st.text_input("Pertanyaan", placeholder="Contoh: Berapa nilai Gini index untuk Vietnam pada tahun 2022?")
ask_clicked = st.button("Tanya", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Memproses pertanyaan..."):
        try:
            resp = httpx.post(f"{API_BASE}/ask", json={"question": question}, timeout=120.0)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            result = {"error": str(exc)}
    st.session_state.history.insert(0, {"question": question, "result": result})

if st.session_state.history:
    latest = st.session_state.history[0]
    result = latest["result"]

    if "error" in result:
        st.error(f"Gagal menjawab: {result['error']}")
    else:
        route_label = {"SQL": "Data terstruktur (SQL)", "DOC": "Dokumen (pencarian)",
                       "HYBRID": "Gabungan (SQL + dokumen)", "ABSTAIN": "Tidak dijawab"}.get(result["route"], result["route"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Rute", route_label)
        col2.metric("Keyakinan", f"{result['confidence']:.2f}")
        col3.metric("Status", "Abstain" if result["abstained"] else "Terjawab")

        st.subheader("Jawaban")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sumber")
            for s in result["sources"]:
                if s.get("url"):
                    st.markdown(f"- [{s['title']}]({s['url']})")
                else:
                    st.markdown(f"- {s['title']}")

        if result.get("sql"):
            with st.expander("Tampilkan SQL"):
                st.code(result["sql"], language="sql")

            country_match = _COUNTRY_RE.search(result["sql"])
            indicator_match = _INDICATOR_RE.search(result["sql"])
            if country_match and indicator_match:
                series = fetch_series(country_match.group(1), indicator_match.group(1))
                if series is not None and not series.empty:
                    st.subheader(f"Tren {indicator_match.group(1)} -- {country_match.group(1)}")
                    st.line_chart(series)

        if result["route"] in ("DOC", "HYBRID"):
            with st.expander("Tampilkan potongan dokumen yang diambil"):
                try:
                    search_resp = httpx.post(f"{API_BASE}/search", json={"query": latest["question"], "k": 8}, timeout=30.0)
                    search_resp.raise_for_status()
                    for chunk in search_resp.json()["chunks"]:
                        st.markdown(f"**[{chunk['chunk_id']}]** {chunk.get('title', '')}")
                        st.caption(chunk["text"][:400])
                except Exception as exc:
                    st.caption(f"Tidak dapat memuat potongan dokumen: {exc}")

        with st.expander("Latensi per tahap"):
            st.json(result.get("per_stage_latency_ms", {}))

st.divider()
st.subheader("Riwayat")
for item in st.session_state.history[1:11]:
    with st.expander(item["question"]):
        st.write(item["result"].get("answer", item["result"].get("error", "")))
