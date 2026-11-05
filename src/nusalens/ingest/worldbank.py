import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .base import SourceError, append_manifest, file_rows, sha256_file


WORLDBANK_INDICATORS = [
    "NY.GDP.PCAP.CD", "NY.GDP.PCAP.KD.ZG", "NY.GDP.MKTP.KD.ZG",
    "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS", "SP.DYN.LE00.IN",
    "IT.NET.USER.ZS", "SP.URB.TOTL.IN.ZS", "SP.POP.TOTL",
    "SP.POP.GROW", "EN.POP.DNST", "NV.AGR.TOTL.ZS",
    "NV.IND.TOTL.ZS", "NV.SRV.TOTL.ZS", "BX.KLT.DINV.WD.GD.ZS",
    "BM.TRF.GOVF.XP.ZS", "SE.XPD.TOTL.SE.ZS", "SE.XPD.SECO.SE.ZS",
    "SE.XPD.TERT.SE.ZS", "SH.XPD.CHEX.PT.CD", "SH.XPD.GHED.PC.CD",
    "EG.ELC.ACCS.ZS", "EG.FEC.RNEW.ZS", "AG.LND.FRST.ZS",
    "AG.LND.TOTL.K2", "EN.ATM.CO2E.PC", "EN.ATM.PM25.MC.M3",
    "SI.POV.GINI", "SI.POV.DDAY", "GC.DOD.TOTL.GD.ZS",
    # Additional indicators with stronger data coverage for Asia
    "NY.GDP.DEFL.ZS", "FP.CPI.TOTL", "NE.EXP.GNFS.CD",
    "NE.IMP.GNFS.CD", "CM.MKT.LCAP.GD.ZS", "CM.MKT.TRAD.GD.ZS",
    # Third batch: socio-economic, health, education, trade
    "SE.PRM.ENRR", "SE.SEC.ENRR", "SE.TER.ENRR",
    "SL.AGR.EMPL.ZS", "SL.IND.EMPL.ZS", "SL.SRV.EMPL.ZS",
    "SH.DYN.MORT", "SH.IMM.IBCG", "SH.STA.MMRT",
    "EG.USE.PCAP.KG.OE", "IS.AIR.PSGR",
    "NE.TRD.GNFS.ZS", "NE.GDI.TOTL.ZS", "NE.CON.PRVT.ZS",
    "GC.TAX.TOTL.GD.ZS", "FS.AST.PRVT.GD.ZS", "FR.INR.RINR",
    "SP.DYN.TFRT.IN", "SP.DYN.IMRT.IN", "SP.ADO.TFRT",
    "AG.PRD.FOOD.XD", "AG.YLD.CREL.KG", "ER.H2O.FWTL.ZS",
]
# Panel widened beyond the original 6: with ~30 usable indicator codes the
# 6-country panel caps out near 8.5k rows, well short of the 20k DoD target.
ASEAN_COUNTRIES = [
    "IDN", "MYS", "THA", "VNM", "PHL", "SGP",
    "KHM", "LAO", "MMR", "BRN",
    "CHN", "JPN", "KOR", "IND", "AUS", "NZL",
]


def fetch_worldbank(connector, countries, per_page, bronze_root):
    pages = []
    skipped = []
    for indicator in WORLDBANK_INDICATORS:
        for country in countries:
            page = 1
            got_rows = 0
            while True:
                url = f"{connector.base_url}/country/{country}/indicator/{indicator}"
                params = {"format": "json", "per_page": per_page, "page": page}
                try:
                    payload = connector.fetch_with_retry(url, params=params)
                except SourceError as exc:
                    skipped.append({"country": country, "indicator": indicator, "page": page, "reason": str(exc)})
                    break
                if not isinstance(payload, list) or len(payload) < 2:
                    skipped.append({"country": country, "indicator": indicator, "page": page, "reason": "malformed_or_empty_envelope"})
                    break
                meta = payload[0]
                records = payload[1] or []
                if records:
                    pages.append({"country": country, "indicator": indicator, "page": page, "data": records})
                    got_rows += len(records)
                total_pages = meta.get("pages", 1)
                if page >= total_pages:
                    break
                page += 1
            if got_rows == 0 and not any(s["indicator"] == indicator and s["country"] == country for s in skipped):
                skipped.append({"country": country, "indicator": indicator, "page": 0, "reason": "no_observations_returned"})
    out_path = Path(bronze_root) / "worldbank_raw.json"
    out_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    skip_path = Path(bronze_root) / "worldbank_skipped.json"
    skip_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def normalize_worldbank(bronze_path, silver_path, gold_path):
    raw = json.loads(Path(bronze_path).read_text(encoding="utf-8"))
    rows = []
    for chunk in raw:
        for rec in chunk["data"]:
            if rec.get("value") is None:
                continue
            rows.append({
                "country_iso3": rec.get("countryiso3code") or chunk["country"],
                "country_value": rec.get("country", {}).get("value") if isinstance(rec.get("country"), dict) else chunk["country"],
                "indicator_code": rec.get("indicator", {}).get("id") or chunk["indicator"],
                "indicator_name": rec.get("indicator", {}).get("value") if isinstance(rec.get("indicator"), dict) else "",
                "year": int(rec["date"]),
                "value": float(rec["value"]),
                "unit": "",
            })
    seen = set()
    deduped = []
    for r in rows:
        key = (r["country_iso3"], r["indicator_code"], r["year"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rows = [r for r in deduped if 1960 <= r["year"] <= 2024]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, silver_path)
    pq.write_table(table, gold_path)
    return len(rows)


def run_worldbank(connector, run_id, bronze_root, gold_root, manifest_path, per_page):
    bronze_path = fetch_worldbank(connector, ASEAN_COUNTRIES, per_page, bronze_root)
    silver_path = bronze_root / "worldbank_silver.parquet"
    gold_path = gold_root / "indicators.parquet"
    count = normalize_worldbank(bronze_path, silver_path, gold_path)
    sha = sha256_file(gold_path)
    append_manifest(manifest_path, {
        "path": gold_path.relative_to(gold_root.parents[1]).as_posix(),
        "source": "S1_worldbank",
        "rows": count,
        "sha256": sha,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": "0.1.0",
    })
    return gold_path, count, sha