#!/usr/bin/env python3
"""Ingestione delle elezioni regionali 2025 (archivio Min. Interno).

Ogni ZIP contiene il file *_SCRUTINI.csv con i voti di lista per comune; il file
di novembre raccoglie più regioni (Veneto/Campania/Puglia), che vanno splittate
in elezioni distinte. Le liste sono coalizionali e vengono riconciliate per
sottostringa (es. "FRATELLI D'ITALIA - GIORGIA MELONI" -> FdI).

Sono segnali recenti (ott-nov 2025): accorciano il salto temporale del nowcast.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import PARTY_RESULTS, TURNOUT  # noqa: E402
from consenso.etl.base import archive_raw, http_get  # noqa: E402
from consenso.etl.features import compute_shares  # noqa: E402
from consenso.etl.geo import ensure_nation, region_id, upsert_geography  # noqa: E402
from consenso.etl.sources.eligendo import register_election  # noqa: E402
from consenso.etl.validate import validate_election  # noqa: E402
from scripts.load_opendata import _norm, _party_of, _region_to_istat  # noqa: E402

BASE = "https://elezionistorico.interno.gov.it/daithome/documenti/opendata/"
JOBS = [
    ("regionali/regionali-20251005.zip", "2025-10-05"),
    ("regionali/regionali-20251012.zip", "2025-10-12"),
    ("regionali/regionali-20251123.zip", "2025-11-23"),
]


def _i(x) -> int:
    try:
        return int(float(str(x).replace(".", "").replace(",", ".")))
    except Exception:
        return 0


def ingest_region(df: pd.DataFrame, region: str, date: str) -> dict:
    rid = _region_to_istat(region)
    if not rid:
        return {"region": region, "skipped": "regione non mappata"}
    reg2 = rid[-2:]
    eid = f"elez:{date[:4]}_reg_{region.lower().replace(' ', '')}"
    db = get_db()
    upsert_geography({"_id": rid, "level": "regione", "name": region.title(),
                      "parent": "ISTAT:IT", "region": rid, "istat_code": str(int(reg2))})

    df = df.copy()
    df["_com"] = df["COMUNE"].map(_norm)
    df["_gid"] = "COM:" + reg2 + ":" + df["_com"]
    df["_party"] = df["LISTA"].map(_party_of)
    df["_votes"] = df["VOTI_LISTA"].map(_i)
    # una riga per (comune, lista): evita doppi conteggi sui candidati
    ld = df.drop_duplicates(["_gid", "LISTA"])

    register_election(eid, "regionali", date,
                      {"level": "regionale", "geo_ids": [rid]})
    # geografie comune -> regione
    for gid, com in ld.drop_duplicates("_gid")[["_gid", "COMUNE"]].itertuples(index=False):
        upsert_geography({"_id": gid, "level": "comune", "name": com,
                          "parent": rid, "region": rid})

    valid = ld.groupby("_gid")["_votes"].sum().to_dict()
    by = ld.groupby(["_gid", ld["_party"].fillna("party:ALTRI"), "LISTA"])["_votes"].sum()
    db[PARTY_RESULTS].delete_many({"election_id": eid})
    rdocs = [{"election_id": eid, "geo_id": gid, "geo_level": "comune",
              "party_id": (None if pid == "party:ALTRI" else pid),
              "raw_label": lbl, "votes": int(v), "valid_votes_area": int(valid.get(gid, 0)),
              "share": None}
             for (gid, pid, lbl), v in by.items()]
    if rdocs:
        db[PARTY_RESULTS].insert_many(rdocs)

    # affluenza per comune + aggregato regionale
    tt = ld.drop_duplicates("_gid")[["_gid", "ELETTORI", "VOTANTI"]]
    db[TURNOUT].delete_many({"election_id": eid})
    tdocs, te, tv = [], 0, 0
    for rec in tt.to_dict("records"):
        e, v = _i(rec["ELETTORI"]), _i(rec["VOTANTI"])
        te += e; tv += v
        tdocs.append({"election_id": eid, "geo_id": rec["_gid"], "geo_level": "comune",
                      "eligible": e, "voters": v, "turnout": (v / e) if e else None})
    tdocs.append({"election_id": eid, "geo_id": rid, "geo_level": "regione",
                  "eligible": te, "voters": tv, "turnout": (tv / te) if te else None})
    db[TURNOUT].insert_many(tdocs)

    compute_shares(eid)
    nerr = sum(len(x) for x in validate_election(eid).values())
    return {"election_id": eid, "comuni": len(tt), "righe": len(rdocs),
            "affluenza": round(tv / te, 3) if te else None, "anomalie": nerr}


def ingest_regionali_zip(raw: bytes, date: str) -> list:
    """Ingerisce un ZIP regionali (formato SCRUTINI corrente), splittando per
    regione. Riutilizzabile dall'auto-sync per qualsiasi tornata regionale."""
    ensure_nation()
    z = zipfile.ZipFile(io.BytesIO(raw))
    cand = [n for n in z.namelist() if "SCRUTINI" in n and "SEZ" not in n
            and n.lower().endswith(".csv")]
    if not cand:
        return [{"skipped": "nessun file SCRUTINI nel formato atteso"}]
    df = pd.read_csv(io.BytesIO(z.read(cand[0])), sep=";", dtype=str, encoding="latin-1")
    required = {"REGIONE", "COMUNE", "LISTA", "VOTI_LISTA", "ELETTORI", "VOTANTI"}
    if not required.issubset(df.columns):
        return [{"skipped": "schema regionali non riconosciuto",
                 "cols": list(df.columns)[:8]}]
    out = []
    for region in sorted(df["REGIONE"].dropna().unique()):
        out.append(ingest_region(df[df["REGIONE"] == region], region, date))
    return out


def ingest_comunali_zip(raw: bytes, date: str) -> list:
    """Ingerisce un ZIP comunali (formato SCRUTINI CSV). Una sola elezione per
    data, comuni di tutte le regioni. Le liste civiche locali confluiscono in
    'Altri' (non sono partiti nazionali).

    NB: le comunali sono ESCLUSE dal modello di consenso nazionale (dominanza di
    liste civiche) — qui si archivia solo il dato comune-level.
    """
    z = zipfile.ZipFile(io.BytesIO(raw))
    cand = [n for n in z.namelist() if "SCRUTINI" in n.upper() and "SEZ" not in n.upper()
            and n.lower().endswith(".csv")]
    if not cand:
        return [{"skipped": "comunali: formato non CSV (xlsx/old)"}]
    df = pd.read_csv(io.BytesIO(z.read(cand[0])), sep=";", dtype=str, encoding="latin-1")
    required = {"REGIONE", "COMUNE", "LISTA", "VOTI_LISTA", "ELETTORI", "VOTANTI"}
    if not required.issubset(df.columns):
        return [{"skipped": "comunali: schema non riconosciuto"}]

    ensure_nation()
    db = get_db()
    eid = f"elez:{date}_comunali"
    df = df.copy()
    df["_reg"] = df["REGIONE"].map(_region_to_istat)
    df = df.dropna(subset=["_reg"])
    df["_gid"] = "COM:" + df["_reg"].str[-2:] + ":" + df["COMUNE"].map(_norm)
    df["_party"] = df["LISTA"].map(_party_of)
    df["_votes"] = df["VOTI_LISTA"].map(_i)
    ld = df.drop_duplicates(["_gid", "LISTA"])

    register_election(eid, "comunali", date, {"level": "nazionale"})
    for gid, reg, com in ld.drop_duplicates("_gid")[["_gid", "_reg", "COMUNE"]].itertuples(index=False):
        upsert_geography({"_id": gid, "level": "comune", "name": com,
                          "parent": reg, "region": reg})

    valid = ld.groupby("_gid")["_votes"].sum().to_dict()
    by = ld.groupby(["_gid", ld["_party"].fillna("party:ALTRI"), "LISTA"])["_votes"].sum()
    db[PARTY_RESULTS].delete_many({"election_id": eid})
    rdocs = [{"election_id": eid, "geo_id": gid, "geo_level": "comune",
              "party_id": (None if pid == "party:ALTRI" else pid),
              "raw_label": lbl, "votes": int(v), "valid_votes_area": int(valid.get(gid, 0)),
              "share": None} for (gid, pid, lbl), v in by.items()]
    for i in range(0, len(rdocs), 20000):
        db[PARTY_RESULTS].insert_many(rdocs[i:i + 20000])

    tt = ld.drop_duplicates("_gid")[["_gid", "ELETTORI", "VOTANTI"]]
    db[TURNOUT].delete_many({"election_id": eid})
    tdocs = [{"election_id": eid, "geo_id": r["_gid"], "geo_level": "comune",
              "eligible": _i(r["ELETTORI"]), "voters": _i(r["VOTANTI"]),
              "turnout": (_i(r["VOTANTI"]) / _i(r["ELETTORI"])) if _i(r["ELETTORI"]) else None}
             for r in tt.to_dict("records")]
    if tdocs:
        db[TURNOUT].insert_many(tdocs)

    compute_shares(eid)
    nerr = sum(len(x) for x in validate_election(eid).values())
    return [{"election_id": eid, "comuni": len(tt), "righe": len(rdocs), "anomalie": nerr}]


def main() -> int:
    for rel, date in JOBS:
        raw = http_get(BASE + rel)
        archive_raw(f"opendata:{rel}", raw, {"date": date})
        for res in ingest_regionali_zip(raw, date):
            print(res)
    print("REGIONALI 2025 CARICATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
