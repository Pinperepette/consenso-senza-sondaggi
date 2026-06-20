#!/usr/bin/env python3
"""Ingestione del dataset comunali ricco (catalogoagid) per la vista comune-level.

Il file catalogoagid/comunali-*.csv raccoglie più tornate (campo DATAELEZIONE) con
i voti di lista per comune (DESCRLISTA/VOTILISTA), su centinaia di comuni incluse
le grandi città. Lo splittiamo per data in elezioni 'comunali' comune-level.

NB: dato escluso dal modello nazionale (liste civiche dominanti) — serve solo per
la mappa comunale "dove i partiti nazionali sfondano localmente".
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import PARTY_RESULTS  # noqa: E402
from consenso.etl.base import archive_raw, http_get  # noqa: E402
from consenso.etl.features import compute_shares  # noqa: E402
from consenso.etl.geo import ensure_nation, upsert_geography  # noqa: E402
from consenso.etl.sources.eligendo import register_election  # noqa: E402
from scripts.load_opendata import _norm, _party_of, _region_to_istat  # noqa: E402
from scripts.load_regionali_2025 import _i  # noqa: E402

BASE = "https://elezionistorico.interno.gov.it/daithome/documenti/opendata/"
FILES = ["catalogoagid/comunali-20240609.csv", "catalogoagid/comunali-20240728.csv"]


def _date(s: str):
    try:
        return datetime.strptime(str(s).split(" ")[0], "%d/%m/%Y").date().isoformat()
    except Exception:
        return None


def ingest(rel: str) -> list:
    raw = http_get(BASE + rel)
    archive_raw(f"opendata:{rel}", raw, {})
    df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str, encoding="latin-1")
    need = {"DATAELEZIONE", "REGIONE", "COMUNE", "DESCRLISTA", "VOTILISTA"}
    if not need.issubset(df.columns):
        return [{"skipped": rel, "cols": list(df.columns)[:8]}]
    ensure_nation()
    db = get_db()
    df["_date"] = df["DATAELEZIONE"].map(_date)
    df["_reg"] = df["REGIONE"].map(_region_to_istat)
    df = df.dropna(subset=["_date", "_reg"])
    df["_gid"] = "COM:" + df["_reg"].str[-2:] + ":" + df["COMUNE"].map(_norm)
    df["_party"] = df["DESCRLISTA"].map(_party_of)
    df["_votes"] = df["VOTILISTA"].map(_i)
    out = []
    for date, g in df.groupby("_date"):
        eid = f"elez:{date}_comunali"
        ld = g.drop_duplicates(["_gid", "DESCRLISTA"])
        register_election(eid, "comunali", date, {"level": "nazionale"})
        for gid, reg, com in ld.drop_duplicates("_gid")[["_gid", "_reg", "COMUNE"]].itertuples(index=False):
            upsert_geography({"_id": gid, "level": "comune", "name": com,
                              "parent": reg, "region": reg})
        valid = ld.groupby("_gid")["_votes"].sum().to_dict()
        by = ld.groupby(["_gid", ld["_party"].fillna("party:ALTRI"), "DESCRLISTA"])["_votes"].sum()
        db[PARTY_RESULTS].delete_many({"election_id": eid})
        rdocs = [{"election_id": eid, "geo_id": gid, "geo_level": "comune",
                  "party_id": (None if pid == "party:ALTRI" else pid), "raw_label": lbl,
                  "votes": int(v), "valid_votes_area": int(valid.get(gid, 0)), "share": None}
                 for (gid, pid, lbl), v in by.items()]
        for i in range(0, len(rdocs), 20000):
            db[PARTY_RESULTS].insert_many(rdocs[i:i + 20000])
        compute_shares(eid)
        out.append({"election_id": eid, "comuni": len(valid), "righe": len(rdocs)})
    return out


def main() -> int:
    for rel in FILES:
        try:
            for r in ingest(rel):
                print(r)
        except Exception as e:
            print(rel, "ERR", repr(e)[:120])
    print("COMUNALI CATALOGO CARICATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
