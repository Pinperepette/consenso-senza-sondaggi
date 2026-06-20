#!/usr/bin/env python3
"""Ingestione dei dataset open-data REALI del Ministero dell'Interno a livello
comune (Eligendo Archivio).

Base download (scoperta dal portale):
  https://elezionistorico.interno.gov.it/daithome/documenti/opendata/<path>

Gestisce i due schemi reali:
  - Camera 2022:  COMUNE, CIRC-REG, ELETTORITOT, VOTANTITOT, SKBIANCHE, DESCRLISTA, VOTILISTA
  - Europee 2024: DESCCOMUNE, DESCREGIONE, DESCPROVINCIA, ELETTORI, VOTANTI, NUMSCHEDEBIANCHE, DESCLISTA, NUMVOTI

Aggrega i voti di lista per comune, costruisce le geografie comune->regione e
scrive party_results + turnout (comune e nazionale).
"""
from __future__ import annotations

import io
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import (GEOGRAPHIES, PARTY_RESULTS, TURNOUT)  # noqa: E402
from consenso.etl.base import archive_raw, http_get  # noqa: E402
from consenso.etl.geo import comune_id, ensure_nation, region_id, upsert_geography  # noqa: E402
from consenso.etl.sources.eligendo import register_election  # noqa: E402

BASE = "https://elezionistorico.interno.gov.it/daithome/documenti/opendata/"

# mappa nome regione (normalizzato) -> codice ISTAT
REGION_CODE = {
    "PIEMONTE": 1, "VALLE D AOSTA": 2, "VALLEE D AOSTE": 2, "LOMBARDIA": 3,
    "TRENTINO ALTO ADIGE": 4, "TRENTINO": 4, "VENETO": 5, "FRIULI VENEZIA GIULIA": 6,
    "LIGURIA": 7, "EMILIA ROMAGNA": 8, "TOSCANA": 9, "UMBRIA": 10, "MARCHE": 11,
    "LAZIO": 12, "ABRUZZO": 13, "MOLISE": 14, "CAMPANIA": 15, "PUGLIA": 16,
    "BASILICATA": 17, "CALABRIA": 18, "SICILIA": 19, "SARDEGNA": 20,
}

# colonne candidate per i vari schemi reali (2008/2009 storici inclusi)
COLS = {
    "comune": ["COMUNE", "DESCCOMUNE"],
    "region": ["DESCREGIONE", "REGIONE", "CIRC-REG", "CIRCOSCRIZIONE"],
    "eligible": ["ELETTORITOT", "ELETTORI"],
    "voters": ["VOTANTITOT", "VOTANTI"],
    "blank": ["SKBIANCHE", "NUMSCHEDEBIANCHE", "SCHEDE_BIANCHE"],
    "list": ["DESCRLISTA", "DESCLISTA", "LISTA"],
    "votes": ["VOTILISTA", "NUMVOTI", "VOTI_LISTA"],
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9 ]+", " ", s.upper()).strip()


def _region_to_istat(raw: str) -> str | None:
    n = _norm(raw)
    n = re.sub(r"\s*\d+$", "", n).strip()          # "PIEMONTE 1" -> "PIEMONTE"
    for key, code in REGION_CODE.items():
        if n.startswith(key):
            return region_id(str(code))
    return None


def _party_of(label: str) -> str | None:
    """Riconciliazione per continuità dell'elettorato (anche storica).

    PdL -> FI (lista di centrodestra a guida Berlusconi, continuazione principale);
    Sinistra Arcobaleno / Sinistra e Libertà / SEL -> AVS (lineage rosso-verde).
    M5S e FdI non esistevano prima del 2013: nelle elezioni vecchie sono assenti
    e quindi mascherati dal modello.
    """
    n = _norm(label)
    if "FRATELLI" in n: return "party:FDI"
    if "PARTITO DEMOCRATICO" in n: return "party:PD"
    if "MOVIMENTO 5 STELLE" in n: return "party:M5S"
    if any(k in n for k in ("VERDI E SINISTRA", "SINISTRA E LIBERTA",
                            "SINISTRA ARCOBALENO", "SINISTRA ECOLOGIA",
                            "RIFONDAZIONE COMUNISTA")):
        return "party:AVS"
    if "POPOLO DELLA LIBERTA" in n or "FORZA ITALIA" in n: return "party:FI"
    if "LEGA" in n: return "party:LEGA"
    return None                                     # -> confluisce in 'Altri'


def _pick(df: pd.DataFrame, names: list[str]) -> str:
    up = {c.upper(): c for c in df.columns}
    for n in names:
        if n.upper() in up:
            return up[n.upper()]
    raise KeyError(f"nessuna colonna fra {names} in {list(df.columns)}")


def ingest_opendata(election_id: str, etype: str, date: str, rel_path: str,
                    local: str | None = None) -> dict:
    raw = open(local, "rb").read() if local else http_get(BASE + rel_path)
    archive_raw(f"opendata:{rel_path}", raw, {"election_id": election_id})
    df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str, encoding="latin-1")
    c = {k: _pick(df, v) for k, v in COLS.items()}

    df["_com"] = df[c["comune"]].map(_norm)
    df["_reg"] = df[c["region"]].map(_region_to_istat)
    df["_party"] = df[c["list"]].map(_party_of)
    df["_votes"] = pd.to_numeric(df[c["votes"]].str.replace(".", "", regex=False)
                                 .str.replace(",", ".", regex=False), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["_reg"])
    df["_gid"] = "COM:" + df["_reg"].str[-2:] + ":" + df["_com"]

    register_election(election_id, etype, date, {"level": "nazionale"})
    ensure_nation()
    db = get_db()

    # geografie comune -> regione
    geos = df.drop_duplicates("_gid")[["_gid", "_reg", c["comune"]]]
    gops = [{"_id": r["_gid"], "level": "comune", "name": r[c["comune"]],
             "parent": r["_reg"], "region": r["_reg"]} for _, r in geos.iterrows()]
    for g in gops:
        upsert_geography(g)

    # turnout per comune (valori costanti per comune -> primo)
    tcols = [c["eligible"], c["voters"], c["blank"]]
    per_com = df.drop_duplicates("_gid").set_index("_gid")[tcols]
    def _i(x):
        try: return int(float(str(x).replace(".", "").replace(",", ".")))
        except Exception: return 0
    db[TURNOUT].delete_many({"election_id": election_id})
    tdocs = []
    tot_elig = tot_vot = 0
    for gid, row in per_com.iterrows():
        e, v = _i(row[c["eligible"]]), _i(row[c["voters"]])
        tot_elig += e; tot_vot += v
        tdocs.append({"election_id": election_id, "geo_id": gid, "geo_level": "comune",
                      "eligible": e, "voters": v, "turnout": (v/e) if e > 0 else None,
                      "blank": _i(row[c["blank"]]), "invalid": 0})
    # turnout nazionale aggregato (serve al modello per l'affluenza differenziale)
    tdocs.append({"election_id": election_id, "geo_id": "ISTAT:IT", "geo_level": "nazione",
                  "eligible": tot_elig, "voters": tot_vot,
                  "turnout": (tot_vot/tot_elig) if tot_elig else None})
    db[TURNOUT].insert_many(tdocs)

    # party_results: somma voti per (comune, partito); valid = somma voti lista comune
    grp = df.groupby(["_gid"]).agg(valid=("_votes", "sum")).to_dict()["valid"]
    by = df.groupby(["_gid", df["_party"].fillna("party:ALTRI"), c["list"]])["_votes"].sum()
    db[PARTY_RESULTS].delete_many({"election_id": election_id})
    rdocs = []
    for (gid, pid, lbl), v in by.items():
        rdocs.append({"election_id": election_id, "geo_id": gid, "geo_level": "comune",
                      "party_id": (None if pid == "party:ALTRI" else pid),
                      "raw_label": lbl, "votes": int(v),
                      "valid_votes_area": int(grp.get(gid, 0)), "share": None})
    for i in range(0, len(rdocs), 20000):
        db[PARTY_RESULTS].insert_many(rdocs[i:i+20000])

    return {"election_id": election_id, "comuni": len(per_com),
            "righe_risultati": len(rdocs), "affluenza_naz": round(tot_vot/tot_elig, 4)}


if __name__ == "__main__":
    from consenso.etl.features import compute_shares
    from consenso.etl.validate import validate_election
    JOBS = [
        ("elez:2022_politiche", "politiche", "2022-09-25",
         "catalogoagid/camera-2022-Italia-livcomune.csv", "data/raw/camera2022_comune.csv"),
        ("elez:2024_europee", "europee", "2024-06-09",
         "catalogoagid/europee-2024-italia-livcomune.csv", "data/raw/europee2024_comune.csv"),
    ]
    for eid, et, d, rel, loc in JOBS:
        loc = loc if Path(loc).exists() else None
        res = ingest_opendata(eid, et, d, rel, local=loc)
        compute_shares(eid)
        rep = validate_election(eid)
        nerr = sum(len(v) for v in rep.values())
        print(f"{eid}: comuni={res['comuni']} righe={res['righe_risultati']} "
              f"affluenza={res['affluenza_naz']:.1%} anomalie={nerr}")
    print("OPEN DATA COMUNALI CARICATI")
