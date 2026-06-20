"""Modello ML spaziale (stile MRP): predice la quota di un partito in un comune
dalle caratteristiche del territorio.

È il membro "machine learning" dell'ensemble (doc: modelli alternativi). NON
serve per il nowcast temporale (poche epoche, estrapolazione) ma per la
dimensione *spaziale*: capire e predire DOVE i partiti sono forti, con driver
interpretabili (feature importance).

Feature reali per comune:
  - latitudine, longitudine (gradiente geografico Nord-Sud, fortissimo in Italia)
  - popolazione residente (urbanizzazione) e ampiezza dell'elettorato
  - affluenza
  - macro-area (Nord-Ovest/Nord-Est/Centro/Sud/Isole)

Fonti reali: risultati Min. Interno (per comune) + anagrafica comuni con
coordinate e popolazione (opendatasicilia/comuni-italiani, dati ISTAT).
"""
from __future__ import annotations

import io
import os
import unicodedata
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from consenso.db.client import get_db
from consenso.db.schema import PARTY_RESULTS, TURNOUT

GEO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PARTIES = ["party:FDI", "party:PD", "party:M5S", "party:LEGA", "party:FI", "party:AVS"]

MACRO = {1: "NO", 2: "NO", 3: "NO", 7: "NO",          # Nord-Ovest
         4: "NE", 5: "NE", 6: "NE", 8: "NE",          # Nord-Est
         9: "C", 10: "C", 11: "C", 12: "C",           # Centro
         13: "S", 14: "S", 15: "S", 16: "S", 17: "S", 18: "S",   # Sud
         19: "I", 20: "I"}                             # Isole


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9 ]+", " ", s.upper()).strip()


def _load_mef_income() -> pd.DataFrame:
    """Reddito MEF per comune (ISTAT SDMX): reddito medio + struttura occupazionale.

    Codici: NTAXP=contribuenti, TAXABINCR/F=imponibile ammontare/frequenza,
    PENSINCF=pensionati, SELFEMINCF=autonomi, SUBEMPTRINCF=dipendenti.
    """
    path = os.path.join(GEO_DIR, "mef_redditi.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["pro_com_t"])
    m = pd.read_csv(path, dtype=str)
    m = m[(m["AMOUNT_CLASS"] == "TOTAL")]
    m["yr"] = pd.to_numeric(m["TIME_PERIOD"], errors="coerce")
    m = m.sort_values("yr").drop_duplicates(["REF_AREA", "DATA_TYPE"], keep="last")
    m["v"] = pd.to_numeric(m["OBS_VALUE"], errors="coerce")
    piv = m.pivot_table(index="REF_AREA", columns="DATA_TYPE", values="v", aggfunc="last")
    out = pd.DataFrame(index=piv.index)
    ntax = piv.get("NTAXP")
    out["avg_income"] = piv.get("TAXABINCR") / piv.get("TAXABINCF")
    out["pct_pension"] = piv.get("PENSINCF") / ntax
    out["pct_selfemp"] = piv.get("SELFEMINCF") / ntax
    out["pct_employee"] = piv.get("SUBEMPTRINCF") / ntax
    out = out.reset_index().rename(columns={"REF_AREA": "pro_com_t"})
    out["pro_com_t"] = out["pro_com_t"].str.zfill(6)
    return out


def _load_geo_anagrafica() -> pd.DataFrame:
    main = pd.read_csv(os.path.join(GEO_DIR, "main.csv"), dtype=str)
    pop = pd.read_csv(os.path.join(GEO_DIR, "popolazione_2021.csv"), dtype=str)
    pop["pro_com_t"] = pop["pro_com_t"].str.zfill(6)
    main["pro_com_t"] = main["pro_com_t"].str.zfill(6)
    df = main.merge(pop, on="pro_com_t", how="left").merge(
        _load_mef_income(), on="pro_com_t", how="left")
    df["nkey"] = df["comune"].map(_norm)
    df["reg2"] = df["cod_reg"].astype(int).map(lambda c: f"{c:02d}")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["pop"] = pd.to_numeric(df["pop_res_21"], errors="coerce")
    df["macro"] = df["cod_reg"].astype(int).map(MACRO)
    return df[["nkey", "reg2", "lat", "long", "pop", "macro",
               "avg_income", "pct_pension", "pct_selfemp", "pct_employee"]]


def build_dataset(election_id: str) -> pd.DataFrame:
    """Unisce risultati per comune + anagrafica geografica in una tabella ML."""
    db = get_db()
    # quote per partito (pivot comune x partito)
    rows = list(db[PARTY_RESULTS].find(
        {"election_id": election_id, "geo_level": "comune", "party_id": {"$ne": None}},
        {"geo_id": 1, "party_id": 1, "share": 1}))
    res = pd.DataFrame(rows)
    if res.empty:
        raise ValueError("nessun risultato comunale per " + election_id)
    piv = res.pivot_table(index="geo_id", columns="party_id", values="share", aggfunc="first")
    # affluenza ed elettorato
    tr = pd.DataFrame(list(db[TURNOUT].find(
        {"election_id": election_id, "geo_level": "comune"},
        {"geo_id": 1, "eligible": 1, "turnout": 1}))).set_index("geo_id")
    df = piv.join(tr)
    # chiavi per il join con l'anagrafica: geo_id = "COM:RR:NOME"
    df = df.reset_index()
    parts = df["geo_id"].str.split(":", n=2, expand=True)
    df["reg2"] = parts[1]
    df["nkey"] = parts[2]
    geo = _load_geo_anagrafica()
    out = df.merge(geo, on=["nkey", "reg2"], how="inner")
    out["pop_log"] = np.log1p(out["pop"])
    out["elect_log"] = np.log1p(out["eligible"])
    return out


def train(election_id: str = "elez:2022_politiche") -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    df = build_dataset(election_id)
    feat_num = ["lat", "long", "pop_log", "elect_log", "turnout",
                "avg_income", "pct_pension", "pct_selfemp", "pct_employee"]
    macro_dummies = pd.get_dummies(df["macro"], prefix="area")
    X = pd.concat([df[feat_num], macro_dummies], axis=1).fillna(0.0)
    feat_names = list(X.columns)

    report = {"election_id": election_id, "n_comuni": int(len(df)),
              "features": feat_names, "parties": {}}
    for p in PARTIES:
        if p not in df.columns:
            continue
        y = df[p].fillna(0.0).values
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=0)
        r2 = cross_val_score(rf, X.values, y, cv=cv, scoring="r2", n_jobs=-1)
        rf.fit(X.values, y)
        imp = sorted(zip(feat_names, rf.feature_importances_),
                     key=lambda t: -t[1])
        report["parties"][p] = {
            "r2_cv_mean": float(r2.mean()),
            "r2_cv_std": float(r2.std()),
            "top_drivers": [(n, round(float(v), 3)) for n, v in imp[:5]],
            "model": rf,
        }
    return report
