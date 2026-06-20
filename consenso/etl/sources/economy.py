"""Indicatori economici per il 'voto economico' (misery index = inflazione +
disoccupazione). Fonte: World Bank (API JSON pubblica, dati Italia), che
ridistribuisce le serie ufficiali. Scaricati e salvati nella collection
``economics``; ``fundamentals.economic_stress`` li legge da li' con fallback ai
valori statici.
"""
from __future__ import annotations

import json
from typing import Dict

from consenso.db.client import get_db
from consenso.etl.base import http_get

WB = ("https://api.worldbank.org/v2/country/IT/indicator/{ind}"
      "?format=json&date=2008:2030&per_page=200")
INFLATION = "FP.CPI.TOTL.ZG"        # inflazione % annua
UNEMPLOYMENT = "SL.UEM.TOTL.ZS"     # disoccupazione % forza lavoro


def _series(ind: str) -> Dict[int, float]:
    raw = json.loads(http_get(WB.format(ind=ind)))
    out: Dict[int, float] = {}
    if isinstance(raw, list) and len(raw) > 1 and raw[1]:
        for row in raw[1]:
            if row.get("value") is not None:
                out[int(row["date"])] = float(row["value"])
    return out


def fetch_misery() -> int:
    """Scarica inflazione+disoccupazione e salva il misery index per anno."""
    infl, unemp = _series(INFLATION), _series(UNEMPLOYMENT)
    db = get_db()
    docs = []
    for year in sorted(set(infl) & set(unemp)):
        docs.append({"_id": year, "inflation": round(infl[year], 2),
                     "unemployment": round(unemp[year], 2),
                     "misery": round(infl[year] + unemp[year], 2),
                     "source": "worldbank"})
    if docs:
        db["economics"].delete_many({})
        db["economics"].insert_many(docs)
    return len(docs)
