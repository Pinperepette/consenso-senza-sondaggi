"""Fondamentali per la previsione di lungo periodo.

Il momentum (trend) aiuta nel breve ma non a 5 anni; sul lungo contano i fattori
STRUTTURALI. Il primo, il piu' robusto in scienza politica: il **costo del
governare** — chi e' al governo tende a perdere consenso col tempo (logoramento).

Qui la timeline (reale, documentata) dei partiti al governo e un drift di
proiezione che spinge GIU' gli incumbent sull'orizzonte di previsione. Calibrato
sul backtest, configurabile via env. Layer separato e dichiarato: NON tocca la
ricostruzione del passato, agisce solo sulla proiezione in avanti.
"""
from __future__ import annotations

import os
from typing import Dict, List, Set

import numpy as np

from consenso.db.client import get_db

# Partiti al governo per periodo [inizio, fine) (date ISO). Mappa sui 6 partiti.
# Fonte: composizione dei governi italiani.
GOV_TIMELINE = {
    "party:FI":   [("2008-05-08", "2011-11-16"), ("2021-02-13", "2099-01-01")],
    "party:LEGA": [("2008-05-08", "2011-11-16"), ("2018-06-01", "2019-09-05"),
                   ("2021-02-13", "2099-01-01")],
    "party:PD":   [("2013-04-28", "2018-06-01"), ("2019-09-05", "2022-10-22")],
    "party:M5S":  [("2018-06-01", "2022-10-22")],
    "party:FDI":  [("2022-10-22", "2099-01-01")],
    # AVS: sostanzialmente sempre all'opposizione
}


# "Misery index" = inflazione media annua + tasso di disoccupazione (%), Italia.
# Fonte: ISTAT (serie annuali). Cattura il classico voto economico: piu' e' alto,
# piu' l'incumbent paga. Valori storici documentati (aggiornabili da loader ISTAT).
ECON_MISERY = {
    2008: 10.0, 2009: 8.5, 2010: 9.9, 2011: 11.2, 2012: 13.7, 2013: 13.3,
    2014: 12.9, 2015: 12.0, 2016: 11.6, 2017: 12.5, 2018: 11.8, 2019: 10.5,
    2020: 9.2, 2021: 11.4, 2022: 16.2, 2023: 13.3, 2024: 7.6, 2025: 7.5, 2026: 8.0,
}
MISERY_BASELINE = 11.5     # media storica ~ condizioni "normali"


def _misery_table() -> Dict[int, float]:
    """Misery index per anno: dati scaricati (collection economics) sovrascritti
    sui valori statici di fallback."""
    table = dict(ECON_MISERY)
    try:
        for d in get_db()["economics"].find({}, {"misery": 1}):
            table[int(d["_id"])] = float(d["misery"])
    except Exception:  # noqa: BLE001
        pass
    return table


def economic_stress(date: str) -> float:
    """Moltiplicatore [0.5, 1.5] del costo del governare: >1 economia peggiore
    della norma (incumbent erosi di piu'), <1 economia migliore."""
    year = int(date[:4])
    table = _misery_table()
    misery = table.get(year, table.get(max(table), MISERY_BASELINE))
    return float(min(1.5, max(0.5, misery / MISERY_BASELINE)))


def governing_parties(date: str) -> Set[str]:
    """Partiti al governo a una certa data."""
    return {p for p, periods in GOV_TIMELINE.items()
            if any(a <= date < b for a, b in periods)}


def cost_of_governing_drift(parties: List[str], ref_idx: int, gov: Set[str],
                            dt_months: float, kappa: float | None = None,
                            econ_date: str | None = None) -> np.ndarray:
    """Drift (in log-ratio, lunghezza K-1) che logora gli incumbent sull'orizzonte.

    Ogni partito al governo viene spinto giu' di ``kappa * dt`` (per mese), gli
    altri salgono di riflesso via softmax. Se ``econ_date`` e' dato, ``kappa`` e'
    modulato dall'economia (misery index): peggio va, piu' l'incumbent paga.
    """
    if kappa is None:
        from consenso.model.calibration import param
        kappa = param("gov_cost")
    if econ_date:
        kappa = kappa * economic_stress(econ_date)
    K = len(parties)
    drift_full = np.zeros(K)
    # tetto sull'orizzonte: il logoramento non cresce all'infinito (un governo dura
    # al piu' una legislatura), si satura intorno ai ~3 anni.
    h = min(dt_months, 36.0)
    if kappa and h > 0:
        for k, pid in enumerate(parties):
            if pid in gov:
                drift_full[k] = -kappa * h
    return np.delete(drift_full, ref_idx)     # (K-1,)
