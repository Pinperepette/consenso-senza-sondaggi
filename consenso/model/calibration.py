"""Auto-calibrazione dei parametri di proiezione sul backtest.

Invece di scegliere a mano lo smorzamento del trend (phi) e il costo del governare
(kappa), li sceglie minimizzando il CRPS out-of-sample sulle elezioni passate, e
salva i valori in ``model_config`` (letti da nowcast e backtest). Onesto: i numeri
non sono "messi a occhio", li decide la validazione.

Sono parametri di SOLA PROIEZIONE: si calibrano riusando i campioni gia' addestrati
(un training per cutoff), quindi e' economico.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from consenso.db.client import get_db

DEFAULTS = {"trend_damping": 0.93, "gov_cost": 0.02}


def param(name: str, default: Optional[float] = None) -> float:
    """Valore calibrato (collection model_config) con fallback a env e default."""
    doc = get_db()["model_config"].find_one({"_id": "calib"}) or {}
    if name in doc:
        return float(doc[name])
    env = {"trend_damping": "CONSENSO_TREND_DAMPING", "gov_cost": "CONSENSO_GOV_COST"}
    if name in env and os.environ.get(env[name]):
        return float(os.environ[env[name]])
    return float(default if default is not None else DEFAULTS.get(name, 0.0))


def calibrate(cutoffs=("2017-12-31", "2021-12-31", "2023-12-31"),
              phis=(0.90, 0.93, 0.96), kappas=(0.0, 0.01, 0.02, 0.03),
              num_warmup: int = 600, num_samples: int = 600) -> dict:
    """Addestra un modello per cutoff, poi cerca (phi, kappa) che minimizzano il
    CRPS medio sulla prima elezione nazionale dopo ogni cutoff. Salva in model_config."""
    from consenso.pipeline.orchestrate import run_model
    from consenso.validation.backtest import (PARTY_ELECTION_TYPES, _actual_shares,
                                              _crps_sample, _months_between)
    from consenso.model.inference import _finest_level, load_samples  # noqa: F401
    from consenso.model.transforms import alr_inv_np
    from consenso.model.fundamentals import cost_of_governing_drift, governing_parties
    from config import CONFIG

    trained = []
    for c in cutoffs:
        try:
            rid = run_model(up_to_date=c, include_regional=False, include_polls=True,
                            trend=True, num_warmup=num_warmup, num_samples=num_samples)["run_id"]
        except Exception:  # noqa: BLE001 - cutoff senza storia sufficiente: salta
            continue
        hp = get_db()["model_runs"].find_one({"_id": rid})["hyperparams"]
        parties, times = hp["parties"], hp["times"]
        ref = parties.index(CONFIG.model.reference_party)
        tix = {t: i for i, t in enumerate(hp["election_types"])}
        s = load_samples(rid)
        last = s["states"][:, -1, :]
        vel = s.get("velocity")
        lastv = vel[:, -1, :] if vel is not None else None
        t0 = get_db()["elections"].find_one(sort=[("date", 1)])["date"]
        nxt = get_db()["elections"].find_one(
            {"type": {"$in": list(PARTY_ELECTION_TYPES)}, "date": {"$gt": c}}, sort=[("date", 1)])
        actual = _actual_shares(nxt["_id"], parties, ref)
        if actual is None:
            continue
        dt = max(_months_between(t0, nxt["date"]) - times[-1], 0.0)
        rng = np.random.default_rng(0)
        innov = rng.normal(0, 1, last.shape) * (s["rw_scale"][:, None] * np.sqrt(max(dt, 1e-6)))
        trained.append({"parties": parties, "ref": ref, "last": last, "lastv": lastv,
                        "beta": s["beta"][:, tix[nxt["type"]], :], "innov": innov,
                        "actual": actual, "dt": dt, "gov": governing_parties(c), "cutoff": c})

    if not trained:                      # storia insufficiente: tieni i default
        return {"trend_damping": DEFAULTS["trend_damping"],
                "gov_cost": DEFAULTS["gov_cost"], "crps": None, "grid": [],
                "note": "storia insufficiente per calibrare, uso i default"}

    def crps_for(phi, kappa):
        vals = []
        for t in trained:
            drift = t["lastv"] * (t["dt"] * phi ** t["dt"]) if t["lastv"] is not None else 0.0
            gd = cost_of_governing_drift(t["parties"], t["ref"], t["gov"], t["dt"],
                                         kappa=kappa, econ_date=t["cutoff"])
            pred = alr_inv_np(t["last"] + drift + gd[None, :] + t["innov"] + t["beta"], t["ref"])
            for k in range(len(t["parties"])):
                vals.append(_crps_sample(pred[:, k], t["actual"][k]))
        return float(np.mean(vals)) if vals else 9e9

    grid = [(phi, kappa, crps_for(phi, kappa)) for phi in phis for kappa in kappas]
    grid.sort(key=lambda r: r[2])
    best_phi, best_kappa, best_crps = grid[0]
    get_db()["model_config"].replace_one(
        {"_id": "calib"},
        {"_id": "calib", "trend_damping": best_phi, "gov_cost": best_kappa,
         "crps": best_crps, "cutoffs": list(cutoffs)}, upsert=True)
    return {"trend_damping": best_phi, "gov_cost": best_kappa, "crps": best_crps,
            "grid": grid[:5]}
