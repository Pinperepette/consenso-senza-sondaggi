"""Validazione temporale out-of-sample (doc §7).

Schema: si addestra il modello fino a una data T, si proietta lo stato latente
in avanti fino alle elezioni successive e si confronta la previsione con il
risultato reale. Metriche: MAE, CRPS (probabilistica) e coverage dell'IC95%
(calibrazione).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import os

import numpy as np

from config import CONFIG
from consenso.db.client import get_db
from consenso.db.schema import BACKTESTS, ELECTIONS
from consenso.model.inference import (PARTY_ELECTION_TYPES, _finest_level,
                                      _months_between, _national_shares,
                                      load_samples)
from consenso.model.fundamentals import (cost_of_governing_drift,
                                         governing_parties)
from consenso.model.transforms import alr_inv_np
from consenso.pipeline.orchestrate import run_model


def _crps_sample(forecast: np.ndarray, obs: float) -> float:
    """CRPS stimato da campioni: E|X-y| - 0.5 E|X-X'|."""
    term1 = np.mean(np.abs(forecast - obs))
    term2 = 0.5 * np.mean(np.abs(forecast[:, None] - forecast[None, :]))
    return float(term1 - term2)


def _actual_shares(election_id: str, parties: List[str], ref_idx: int) -> Optional[np.ndarray]:
    lvl = _finest_level(election_id)
    if not lvl:
        return None
    votes, total = _national_shares(election_id, lvl)
    if total <= 0:
        return None
    pidx = {p: i for i, p in enumerate(parties)}
    shares = np.zeros(len(parties))
    other = 0
    for pid, v in votes.items():
        if pid in pidx:
            shares[pidx[pid]] += v
        else:
            other += v
    shares[ref_idx] += other
    return shares / total


def run_backtest(train_until: str, num_warmup: Optional[int] = None,
                 num_samples: Optional[int] = None, include_polls: bool = False,
                 trend: bool = False) -> Dict:
    # 1) addestra solo sul passato
    res = run_model(up_to_date=train_until, include_regional=False,
                    include_polls=include_polls, trend=trend,
                    num_warmup=num_warmup, num_samples=num_samples)
    run_id = res["run_id"]
    run = get_db()["model_runs"].find_one({"_id": run_id})
    hp = run["hyperparams"]
    parties: List[str] = hp["parties"]
    times: List[float] = hp["times"]
    ref_idx = parties.index(CONFIG.model.reference_party)
    type_idx = {t: i for i, t in enumerate(hp["election_types"])}

    samples = load_samples(run_id)
    states = samples["states"]                  # (S, T, K-1)
    rw = samples["rw_scale"]                     # (S,)
    beta = samples["beta"]                       # (S, n_types, K-1)
    last_state = states[:, -1, :]                # (S, K-1)
    vel = samples.get("velocity")                # (S, T, K-1) se modello trend
    last_vel = vel[:, -1, :] if vel is not None else None

    t0 = get_db()[ELECTIONS].find_one(sort=[("date", 1)])["date"]
    last_train_months = times[-1]

    # 2) elezioni target dopo T
    targets = list(get_db()[ELECTIONS].find(
        {"type": {"$in": list(PARTY_ELECTION_TYPES)}, "date": {"$gt": train_until}}
    ).sort("date", 1))

    per_target = []
    all_abs_err, all_crps, all_cover = [], [], []
    rng = np.random.default_rng(0)

    for e in targets:
        actual = _actual_shares(e["_id"], parties, ref_idx)
        if actual is None:
            continue
        tgt_months = _months_between(t0, e["date"])
        dt = max(tgt_months - last_train_months, 0.0)
        # proietta lo stato avanti + bias di tipo
        innov = rng.normal(0, 1, last_state.shape) * (rw[:, None] * np.sqrt(max(dt, 1e-6)))
        # trend smorzato: drift = vel * dt * phi^dt (svanisce sull'orizzonte lungo)
        from consenso.model.calibration import param
        phi = param("trend_damping")
        drift = (last_vel * (dt * (phi ** dt))) if last_vel is not None else 0.0
        gd = cost_of_governing_drift(parties, ref_idx, governing_parties(train_until),
                                     dt, econ_date=train_until)
        proj = last_state + drift + gd[None, :] + innov + beta[:, type_idx[e["type"]], :]
        pred_shares = alr_inv_np(proj, ref_idx)          # (S, K)

        per_party = []
        for k, pid in enumerate(parties):
            fc = pred_shares[:, k]
            a = float(actual[k])
            lo, hi = np.quantile(fc, 0.025), np.quantile(fc, 0.975)
            in_ci = bool(lo <= a <= hi)
            abs_err = abs(float(fc.mean()) - a)
            crps = _crps_sample(fc, a)
            per_party.append({"party_id": pid, "pred_mean": float(fc.mean()),
                              "ci95": [float(lo), float(hi)],
                              "actual": a, "in_ci95": in_ci,
                              "abs_err": abs_err, "crps": crps})
            all_abs_err.append(abs_err); all_crps.append(crps); all_cover.append(in_ci)
        per_target.append({"target": e["_id"], "type": e["type"], "date": e["date"],
                           "per_party": per_party})

    metrics = {
        "mae": float(np.mean(all_abs_err)) if all_abs_err else None,
        "crps": float(np.mean(all_crps)) if all_crps else None,
        "coverage95": float(np.mean(all_cover)) if all_cover else None,
        "n_party_obs": len(all_abs_err),
    }
    doc = {"scheme": f"train_until_{train_until}", "train_run": run_id,
           "metrics": metrics, "targets": per_target,
           "created_at": datetime.utcnow()}
    get_db()[BACKTESTS].insert_one(dict(doc))
    return {"scheme": doc["scheme"], "metrics": metrics,
            "n_targets": len(per_target), "train_run": run_id}
