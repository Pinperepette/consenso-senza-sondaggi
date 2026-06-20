"""Nowcast: proiezione dello stato latente nazionale a una data (default oggi).

Riutilizzato sia dalla CLI sia dall'endpoint API/GUI. L'incertezza cresce con la
distanza temporale dall'ultima elezione nazionale (random walk).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np

from config import CONFIG
from consenso.db.client import get_db
from consenso.model.inference import _months_between, load_samples
from consenso.model.transforms import alr_inv_np


def projected_shares(as_of: Optional[str] = None, run_id: Optional[str] = None):
    """Campioni delle quote nazionali proiettate alla data (S, K).

    Restituisce (parties, shares, meta). meta ha as_of, projection_months,
    run_id, last_election. È la base sia per il nowcast sia per gli scenari.
    """
    as_of = as_of or date.today().isoformat()
    run = (get_db()["model_runs"].find_one({"_id": run_id}) if run_id
           else get_db()["model_runs"].find_one(sort=[("created_at", -1)]))
    if not run:
        return None, None, {"error": "nessun run disponibile"}
    hp = run["hyperparams"]
    parties = hp["parties"]
    times = hp["times"]
    ref_idx = parties.index(CONFIG.model.reference_party)
    s = load_samples(run["_id"])
    states, rw = s["states"], s["rw_scale"]
    t0 = get_db()["elections"].find_one(sort=[("date", 1)])["date"]
    if hp.get("include_polls"):
        # i sondaggi recenti tengono aggiornato lo stato: ancora all'ultimo punto
        anchor_idx = len(times) - 1
    else:
        # senza sondaggi: ancora all'ultima elezione NAZIONALE (le regionali distorcono)
        last_nat = get_db()["elections"].find_one(
            {"type": {"$in": ["politiche", "europee"]}}, sort=[("date", -1)])
        anchor_m = _months_between(t0, last_nat["date"]) if last_nat else max(times)
        anchor_idx = int(np.argmin([abs(t - anchor_m) for t in times]))
    dt = max(_months_between(t0, as_of) - times[anchor_idx], 0.0)
    rng = np.random.default_rng(0)
    proj = states[:, anchor_idx, :] + rng.normal(size=states[:, anchor_idx, :].shape) * (
        rw[:, None] * np.sqrt(max(dt, 1e-6)))
    shares = alr_inv_np(proj, ref_idx)
    last_real = get_db()["elections"].find_one(sort=[("date", -1)])
    meta = {"as_of": as_of, "run_id": run["_id"], "projection_months": round(dt, 1),
            "last_election": ({"type": last_real["type"], "date": last_real["date"]}
                              if last_real else None)}
    return parties, shares, meta


def summarize_shares(parties, shares) -> list:
    """Da campioni (S,K) a sommari per partito (media, IC95, P(>soglia))."""
    thr = CONFIG.model.default_thresholds
    out = []
    for k, p in enumerate(parties):
        col = shares[:, k]
        out.append({"party_id": p, "name": p.replace("party:", ""),
                    "mean": float(col.mean()),
                    "ci95": [float(np.quantile(col, 0.025)), float(np.quantile(col, 0.975))],
                    "prob_thresholds": {f">{t}": float(np.mean(col > t)) for t in thr}})
    out.sort(key=lambda r: -r["mean"])
    return out


def nowcast(as_of: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    parties, shares, meta = projected_shares(as_of, run_id)
    if parties is None:
        return {"error": meta.get("error", "nessun run")}
    out_parties = summarize_shares(parties, shares)
    leader_idx = int(np.argmax(shares.mean(0)))
    p_first = float(np.mean(np.argmax(shares, axis=1) == leader_idx))
    return {**meta,
            "leader": parties[leader_idx].replace("party:", ""),
            "p_leader_first": p_first,
            "parties": out_parties}
