"""Nowcast: proiezione dello stato latente nazionale a una data (default oggi).

Riutilizzato sia dalla CLI sia dall'endpoint API/GUI. L'incertezza cresce con la
distanza temporale dall'ultima elezione nazionale (random walk).
"""
from __future__ import annotations

import os
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
    last_poll_date = None
    if hp.get("include_polls"):
        # i sondaggi recenti tengono aggiornato lo stato: ancora all'ultimo punto
        anchor_idx = len(times) - 1
        lastp = get_db()["polls"].find_one(sort=[("date", -1)])
        last_poll_date = lastp["date"] if lastp else None
    else:
        # senza sondaggi: ancora all'ultima elezione NAZIONALE (le regionali distorcono)
        last_nat = get_db()["elections"].find_one(
            {"type": {"$in": ["politiche", "europee"]}}, sort=[("date", -1)])
        anchor_m = _months_between(t0, last_nat["date"]) if last_nat else max(times)
        anchor_idx = int(np.argmin([abs(t - anchor_m) for t in times]))
    # tempo REALE dell'ancora: coi sondaggi e' l'ultimo sondaggio (non il centro del
    # trimestre), cosi' "oggi" non e' una proiezione fittizia di ~1 mese.
    anchor_m = (_months_between(t0, last_poll_date) if last_poll_date
                else times[anchor_idx])
    dt = max(_months_between(t0, as_of) - anchor_m, 0.0)
    rng = np.random.default_rng(0)
    anchor = states[:, anchor_idx, :]
    # trend smorzato: il momentum recente conta nel breve e svanisce sul lungo
    # (dt*phi^dt). Se il modello non e' un trend (niente velocity), e' random walk puro.
    vel = s.get("velocity")
    if vel is not None:
        from consenso.model.calibration import param
        phi = param("trend_damping")
        anchor = anchor + vel[:, anchor_idx, :] * (dt * (phi ** dt))
    # fondamentale: costo del governare (gli incumbent si logorano sull'orizzonte)
    from consenso.model.fundamentals import cost_of_governing_drift, governing_parties
    gov = governing_parties(as_of)
    anchor = anchor + cost_of_governing_drift(parties, ref_idx, gov, dt, econ_date=as_of)[None, :]
    proj = anchor + rng.normal(size=anchor.shape) * (rw[:, None] * np.sqrt(max(dt, 1e-6)))
    shares = alr_inv_np(proj, ref_idx)
    # riferimento onesto: l'ultima elezione NAZIONALE (le regionali non lo sono)
    last_nat = get_db()["elections"].find_one(
        {"type": {"$in": ["politiche", "europee"]}}, sort=[("date", -1)])
    polls_on = bool(hp.get("include_polls"))
    anchor_date = None
    if polls_on:
        lastp = get_db()["polls"].find_one(sort=[("date", -1)])
        anchor_date = lastp["date"] if lastp else None
    meta = {"as_of": as_of, "run_id": run["_id"], "projection_months": round(dt, 1),
            "polls_anchored": polls_on, "anchor_date": anchor_date,
            "last_election": ({"type": last_nat["type"], "date": last_nat["date"]}
                              if last_nat else None)}
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
