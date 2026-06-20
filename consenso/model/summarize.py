"""Da posterior samples a documenti ``estimations`` interpretabili.

Per ogni tempo, partito e livello territoriale produce: media, mediana, sd,
intervalli credibili, quantili, P(quota > soglia) e P(crescita a 6 mesi)
(doc §2.8). Le stime sono scritte taggate col ``run_id`` (versionamento).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from config import CONFIG
from consenso.db.client import get_db
from consenso.db.schema import ESTIMATIONS, MODEL_RUNS
from consenso.model.inference import load_samples
from consenso.model.transforms import alr_inv_np

NATION_ID = "ISTAT:IT"


def _date_from_months(t0: str, months: float) -> str:
    base = datetime.fromisoformat(t0)
    return (base + timedelta(days=months * 30.4375)).date().isoformat()


def _quantile_summary(samples: np.ndarray) -> dict:
    qs = {q: float(np.quantile(samples, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "sd": float(np.std(samples)),
        "ci80": [float(np.quantile(samples, 0.10)), float(np.quantile(samples, 0.90))],
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "quantiles": {str(k): v for k, v in qs.items()},
        "prob_thresholds": {f">{thr}": float(np.mean(samples > thr))
                            for thr in CONFIG.model.default_thresholds},
    }


def summarize_run(run_id: str, t0_date: Optional[str] = None) -> int:
    run = get_db()[MODEL_RUNS].find_one({"_id": run_id})
    if not run:
        raise KeyError(run_id)
    hp = run["hyperparams"]
    parties: List[str] = hp["parties"]
    regions: List[str] = hp.get("regions", [])
    times: List[float] = hp["times"]
    ref_idx = parties.index(CONFIG.model.reference_party)
    K = len(parties)

    samples = load_samples(run_id)
    states = samples["states"]                      # (S, T, K-1)
    delta = samples.get("delta")                    # (S, G, K-1) o assente
    S, T, _ = states.shape

    # base temporale: se nota la data t0 reale, converte i mesi in date
    t0 = t0_date or _earliest_election_date()
    docs: List[dict] = []

    # quote nazionali per (tempo, campione)
    nat_shares = alr_inv_np(states, ref_idx)        # (S, T, K)

    for ti in range(T):
        as_of = _date_from_months(t0, times[ti]) if t0 else f"t{ti}"
        for k, pid in enumerate(parties):
            sk = nat_shares[:, ti, k]
            doc = _build_doc(run_id, pid, NATION_ID, "nazione", as_of, sk,
                             _growth(nat_shares, times, ti, k))
            docs.append(doc)

    # regionali: stato + offset regionale
    if delta is not None and regions:
        for gi, reg in enumerate(regions):
            reg_eta = states + delta[:, gi, :][:, None, :]      # (S, T, K-1)
            reg_shares = alr_inv_np(reg_eta, ref_idx)
            for ti in range(T):
                as_of = _date_from_months(t0, times[ti]) if t0 else f"t{ti}"
                for k, pid in enumerate(parties):
                    sk = reg_shares[:, ti, k]
                    docs.append(_build_doc(run_id, pid, reg, "regione", as_of, sk,
                                           _growth(reg_shares, times, ti, k)))

    if docs:
        get_db()[ESTIMATIONS].insert_many(docs)
    return len(docs)


def _build_doc(run_id, party_id, geo_id, geo_level, as_of, samples, growth) -> dict:
    s = _quantile_summary(samples)
    return {"run_id": run_id, "party_id": party_id, "geo_id": geo_id,
            "geo_level": geo_level, "as_of": as_of, **s, "prob_growth_6m": growth}


def _growth(shares: np.ndarray, times: List[float], ti: int, k: int) -> Optional[float]:
    """P(quota_t > quota_{t-6mesi}) confrontando con il tempo più vicino a -6m."""
    target = times[ti] - 6.0
    prev = None
    for j in range(ti - 1, -1, -1):
        if times[j] <= target + 1e-6:
            prev = j
            break
    if prev is None:
        return None
    return float(np.mean(shares[:, ti, k] > shares[:, prev, k]))


def _earliest_election_date() -> Optional[str]:
    from consenso.db.schema import ELECTIONS

    e = get_db()[ELECTIONS].find_one(sort=[("date", 1)])
    return e["date"] if e else None
