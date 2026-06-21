"""Motore di scenario: applica assunzioni esplicite (shock per partito, nuovo
partito che pesca dai flussi) ai campioni posterior delle quote e ricalcola la
distribuzione. Le assunzioni arrivano dall'AI (o dall'utente) — qui si fa solo
la matematica, propagando l'incertezza. Output: baseline vs scenario.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from consenso.model.nowcast import projected_shares, summarize_shares


def _sd_from_range(mean: float, lo, hi) -> float:
    if lo is None or hi is None:
        return 0.0
    return max((float(hi) - float(lo)) / 3.92, 0.0)


def apply_scenario(parties: List[str], base: np.ndarray, spec: Dict):
    """Applica lo spec ai campioni (S,K). Restituisce (names, shares)."""
    S = base.shape[0]
    names = list(parties)
    out = base.copy()
    rng = np.random.default_rng(0)

    # --- shock per partito (in punti percentuali) ---
    # Se 'to' e' specificato e il delta e' negativo, e' un TRASFERIMENTO esplicito: il
    # partito perde i punti e la destinazione (un altro partito, 'astensione' o un mix)
    # li riceve. Cosi' la magnitudine e' rispettata e la destinazione e' realistica,
    # invece di spalmare i voti su tutti via rinormalizzazione.
    for d in spec.get("deltas", []):
        pid = d.get("party")
        if pid not in names:
            continue
        i = names.index(pid)
        mean = float(d.get("mean", 0)) / 100.0
        sd = _sd_from_range(d.get("mean", 0), d.get("low"), d.get("high")) / 100.0
        shock = rng.normal(mean, sd, S) if sd > 0 else np.full(S, mean)
        to = d.get("to")
        if to and mean < 0:
            loss = np.minimum(-shock, out[:, i])          # non puo' perdere piu' di quanto ha
            out[:, i] = out[:, i] - loss
            dests = to if isinstance(to, dict) else {to: 1.0}
            tot = sum(v for k, v in dests.items() if k != "astensione") or 1.0
            for dst, frac in dests.items():
                if dst == "astensione" or dst not in names:   # astensione: voti che spariscono (la renorm fa salire gli altri)
                    continue
                out[:, names.index(dst)] += loss * (frac / tot)
        else:
            out[:, i] = np.clip(out[:, i] + shock, 0.0, None)

    # --- nuovo partito che pesca dai donatori ---
    nps = spec.get("new_party")
    if nps and nps.get("name"):
        mean = float(nps.get("share_mean", 0)) / 100.0
        sd = _sd_from_range(mean, nps.get("share_low"), nps.get("share_high")) / 100.0
        s = np.clip(rng.normal(mean, sd, S) if sd > 0 else np.full(S, mean), 0.0, None)
        draws = nps.get("draws_from", {})  # {party: frazione}
        total_frac = sum(v for k, v in draws.items() if k != "astensione") or 1.0
        for donor, frac in draws.items():
            if donor == "astensione" or donor not in names:
                continue
            j = names.index(donor)
            take = np.minimum(s * (frac / total_frac), out[:, j])
            out[:, j] = out[:, j] - take
        names = names + [nps["name"]]
        out = np.column_stack([out, s])

    out = np.clip(out, 0.0, None)
    out = out / out.sum(axis=1, keepdims=True)
    return names, out


def run_scenario(spec: Dict, as_of: Optional[str] = None,
                 run_id: Optional[str] = None) -> Dict:
    parties, base, meta = projected_shares(as_of, run_id)
    if parties is None:
        return {"error": meta.get("error", "nessun run")}
    baseline = summarize_shares(parties, base)
    names, scen = apply_scenario(parties, base, spec)
    scenario = summarize_shares(names, scen)
    return {**meta, "spec": spec, "baseline": baseline, "scenario": scenario}
