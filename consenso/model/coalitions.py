"""Coalizioni e proiezione dei seggi (Camera), con incertezza propagata dai
campioni posteriori.

ONESTA': le alleanze sono fluide (il "campo largo" e' un'ipotesi, non un fatto) e
la proiezione seggi e' una STIMA SEMPLIFICATA del Rosatellum (niente collegi
uninominali reali): va letta come ordine di grandezza, non come conteggio.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from consenso.model.nowcast import projected_shares

# Coalizioni di riferimento (modificabili). FN/Vannacci tenuto separato: non e'
# nella maggioranza. Il "Campo largo" e' un'ipotesi di alleanza.
COALITIONS: Dict[str, List[str]] = {
    "Centrodestra": ["party:FDI", "party:LEGA", "party:FI"],
    "Campo largo": ["party:PD", "party:M5S", "party:AVS"],
}

# Camera: 400 seggi. Ripartizione semplificata: quota proporzionale + uninominali.
N_SEATS = 400
N_MAJ = 147               # collegi uninominali (FPTP, vinti dalle coalizioni)
N_PROP = N_SEATS - N_MAJ  # quota proporzionale (soglia 3% di partito)
PROP_THRESHOLD = 0.03
MAJ_SKEW = 1.6            # premio al vincitore nei collegi (share^skew normalizzato)


def _summ(samples: np.ndarray) -> dict:
    return {"mean": float(samples.mean()),
            "ci95": [float(np.quantile(samples, 0.025)),
                     float(np.quantile(samples, 0.975))]}


def coalition_shares(as_of: Optional[str] = None,
                     coalitions: Dict[str, List[str]] = None) -> dict:
    """Quota di ogni coalizione (somma dei membri) con IC95 propagato."""
    coalitions = coalitions or COALITIONS
    parties, shares, meta = projected_shares(as_of)
    if parties is None:
        return {"error": meta.get("error", "nessun run")}
    pidx = {p: i for i, p in enumerate(parties)}
    out = []
    for name, members in coalitions.items():
        idx = [pidx[m] for m in members if m in pidx]
        if not idx:
            continue
        s = shares[:, idx].sum(axis=1)
        out.append({"name": name, "members": [m.replace("party:", "") for m in members
                                              if m in pidx],
                    **_summ(s), "p_first": None})
    # P(coalizione prima) fra quelle definite
    if out:
        mat = np.vstack([shares[:, [pidx[m] for m in coalitions[o["name"]]
                                    if m in pidx]].sum(axis=1) for o in out])
        first = np.argmax(mat, axis=0)
        for i, o in enumerate(out):
            o["p_first"] = float(np.mean(first == i))
    out.sort(key=lambda r: -r["mean"])
    return {"as_of": meta["as_of"], "coalitions": out}


def _dhondt(shares: np.ndarray, n_seats: int) -> np.ndarray:
    """Ripartizione D'Hondt di n_seats fra partiti con quote 'shares' (gia' filtrate)."""
    seats = np.zeros(len(shares), dtype=int)
    if shares.sum() <= 0:
        return seats
    for _ in range(n_seats):
        q = shares / (seats + 1)
        seats[int(np.argmax(q))] += 1
    return seats


def _seats_one_draw(share_vec: np.ndarray, parties: List[str],
                    coalitions: Dict[str, List[str]], pidx: Dict[str, int]) -> Dict[str, int]:
    """Seggi per partito su UN campione: proporzionale (soglia) + uninominali (coalizioni)."""
    # 1) proporzionale: solo partiti >= soglia
    elig = share_vec * (share_vec >= PROP_THRESHOLD)
    prop = _dhondt(elig, N_PROP)
    seats = {parties[i]: int(prop[i]) for i in range(len(parties))}
    # 2) uninominali: alle coalizioni con premio al vincitore, poi ai partiti interni
    cnames = list(coalitions)
    cshare = np.array([sum(share_vec[pidx[m]] for m in coalitions[c] if m in pidx)
                       for c in cnames])
    # i partiti non in coalizione concorrono come "coalizione" di se stessi
    in_coal = {m for c in coalitions for m in coalitions[c]}
    solos = [p for p in parties if p not in in_coal and p != "party:ALTRI"]
    sshare = np.array([share_vec[pidx[p]] for p in solos])
    allnames = cnames + solos
    allshare = np.concatenate([cshare, sshare]) if len(sshare) else cshare
    skewed = np.power(np.clip(allshare, 0, None), MAJ_SKEW)
    maj = _dhondt(skewed, N_MAJ) if skewed.sum() > 0 else np.zeros(len(allnames), int)
    # distribuisci i seggi uninominali di una coalizione fra i suoi partiti
    for j, name in enumerate(allnames):
        members = coalitions.get(name, [name])
        m_in = [m for m in members if m in pidx]
        msh = np.array([share_vec[pidx[m]] for m in m_in])
        if msh.sum() <= 0:
            continue
        alloc = _dhondt(msh, int(maj[j]))
        for k, m in enumerate(m_in):
            seats[m] += int(alloc[k])
    return seats


def seat_projection(as_of: Optional[str] = None,
                    coalitions: Dict[str, List[str]] = None,
                    max_draws: int = 500) -> dict:
    """Proiezione SEMPLIFICATA dei seggi alla Camera (400), con IC95 propagato."""
    coalitions = coalitions or COALITIONS
    parties, shares, meta = projected_shares(as_of)
    if parties is None:
        return {"error": meta.get("error", "nessun run")}
    pidx = {p: i for i, p in enumerate(parties)}
    draws = shares[np.linspace(0, len(shares) - 1, min(max_draws, len(shares))).astype(int)]
    per_party = {p: [] for p in parties}
    coal_tot = {c: [] for c in coalitions}
    for vec in draws:
        s = _seats_one_draw(vec, parties, coalitions, pidx)
        for p in parties:
            per_party[p].append(s[p])
        for c, members in coalitions.items():
            coal_tot[c].append(sum(s[m] for m in members if m in pidx))
    def summ_int(a):
        a = np.array(a)
        return {"mean": float(a.mean()),
                "ci95": [int(np.quantile(a, 0.025)), int(np.quantile(a, 0.975))]}
    parties_out = sorted(
        [{"party": p.replace("party:", ""), **summ_int(v)}
         for p, v in per_party.items() if p != "party:ALTRI"],
        key=lambda r: -r["mean"])
    coals_out = []
    for c, v in coal_tot.items():
        a = np.array(v)
        coals_out.append({"name": c, **summ_int(v),
                          "p_majority": float(np.mean(a >= N_SEATS // 2 + 1))})
    coals_out.sort(key=lambda r: -r["mean"])
    return {"as_of": meta["as_of"], "n_seats": N_SEATS, "majority": N_SEATS // 2 + 1,
            "parties": parties_out, "coalitions": coals_out,
            "note": "stima semplificata del Rosatellum (no collegi reali), ordine di grandezza"}
