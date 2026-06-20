"""Assemblaggio dati e stima dei flussi elettorali fra due elezioni consecutive.

Allinea le aree presenti in entrambe le elezioni, costruisce le quote dell'elezione
A (incluse l'astensione) e i conteggi dell'elezione B, esegue l'ecological
inference (:mod:`consenso.model.flows`) e salva la matrice in ``flow_models``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from consenso.db.client import get_db
from consenso.db.schema import FLOW_MODELS, PARTY_RESULTS, TURNOUT
from consenso.model.flows import (ABSTENTION, FlowData, flow_model,
                                  summarize_flows)

OTHER = "party:ALTRI"


def _votes_by_area(election_id: str, level: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in get_db()[PARTY_RESULTS].find(
            {"election_id": election_id, "geo_level": level},
            {"geo_id": 1, "party_id": 1, "votes": 1}):
        pid = r.get("party_id") or OTHER
        out.setdefault(r["geo_id"], {})
        out[r["geo_id"]][pid] = out[r["geo_id"]].get(pid, 0) + r["votes"]
    return out


def _eligible_by_area(election_id: str) -> Dict[str, int]:
    return {t["geo_id"]: t.get("eligible", 0)
            for t in get_db()[TURNOUT].find({"election_id": election_id},
                                            {"geo_id": 1, "eligible": 1})}


def _top_parties(votes_by_area: Dict[str, Dict[str, int]], top: int) -> List[str]:
    tot: Dict[str, int] = {}
    for area in votes_by_area.values():
        for pid, v in area.items():
            tot[pid] = tot.get(pid, 0) + v
    ranked = [p for p in sorted(tot, key=tot.get, reverse=True) if p != OTHER][:top]
    return ranked


def assemble_flow_data(from_election: str, to_election: str, level: str = "comune",
                       top_parties: int = 6, concentration: float = 20.0,
                       hierarchical: bool = True) -> FlowData:
    va = _votes_by_area(from_election, level)
    vb = _votes_by_area(to_election, level)
    elig_a = _eligible_by_area(from_election)
    elig_b = _eligible_by_area(to_election)

    areas = [g for g in va.keys() & vb.keys() if elig_a.get(g) and elig_b.get(g)]
    if not areas:
        raise ValueError("nessuna area in comune con aventi diritto fra le due elezioni")

    cats_from = _top_parties(va, top_parties) + [OTHER, ABSTENTION]
    cats_to = _top_parties(vb, top_parties) + [OTHER, ABSTENTION]
    fi = {c: i for i, c in enumerate(cats_from)}
    ti = {c: i for i, c in enumerate(cats_to)}

    X = np.zeros((len(areas), len(cats_from)))
    CB = np.zeros((len(areas), len(cats_to)))
    N = np.zeros(len(areas))

    for a, g in enumerate(areas):
        ea = elig_a[g]
        N[a] = elig_b[g]
        # elezione A: quote (incl. astensione)
        voters_a = 0
        for pid, v in va[g].items():
            voters_a += v
            X[a, fi.get(pid, fi[OTHER])] += v
        X[a, fi[ABSTENTION]] = max(ea - voters_a, 0)
        s = X[a].sum()
        if s > 0:
            X[a] /= s
        # elezione B: conteggi (incl. astensione)
        voters_b = 0
        for pid, v in vb[g].items():
            voters_b += v
            CB[a, ti.get(pid, ti[OTHER])] += v
        CB[a, ti[ABSTENTION]] = max(elig_b[g] - voters_b, 0)
        # rendi i conteggi coerenti col totale N (arrotondamenti)
        diff = int(round(N[a])) - int(CB[a].sum())
        CB[a, ti[ABSTENTION]] = max(CB[a, ti[ABSTENTION]] + diff, 0)

    return FlowData(categories_from=cats_from, categories_to=cats_to,
                    x=X, counts_b=CB.astype(int), N=N.astype(int),
                    concentration=concentration, hierarchical=hierarchical)


def run_flow_model(from_election: str, to_election: str, level: str = "comune",
                   geo_scope: str = "ISTAT:IT", run_id: Optional[str] = None,
                   num_warmup: int = 500, num_samples: int = 500,
                   seed: int = 0, **kw) -> dict:
    import jax
    from numpyro.infer import MCMC, NUTS

    data = assemble_flow_data(from_election, to_election, level=level, **kw)
    mcmc = MCMC(NUTS(flow_model), num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data)
    posterior = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    summary = summarize_flows(posterior, data)

    doc = {"run_id": run_id, "from_election": from_election, "to_election": to_election,
           "geo_scope": geo_scope, "level": level, "n_areas": int(data.x.shape[0]),
           **summary}
    get_db()[FLOW_MODELS].insert_one(dict(doc))
    return summary
