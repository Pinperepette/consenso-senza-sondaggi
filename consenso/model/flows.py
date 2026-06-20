"""Stima dei flussi elettorali via ecological inference bayesiana (doc §5).

Dati: per ogni area (idealmente sezione, altrimenti comune) le quote dell'elezione
A e i conteggi dell'elezione B, incluse le categorie di astensione. Stimiamo una
matrice di transizione le cui righe (destinazioni di chi proveniva da i) hanno
prior di Dirichlet centrato su una matrice nazionale ``Pbar`` con massa sulla
diagonale (fedeltà). Vincolo contabile: q_a = P_a^T x_a; likelihood multinomiale
sui conteggi di B.

Limite dichiarato: ecological fallacy. Stimiamo flussi *compatibili* con gli
aggregati e i prior, non comportamenti individuali certi.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

ABSTENTION = "astensione"


@dataclass
class FlowData:
    categories_from: List[str]    # C_from (parties A + astensione)
    categories_to: List[str]      # C_to   (parties B + astensione)
    x: np.ndarray                 # (A, C_from) quote elezione A (sommano a 1)
    counts_b: np.ndarray          # (A, C_to) conteggi elezione B
    N: np.ndarray                 # (A,) totale aventi diritto per area
    concentration: float = 20.0
    hierarchical: bool = True


def diagonal_prior(categories_from: List[str], categories_to: List[str],
                   diag: float = 5.0, off: float = 1.0) -> np.ndarray:
    """Prior di Dirichlet per riga, con massa extra sulla fedeltà (i->i)."""
    Cf, Ct = len(categories_from), len(categories_to)
    prior = np.full((Cf, Ct), off, dtype=float)
    to_index = {c: j for j, c in enumerate(categories_to)}
    for i, c in enumerate(categories_from):
        if c in to_index:
            prior[i, to_index[c]] = diag
    return prior


def flow_model(data: FlowData):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    Cf, Ct = len(data.categories_from), len(data.categories_to)
    A = data.x.shape[0]
    prior = jnp.asarray(diagonal_prior(data.categories_from, data.categories_to))
    x = jnp.asarray(data.x)
    counts = jnp.asarray(data.counts_b)
    N = jnp.asarray(data.N)

    # matrice nazionale: una Dirichlet per riga
    with numpyro.plate("from_rows", Cf):
        Pbar = numpyro.sample("Pbar", dist.Dirichlet(prior))     # (Cf, Ct)

    if data.hierarchical and A > 1:
        conc = data.concentration
        # P per area attorno a Pbar
        with numpyro.plate("areas", A, dim=-2):
            with numpyro.plate("from_rows_a", Cf, dim=-1):
                P = numpyro.sample("P", dist.Dirichlet(conc * Pbar[None, :, :]))  # (A,Cf,Ct)
        q = jnp.einsum("ai,aij->aj", x, P)          # (A, Ct)
    else:
        q = jnp.einsum("ai,ij->aj", x, Pbar)        # (A, Ct)

    q = jnp.clip(q, 1e-9, 1.0)
    q = q / jnp.sum(q, axis=-1, keepdims=True)
    with numpyro.plate("obs_areas", A):
        numpyro.sample("counts", dist.Multinomial(total_count=N.astype(int), probs=q),
                       obs=counts)


def summarize_flows(posterior: Dict[str, np.ndarray], data: FlowData) -> dict:
    """Da campioni posterior a sommari interpretabili."""
    Pbar = np.asarray(posterior["Pbar"])           # (S, Cf, Ct)
    mean = Pbar.mean(axis=0)
    sd = Pbar.std(axis=0)
    cf, ct = data.categories_from, data.categories_to
    loyalty = {}
    to_index = {c: j for j, c in enumerate(ct)}
    for i, c in enumerate(cf):
        if c in to_index:
            loyalty[c] = float(mean[i, to_index[c]])
    return {
        "parties_from": cf,
        "parties_to": ct,
        "transfer_matrix_mean": mean.tolist(),
        "transfer_matrix_sd": sd.tolist(),
        "loyalty": loyalty,
        "method": "bayesian_ecological_inference",
    }
