"""Trasformazioni fra il simplesso delle quote e lo spazio log-ratio (ALR).

Lo stato latente del modello vive in R^{K-1} (log-ratio additivo rispetto a un
partito di riferimento). Questo garantisce che le quote ricostruite siano in
(0,1) e sommino a 1, senza vincoli espliciti nel campionatore.

Funzioni disponibili sia in numpy (per pre/post-processing) sia in jax.numpy
(per l'interno del modello NumPyro), distinte dal suffisso ``_np`` / ``_jax``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-9


def alr_np(p: np.ndarray, ref_idx: int) -> np.ndarray:
    """Quote (asse finale lungo K) -> log-ratio (lunghezza K-1).

    p può essere (..., K). Restituisce (..., K-1) escludendo il riferimento.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    logp = np.log(p)
    ref = logp[..., ref_idx:ref_idx + 1]
    eta = logp - ref
    return np.delete(eta, ref_idx, axis=-1)


def alr_inv_np(eta: np.ndarray, ref_idx: int) -> np.ndarray:
    """Log-ratio (..., K-1) -> quote (..., K) via softmax con riferimento a 0."""
    eta = np.asarray(eta, dtype=float)
    # reinserisce lo 0 del riferimento
    full = np.insert(eta, ref_idx, 0.0, axis=-1)
    m = np.max(full, axis=-1, keepdims=True)
    ex = np.exp(full - m)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def alr_inv_jax(eta, ref_idx: int, K: int):
    """Versione jax di ``alr_inv_np`` (eta ha forma (..., K-1))."""
    import jax.numpy as jnp

    lead = eta[..., :ref_idx]
    trail = eta[..., ref_idx:]
    zeros = jnp.zeros(eta.shape[:-1] + (1,))
    full = jnp.concatenate([lead, zeros, trail], axis=-1)
    m = jnp.max(full, axis=-1, keepdims=True)
    ex = jnp.exp(full - m)
    return ex / jnp.sum(ex, axis=-1, keepdims=True)


def shares_to_eta(shares_by_party: Sequence[float], ref_idx: int) -> np.ndarray:
    return alr_np(np.asarray(shares_by_party, dtype=float), ref_idx)
