"""State Space Model bayesiano gerarchico (NumPyro).

Concetto (doc §2): ogni elezione è una misura distorta di uno stato latente
continuo ``η_t`` (consenso nazionale in log-ratio). Lo stato evolve come random
walk in tempo continuo; l'osservazione vede lo stato attraverso:
  - un bias di tipo-elezione ``β`` (politiche = ancora, β=0);
  - un offset regionale gerarchico ``δ`` (partial pooling);
  - un'elasticità all'affluenza ``λ``;
  - un rumore di misura ``R`` inversamente proporzionale alla rappresentatività
    dell'elezione (passato in ``obs_sd``).

L'oggetto :class:`ModelData` è il contratto fra l'assemblaggio dati e il modello.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ModelData:
    parties: List[str]            # lunghezza K (incl. riferimento)
    ref_idx: int
    election_types: List[str]     # lunghezza n_types
    anchor_type_idx: int
    regions: List[str]            # lunghezza G (geo_idx region r -> r+1)
    times: np.ndarray             # (T,) tempi unici ordinati, in mesi da t0
    # osservazioni (N righe):
    obs_eta: np.ndarray           # (N, K-1) log-ratio osservati
    obs_mask: np.ndarray          # (N, K-1) bool: partito presente nell'osservazione
    obs_time_idx: np.ndarray      # (N,) indice in times
    obs_type_idx: np.ndarray      # (N,) indice in election_types
    obs_geo_idx: np.ndarray       # (N,) 0=nazionale, r+1=regione r
    obs_turnout_dev: np.ndarray   # (N,) scarto affluenza
    obs_sd: np.ndarray            # (N,) deviazione std di misura
    rw_scale_prior: float = 0.05
    include_polls: bool = False    # se i sondaggi alimentano lo stato (presente agganciato)
    trend: bool = False            # dinamica local-linear-trend (livello+velocita') vs random walk

    @property
    def K(self) -> int:
        return len(self.parties)

    @property
    def T(self) -> int:
        return len(self.times)


def consensus_model(data: ModelData):
    """Modello NumPyro. Richiede jax/numpyro installati."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    Km1 = data.K - 1
    T = data.T
    n_types = len(data.election_types)
    G = len(data.regions)

    times = jnp.asarray(data.times)
    dt = jnp.diff(times)                       # (T-1,) gap temporali in mesi
    sqrt_dt = jnp.sqrt(jnp.clip(dt, 1e-6))

    # --- Stato iniziale e dinamica (random walk in tempo continuo, non centrato) ---
    eta0 = numpyro.sample("eta0", dist.Normal(jnp.zeros(Km1), 2.0).to_event(1))
    rw_scale = numpyro.sample("rw_scale", dist.HalfNormal(data.rw_scale_prior))
    if T > 1 and getattr(data, "trend", False):
        # LOCAL LINEAR TREND: stato = livello + velocita' (la velocita' e' essa stessa
        # un random walk). Estrapola la traiettoria recente, non la appiattisce.
        vel_scale = numpyro.sample("vel_scale", dist.HalfNormal(0.01))
        v0 = numpyro.sample("v0", dist.Normal(jnp.zeros(Km1), 0.02).to_event(1))
        zv = numpyro.sample("zv", dist.Normal(jnp.zeros((T - 1, Km1)), 1.0).to_event(2))
        v_steps = zv * (vel_scale * sqrt_dt[:, None])
        v = jnp.concatenate([v0[None, :], v0[None, :] + jnp.cumsum(v_steps, axis=0)], axis=0)
        z = numpyro.sample("z", dist.Normal(jnp.zeros((T - 1, Km1)), 1.0).to_event(2))
        level_steps = v[:-1] * dt[:, None] + z * (rw_scale * sqrt_dt[:, None])
        states = jnp.concatenate(
            [eta0[None, :], eta0[None, :] + jnp.cumsum(level_steps, axis=0)], axis=0)
        numpyro.deterministic("velocity", v)
    elif T > 1:
        # RANDOM WALK in tempo continuo (default)
        z = numpyro.sample("z", dist.Normal(jnp.zeros((T - 1, Km1)), 1.0).to_event(2))
        steps = z * (rw_scale * sqrt_dt[:, None])
        states = jnp.concatenate([eta0[None, :], eta0[None, :] + jnp.cumsum(steps, axis=0)],
                                 axis=0)                       # (T, K-1)
    else:
        states = eta0[None, :]

    # --- Bias tipo-elezione: ancora fissata a 0 ---
    beta_free = numpyro.sample(
        "beta", dist.Normal(jnp.zeros((n_types, Km1)), 1.0).to_event(2))
    # azzera la riga dell'ancora con una maschera (così resta identificabile)
    anchor_mask = jnp.asarray(
        [0.0 if i == data.anchor_type_idx else 1.0 for i in range(n_types)])
    beta = beta_free * anchor_mask[:, None]

    # --- Elasticità all'affluenza ---
    lam = numpyro.sample("lambda", dist.Normal(jnp.zeros(Km1), 0.5).to_event(1))

    # --- Offset regionali gerarchici (partial pooling) ---
    if G > 0:
        sigma_reg = numpyro.sample(
            "sigma_reg", dist.HalfNormal(0.5 * jnp.ones(Km1)).to_event(1))
        delta = numpyro.sample(
            "delta", dist.Normal(jnp.zeros((G, Km1)), 1.0).to_event(2)) * sigma_reg[None, :]
        delta_full = jnp.concatenate([jnp.zeros((1, Km1)), delta], axis=0)   # (G+1, K-1)
    else:
        delta_full = jnp.zeros((1, Km1))

    # --- Media attesa per osservazione ---
    ti = jnp.asarray(data.obs_time_idx)
    tyi = jnp.asarray(data.obs_type_idx)
    gi = jnp.asarray(data.obs_geo_idx)
    tdev = jnp.asarray(data.obs_turnout_dev)

    mu = (states[ti]                       # (N, K-1)
          + beta[tyi]
          + delta_full[gi]
          + lam[None, :] * tdev[:, None])

    sd = jnp.asarray(data.obs_sd)[:, None]
    mask = jnp.asarray(data.obs_mask)
    y = jnp.asarray(data.obs_eta)

    with numpyro.handlers.mask(mask=mask):
        numpyro.sample("y", dist.Normal(mu, sd), obs=y)

    # esponi gli stati come deterministici per il post-processing
    numpyro.deterministic("states", states)
