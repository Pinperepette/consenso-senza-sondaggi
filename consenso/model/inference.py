"""Assemblaggio dati dal DB, esecuzione NUTS e persistenza del run.

Trasforma i risultati elettorali curated in :class:`ModelData`, lancia il
campionamento, salva i posterior samples in GridFS e scrive il manifest
immutabile in ``model_runs`` (doc §3.4, §4.4).
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ANCHOR_ELECTION_TYPE, CONFIG, ELECTION_TYPES
from consenso.db.client import get_db, get_gridfs
from consenso.db.schema import ELECTIONS, MODEL_RUNS, PARTY_RESULTS, TURNOUT
from consenso.etl.features import mean_turnout_by_type
from consenso.etl.geo import region_of
from consenso.model.state_space import ModelData, consensus_model
from consenso.model.transforms import alr_np

# tipi che alimentano il modello di consenso nazionale.
# Esclusi: referendum (niente partiti) e comunali (dominanza di liste civiche
# locali -> non rappresentative del consenso nazionale; restano dato comune-level).
PARTY_ELECTION_TYPES = tuple(
    t for t in ELECTION_TYPES if t not in ("referendum", "comunali"))

# deviazione std di misura di base per tipo (più alto = meno rappresentativo del
# consenso nazionale strutturale)
BASE_OBS_SD = {
    "politiche": 0.03,
    "europee": 0.05,
    # una regionale copre una sola regione: è una misura DEBOLE del consenso
    # nazionale (alta varianza) -> rifinisce gli offset regionali senza scuotere
    # lo stato nazionale "del momento".
    "regionali": 0.18,
    "comunali": 0.20,
}

# numero massimo di partiti modellati esplicitamente (gli altri confluiscono nel
# riferimento "Altri")
DEFAULT_MAX_PARTIES = 9


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _months_between(d0: str, d1: str) -> float:
    a = datetime.fromisoformat(d0)
    b = datetime.fromisoformat(d1)
    return (b - a).days / 30.4375


def _finest_level(election_id: str) -> Optional[str]:
    levels = set(get_db()[PARTY_RESULTS].distinct("geo_level", {"election_id": election_id}))
    for lvl in ("comune", "provincia", "regione", "nazione"):
        if lvl in levels:
            return lvl
    return None


def _national_shares(election_id: str, level: str) -> Tuple[Dict[str, int], int]:
    """Voti per party_id (None -> 'altri') aggregati al livello scelto + totale."""
    pipeline = [
        {"$match": {"election_id": election_id, "geo_level": level}},
        {"$group": {"_id": "$party_id", "v": {"$sum": "$votes"}}},
    ]
    votes: Dict[str, int] = {}
    total = 0
    for r in get_db()[PARTY_RESULTS].aggregate(pipeline):
        pid = r["_id"] or "party:ALTRI"
        votes[pid] = votes.get(pid, 0) + r["v"]
        total += r["v"]
    return votes, total


def _geo_region_map(level: str) -> Dict[str, str]:
    """Mappa geo_id -> regione precaricata in memoria (evita una query per riga)."""
    from consenso.db.schema import GEOGRAPHIES

    return {g["_id"]: g.get("region")
            for g in get_db()[GEOGRAPHIES].find({"level": level}, {"region": 1})}


def _regional_shares(election_id: str, level: str) -> Dict[str, Dict[str, int]]:
    """Voti per (regione -> party_id), aggregando dal livello comune in memoria."""
    geo2reg = _geo_region_map(level)
    out: Dict[str, Dict[str, int]] = {}
    cur = get_db()[PARTY_RESULTS].find(
        {"election_id": election_id, "geo_level": level},
        {"geo_id": 1, "party_id": 1, "votes": 1})
    for r in cur:
        reg = geo2reg.get(r["geo_id"])
        if not reg:
            continue
        pid = r.get("party_id") or "party:ALTRI"
        out.setdefault(reg, {})
        out[reg][pid] = out[reg].get(pid, 0) + r["votes"]
    return out


def select_party_universe(election_ids: List[str], max_parties: int) -> List[str]:
    """Top partiti per voti totali + riferimento 'Altri' come ultimo elemento."""
    totals: Dict[str, int] = {}
    for eid in election_ids:
        lvl = _finest_level(eid)
        if not lvl:
            continue
        votes, _ = _national_shares(eid, lvl)
        for pid, v in votes.items():
            if pid == "party:ALTRI":
                continue
            totals[pid] = totals.get(pid, 0) + v
    ranked = sorted(totals, key=totals.get, reverse=True)[:max_parties]
    ref = CONFIG.model.reference_party
    parties = [p for p in ranked if p != ref]
    parties.append(ref)            # il riferimento è l'ultimo
    return parties


def assemble_model_data(election_ids: Optional[List[str]] = None,
                        up_to_date: Optional[str] = None,
                        max_parties: int = DEFAULT_MAX_PARTIES,
                        include_regional: bool = True) -> ModelData:
    q: Dict = {"type": {"$in": list(PARTY_ELECTION_TYPES)}}
    if election_ids:
        q["_id"] = {"$in": election_ids}
    if up_to_date:
        q["date"] = {"$lte": up_to_date}
    elections = list(get_db()[ELECTIONS].find(q).sort("date", 1))
    if not elections:
        raise ValueError("nessuna elezione selezionata per il modello")

    eids = [e["_id"] for e in elections]
    parties = select_party_universe(eids, max_parties)
    ref_idx = parties.index(CONFIG.model.reference_party)
    K = len(parties)
    pidx = {p: i for i, p in enumerate(parties)}

    types = list(PARTY_ELECTION_TYPES)
    type_idx = {t: i for i, t in enumerate(types)}
    anchor_type_idx = type_idx[ANCHOR_ELECTION_TYPE]

    t0 = elections[0]["date"]
    # tempi unici
    date_list = sorted({e["date"] for e in elections})
    times = np.array([_months_between(t0, d) for d in date_list])
    time_idx_of_date = {d: i for i, d in enumerate(date_list)}

    regions: List[str] = []
    region_index: Dict[str, int] = {}

    obs_eta: List[np.ndarray] = []
    obs_mask: List[np.ndarray] = []
    obs_time_idx: List[int] = []
    obs_type_idx: List[int] = []
    obs_geo_idx: List[int] = []
    obs_turnout_dev: List[float] = []
    obs_sd: List[float] = []

    def shares_vector(votes: Dict[str, int], total: int):
        """Costruisce il vettore quote di lunghezza K e la maschera dei presenti."""
        shares = np.zeros(K)
        present = np.zeros(K, dtype=bool)
        other = 0
        for pid, v in votes.items():
            if pid in pidx:
                shares[pidx[pid]] += v
                present[pidx[pid]] = True
            else:
                other += v
        shares[ref_idx] += other
        present[ref_idx] = True
        if total > 0:
            shares = shares / total
        return shares, present

    def region_slot(reg: str) -> int:
        if reg not in region_index:
            region_index[reg] = len(regions)
            regions.append(reg)
        return region_index[reg] + 1

    def add_obs(votes, total, etype, edate, geo_idx, sd):
        shares, present = shares_vector(votes, total)
        obs_eta.append(alr_np(shares, ref_idx))
        # maschera: dimensioni con quota ~0 (partito assente) sono ignorate
        obs_mask.append(np.delete(present & (shares > 1e-6), ref_idx))
        obs_time_idx.append(time_idx_of_date[edate])
        obs_type_idx.append(type_idx[etype])
        obs_geo_idx.append(geo_idx)
        obs_turnout_dev.append(_turnout_dev(eid, etype))
        obs_sd.append(sd)

    for e in elections:
        eid, etype, edate = e["_id"], e["type"], e["date"]
        lvl = _finest_level(eid)
        if not lvl:
            continue
        scope = e.get("scope", {})
        scope_level = scope.get("level", "nazionale")
        sd = BASE_OBS_SD.get(etype, 0.08)

        # una regionale (o un'elezione a scope regionale) è un SEGNALE REGIONALE,
        # non nazionale: va legata al proprio offset di regione, non a η nazionale.
        is_regional_scope = scope_level in ("regionale", "regional") or etype == "regionali"
        scope_regions = scope.get("geo_ids") or []

        if is_regional_scope and scope_regions:
            votes, total = _national_shares(eid, lvl)   # qui "national_shares" = totale della regione
            if total <= 0:
                continue
            add_obs(votes, total, etype, edate, region_slot(scope_regions[0]), sd)
            continue

        # elezione nazionale (politiche/europee): osservazione nazionale...
        votes, total = _national_shares(eid, lvl)
        if total <= 0:
            continue
        add_obs(votes, total, etype, edate, 0, sd)

        # ...più eventuali sotto-osservazioni regionali se ci sono dati per comune
        if include_regional:
            for reg, rvotes in _regional_shares(eid, lvl).items():
                rtot = sum(rvotes.values())
                if rtot > 0:
                    add_obs(rvotes, rtot, etype, edate, region_slot(reg), sd * 1.3)

    data = ModelData(
        parties=parties, ref_idx=ref_idx, election_types=types,
        anchor_type_idx=anchor_type_idx, regions=regions, times=times,
        obs_eta=np.asarray(obs_eta), obs_mask=np.asarray(obs_mask),
        obs_time_idx=np.asarray(obs_time_idx), obs_type_idx=np.asarray(obs_type_idx),
        obs_geo_idx=np.asarray(obs_geo_idx), obs_turnout_dev=np.asarray(obs_turnout_dev),
        obs_sd=np.asarray(obs_sd), rw_scale_prior=CONFIG.model.rw_scale_per_month,
    )
    return data


def _turnout_dev(eid: str, etype: str) -> float:
    """Scarto dell'affluenza dell'elezione dalla media storica del suo tipo.

    Usa l'affluenza più aggregata disponibile per l'elezione (nazione se c'è,
    altrimenti il livello regione per le regionali).
    """
    t = get_db()[TURNOUT].find_one(
        {"election_id": eid}, sort=[("eligible", -1)])
    if not t or t.get("turnout") is None:
        return 0.0
    base = mean_turnout_by_type(etype, exclude_election=eid)
    return float(t["turnout"] - base) if base is not None else 0.0


def _code_hash() -> str:
    import consenso.model.state_space as ss

    src = ss.__file__
    with open(src, "rb") as fh:
        return "sha256:" + hashlib.sha256(fh.read()).hexdigest()[:16]


def _data_hash(data: ModelData) -> str:
    h = hashlib.sha256()
    h.update(data.obs_eta.tobytes())
    h.update(data.obs_time_idx.tobytes())
    h.update(",".join(data.parties).encode())
    return "sha256:" + h.hexdigest()[:16]


def run_inference(data: ModelData, seed: int = 0,
                  num_warmup: Optional[int] = None,
                  num_samples: Optional[int] = None,
                  num_chains: Optional[int] = None) -> Tuple[str, Dict[str, np.ndarray]]:
    """Esegue NUTS, salva posterior + manifest. Restituisce (run_id, samples)."""
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    mc = CONFIG.model
    num_warmup = num_warmup or mc.num_warmup
    num_samples = num_samples or mc.num_samples
    num_chains = num_chains or mc.num_chains
    numpyro.set_host_device_count(num_chains)

    kernel = NUTS(consensus_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    # diagnostica R-hat (richiede >=2 catene; con 1 catena non è definito)
    rhat_max = None
    if num_chains >= 2:
        import arviz as az

        idata = az.from_numpyro(mcmc)
        rhat = az.rhat(idata)
        vals = [float(np.nanmax(rhat[v].values)) for v in rhat.data_vars]
        vals = [v for v in vals if np.isfinite(v)]
        rhat_max = float(max(vals)) if vals else None

    run_id = "run:" + _utcnow().strftime("%Y%m%dT%H%M%S")
    ref = _save_samples(run_id, samples, data)
    get_db()[MODEL_RUNS].insert_one({
        "_id": run_id, "model_version": "ssm-1.0.0",
        "code_hash": _code_hash(), "input_data_hash": _data_hash(data),
        "hyperparams": {"rw_scale_prior": data.rw_scale_prior,
                        "reference_party": CONFIG.model.reference_party,
                        "parties": data.parties, "regions": data.regions,
                        "election_types": data.election_types,
                        "times": data.times.tolist()},
        "elections_used": [e for e in []],
        "inference": {"method": "NUTS", "draws": num_samples,
                      "chains": num_chains, "rhat_max": rhat_max},
        "samples_ref": ref,
        "created_at": _utcnow(),
        "status": ("completed" if (rhat_max is None or rhat_max < mc.rhat_threshold)
                   else "completed_low_quality"),
    })
    return run_id, samples


def _save_samples(run_id: str, samples: Dict[str, np.ndarray], data: ModelData) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **samples)
    buf.seek(0)
    fid = get_gridfs().put(buf.read(), filename=f"{run_id}.npz", run_id=run_id)
    return f"gridfs://{fid}"


def load_samples(run_id: str) -> Dict[str, np.ndarray]:
    run = get_db()[MODEL_RUNS].find_one({"_id": run_id})
    if not run:
        raise KeyError(run_id)
    fid = run["samples_ref"].split("//", 1)[1]
    from bson import ObjectId

    data = get_gridfs().get(ObjectId(fid)).read()
    npz = np.load(io.BytesIO(data))
    return {k: npz[k] for k in npz.files}
