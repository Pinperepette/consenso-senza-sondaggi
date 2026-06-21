"""Orchestrazione della pipeline: ingest -> validate -> features -> model -> summarize.

Espone funzioni componibili usate sia dagli script CLI sia dallo scheduler.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from consenso.etl import features, validate
from consenso.etl.base import audit
from consenso.model.inference import assemble_model_data, run_inference
from consenso.model.summarize import summarize_run


def post_ingest(election_id: str) -> Dict:
    """Da eseguire dopo aver caricato i risultati e l'affluenza di un'elezione."""
    features.compute_shares(election_id)
    report = validate.validate_election(election_id)
    n_errors = sum(len(v) for v in report.values())
    audit("pipeline", "post_ingest", {"election_id": election_id, "errors": n_errors})
    return {"election_id": election_id, "validation": report, "n_errors": n_errors}


def run_model(election_ids: Optional[List[str]] = None,
              up_to_date: Optional[str] = None,
              include_regional: bool = True,
              include_polls: bool = False,
              trend: bool = True,
              num_warmup: Optional[int] = None,
              num_samples: Optional[int] = None,
              num_chains: Optional[int] = None) -> Dict:
    """Assembla i dati, esegue l'inferenza e materializza le stime."""
    data = assemble_model_data(election_ids=election_ids, up_to_date=up_to_date,
                               include_regional=include_regional,
                               include_polls=include_polls, trend=trend)
    run_id, _ = run_inference(data, num_warmup=num_warmup, num_samples=num_samples,
                              num_chains=num_chains)
    n_est = summarize_run(run_id)
    audit("pipeline", "run_model", {"run_id": run_id, "n_estimations": n_est,
                                    "n_obs": int(data.obs_eta.shape[0])})
    return {"run_id": run_id, "n_estimations": n_est,
            "n_parties": data.K, "n_obs": int(data.obs_eta.shape[0]),
            "n_times": data.T, "n_regions": len(data.regions)}
