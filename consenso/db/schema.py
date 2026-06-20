"""Definizione delle collection MongoDB e dei loro indici.

Convenzione (dal documento tecnico §3):
  - raw_*    : payload grezzi immutabili, append-only (audit/riproducibilità)
  - curated  : dati normalizzati pronti per il modello
  - derived  : output del modello, versionati per run

Tutte le scritture derived sono immutabili: una nuova stima = nuovo run_id,
mai update in place.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pymongo import ASCENDING, DESCENDING, GEOSPHERE, IndexModel

from .client import get_db

# Nomi canonici delle collection -------------------------------------------------
RAW = "raw_ingestions"            # payload grezzi (un doc per file/risorsa scaricata)
GEOGRAPHIES = "geographies"
GEO_REMAP = "geo_remap"           # mapping storico fra codici comune (fusioni/cessazioni)
PARTIES = "parties"
PARTY_ALIASES = "party_aliases"
COALITIONS = "coalitions"
ELECTIONS = "elections"
PARTY_RESULTS = "party_results"
TURNOUT = "turnout"
DEMOGRAPHICS = "demographics"
RECONCILE_QUEUE = "reconcile_queue"   # etichette nuove in attesa di revisione umana
QUARANTINE = "quarantine"             # dati che falliscono le quadrature
MODEL_RUNS = "model_runs"
ESTIMATIONS = "estimations"
FLOW_MODELS = "flow_models"
BACKTESTS = "backtests"
AUDIT_LOG = "audit_log"

ALL_COLLECTIONS: Tuple[str, ...] = (
    RAW, GEOGRAPHIES, GEO_REMAP, PARTIES, PARTY_ALIASES, COALITIONS, ELECTIONS,
    PARTY_RESULTS, TURNOUT, DEMOGRAPHICS, RECONCILE_QUEUE, QUARANTINE,
    MODEL_RUNS, ESTIMATIONS, FLOW_MODELS, BACKTESTS, AUDIT_LOG,
)

# Indici per collection ----------------------------------------------------------
INDEXES: Dict[str, List[IndexModel]] = {
    RAW: [
        # idempotenza: lo stesso payload non viene caricato due volte
        IndexModel([("source_hash", ASCENDING)], unique=True, name="uq_source_hash"),
        IndexModel([("source", ASCENDING), ("ingested_at", DESCENDING)], name="src_time"),
    ],
    GEOGRAPHIES: [
        IndexModel([("level", ASCENDING), ("parent", ASCENDING)], name="level_parent"),
        IndexModel([("istat_code", ASCENDING), ("valid_from", ASCENDING)], name="istat_valid"),
        IndexModel([("centroid", GEOSPHERE)], name="geo_centroid", sparse=True),
    ],
    GEO_REMAP: [
        IndexModel([("old_code", ASCENDING), ("year", ASCENDING)], name="old_year"),
    ],
    PARTIES: [
        IndexModel([("canonical_name", ASCENDING)], name="canon_name"),
    ],
    PARTY_ALIASES: [
        IndexModel([("raw_label", ASCENDING), ("valid_from", ASCENDING)], name="label_valid"),
        IndexModel([("party_id", ASCENDING)], name="party"),
    ],
    COALITIONS: [
        IndexModel([("election_id", ASCENDING)], name="election"),
    ],
    ELECTIONS: [
        IndexModel([("type", ASCENDING), ("date", DESCENDING)], name="type_date"),
        IndexModel([("date", DESCENDING)], name="date"),
    ],
    PARTY_RESULTS: [
        IndexModel([("election_id", ASCENDING), ("geo_id", ASCENDING)], name="elec_geo"),
        IndexModel([("party_id", ASCENDING), ("geo_level", ASCENDING), ("geo_id", ASCENDING)],
                   name="party_geo"),
    ],
    TURNOUT: [
        IndexModel([("election_id", ASCENDING), ("geo_id", ASCENDING)], unique=True,
                   name="uq_elec_geo"),
    ],
    DEMOGRAPHICS: [
        IndexModel([("geo_id", ASCENDING), ("year", DESCENDING)], name="geo_year"),
    ],
    RECONCILE_QUEUE: [
        IndexModel([("raw_label", ASCENDING)], unique=True, name="uq_label"),
        IndexModel([("status", ASCENDING)], name="status"),
    ],
    QUARANTINE: [
        IndexModel([("election_id", ASCENDING)], name="election"),
        IndexModel([("created_at", DESCENDING)], name="time"),
    ],
    MODEL_RUNS: [
        IndexModel([("created_at", DESCENDING)], name="time"),
        IndexModel([("status", ASCENDING), ("created_at", DESCENDING)], name="status_time"),
    ],
    ESTIMATIONS: [
        IndexModel([("run_id", ASCENDING), ("geo_level", ASCENDING), ("as_of", DESCENDING)],
                   name="run_level_time"),
        IndexModel([("party_id", ASCENDING), ("geo_id", ASCENDING), ("as_of", DESCENDING)],
                   name="party_geo_time"),
    ],
    FLOW_MODELS: [
        IndexModel([("run_id", ASCENDING)], name="run"),
        IndexModel([("from_election", ASCENDING), ("to_election", ASCENDING),
                    ("geo_scope", ASCENDING)], name="from_to_scope"),
    ],
    BACKTESTS: [
        IndexModel([("scheme", ASCENDING), ("target", ASCENDING)], name="scheme_target"),
    ],
    AUDIT_LOG: [
        IndexModel([("ts", DESCENDING)], name="ts"),
        IndexModel([("actor", ASCENDING), ("action", ASCENDING)], name="actor_action"),
    ],
}


def ensure_collections() -> List[str]:
    """Crea (idempotente) tutte le collection e i relativi indici.

    Restituisce la lista delle collection garantite.
    """
    db = get_db()
    existing = set(db.list_collection_names())
    created: List[str] = []
    for name in ALL_COLLECTIONS:
        if name not in existing:
            db.create_collection(name)
            created.append(name)
        idx = INDEXES.get(name)
        if idx:
            db[name].create_indexes(idx)
    return created
