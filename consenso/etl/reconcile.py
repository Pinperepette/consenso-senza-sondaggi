"""Riconciliazione delle etichette di lista verso entità partito canoniche.

Regola d'oro (doc §6): nessun merge silenzioso. Un'etichetta mai vista finisce
in una coda di revisione (``reconcile_queue``) con suggerimenti automatici, ma
la conferma è umana. Le relazioni sono tipizzate: alias|merge|split|
coalition_member|civic e portano un ``weight`` (per scissioni/fusioni).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rapidfuzz import fuzz, process

from consenso.db.client import get_db
from consenso.db.schema import PARTIES, PARTY_ALIASES, RECONCILE_QUEUE


def normalize_label(label: str) -> str:
    """Normalizza un'etichetta grezza per il matching (maiuscole, spazi, punteggiatura)."""
    s = label.upper().strip()
    s = re.sub(r"[^A-Z0-9ÀÈÉÌÒÙ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def resolve_label(raw_label: str, on_date: Optional[str] = None) -> Optional[str]:
    """Mappa un'etichetta a un ``party_id`` se esiste un alias valido alla data."""
    norm = normalize_label(raw_label)
    q = {"$or": [{"raw_label": raw_label}, {"raw_label_norm": norm}]}
    cur = get_db()[PARTY_ALIASES].find(q)
    candidates = list(cur)
    if not candidates:
        return None
    if on_date:
        for a in candidates:
            vf, vt = a.get("valid_from"), a.get("valid_to")
            if (vf is None or vf <= on_date) and (vt is None or on_date <= vt):
                return a["party_id"]
    return candidates[0]["party_id"]


def _alias_index() -> List[Tuple[str, str]]:
    """Coppie (etichetta_normalizzata, party_id) note, per il fuzzy match."""
    out: List[Tuple[str, str]] = []
    for a in get_db()[PARTY_ALIASES].find({}, {"raw_label": 1, "party_id": 1}):
        out.append((normalize_label(a["raw_label"]), a["party_id"]))
    for p in get_db()[PARTIES].find({}, {"canonical_name": 1}):
        out.append((normalize_label(p["canonical_name"]), p["_id"]))
    return out


def suggest(raw_label: str, limit: int = 5) -> List[dict]:
    """Suggerisce le entità partito più simili (token-sort ratio)."""
    norm = normalize_label(raw_label)
    index = _alias_index()
    if not index:
        return []
    choices = {i: lbl for i, (lbl, _) in enumerate(index)}
    matches = process.extract(norm, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    out = []
    for _, score, idx in matches:
        lbl, pid = index[idx]
        out.append({"party_id": pid, "matched_label": lbl, "score": float(score)})
    return out


def enqueue_unknown(raw_label: str, context: Optional[dict] = None) -> None:
    """Mette un'etichetta sconosciuta in coda di revisione con suggerimenti."""
    get_db()[RECONCILE_QUEUE].update_one(
        {"raw_label": raw_label},
        {"$set": {"raw_label": raw_label, "raw_label_norm": normalize_label(raw_label),
                  "suggestions": suggest(raw_label), "status": "pending",
                  "context": context or {}},
         "$inc": {"occurrences": 1}},
        upsert=True,
    )


def register_party(party_id: str, canonical_name: str, family: Optional[str] = None) -> None:
    get_db()[PARTIES].update_one(
        {"_id": party_id},
        {"$set": {"canonical_name": canonical_name, "family": family}},
        upsert=True,
    )


def register_alias(raw_label: str, party_id: str, relation: str = "alias",
                   weight: float = 1.0, valid_from: Optional[str] = None,
                   valid_to: Optional[str] = None, source: str = "manual") -> None:
    get_db()[PARTY_ALIASES].update_one(
        {"raw_label": raw_label, "valid_from": valid_from},
        {"$set": {"raw_label": raw_label, "raw_label_norm": normalize_label(raw_label),
                  "party_id": party_id, "relation": relation, "weight": weight,
                  "valid_to": valid_to, "source": source}},
        upsert=True,
    )
    # appena registrato un alias, l'eventuale voce in coda è risolta
    get_db()[RECONCILE_QUEUE].update_one(
        {"raw_label": raw_label}, {"$set": {"status": "resolved"}}
    )
