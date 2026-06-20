"""Costruzione delle feature derivate usate dal modello.

  - quota (share) = voti / voti validi area;
  - affluenza differenziale = affluenza elezione − media storica del suo tipo;
  - join con la demografia ISTAT più recente alla data dell'elezione.
"""
from __future__ import annotations

from typing import Dict, Optional

from pymongo import UpdateOne

from consenso.db.client import get_db
from consenso.db.schema import DEMOGRAPHICS, ELECTIONS, PARTY_RESULTS, TURNOUT


def compute_shares(election_id: str) -> int:
    """Calcola e salva ``share`` per ogni party_result dell'elezione."""
    ops = []
    for r in get_db()[PARTY_RESULTS].find(
        {"election_id": election_id}, {"votes": 1, "valid_votes_area": 1}
    ):
        valid = r.get("valid_votes_area") or 0
        share = (r["votes"] / valid) if valid > 0 else None
        ops.append(UpdateOne({"_id": r["_id"]}, {"$set": {"share": share}}))
    if ops:
        get_db()[PARTY_RESULTS].bulk_write(ops)
    return len(ops)


def mean_turnout_by_type(election_type: str, exclude_election: Optional[str] = None) -> Optional[float]:
    """Affluenza media nazionale storica per un tipo di elezione."""
    match = {"_meta_type": election_type}
    # join via elections per recuperare il tipo
    elec_ids = [e["_id"] for e in get_db()[ELECTIONS].find(
        {"type": election_type}, {"_id": 1})]
    if exclude_election:
        elec_ids = [e for e in elec_ids if e != exclude_election]
    if not elec_ids:
        return None
    pipeline = [
        {"$match": {"election_id": {"$in": elec_ids}, "geo_level": "nazione"}},
        {"$group": {"_id": None, "m": {"$avg": "$turnout"}}},
    ]
    rows = list(get_db()[TURNOUT].aggregate(pipeline))
    if rows:
        return rows[0]["m"]
    # fallback: media su tutte le aree
    pipeline[0]["$match"].pop("geo_level")
    rows = list(get_db()[TURNOUT].aggregate(pipeline))
    return rows[0]["m"] if rows else None


def differential_turnout(election_id: str) -> Dict[str, float]:
    """Scarto di affluenza per area rispetto alla media storica del tipo."""
    elec = get_db()[ELECTIONS].find_one({"_id": election_id}, {"type": 1})
    if not elec:
        return {}
    baseline = mean_turnout_by_type(elec["type"], exclude_election=election_id)
    out: Dict[str, float] = {}
    for t in get_db()[TURNOUT].find({"election_id": election_id}, {"geo_id": 1, "turnout": 1}):
        if baseline is not None and t.get("turnout") is not None:
            out[t["geo_id"]] = t["turnout"] - baseline
    return out


def attach_demographics(geo_id: str, on_year: int) -> Optional[dict]:
    """Demografia più recente per l'area, con anno ≤ on_year."""
    return get_db()[DEMOGRAPHICS].find_one(
        {"geo_id": geo_id, "year": {"$lte": on_year}}, sort=[("year", -1)]
    )
