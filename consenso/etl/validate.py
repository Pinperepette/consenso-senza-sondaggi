"""Validazioni di quadratura sui dati curated.

Le anomalie non vengono scartate: vanno in ``quarantine`` con un motivo, così
restano ispezionabili e non spariscono silenziosamente (doc §4.2).
"""
from __future__ import annotations

from typing import Dict, List

from consenso.db.client import get_db
from consenso.db.schema import PARTY_RESULTS, QUARANTINE, TURNOUT
from consenso.etl.base import utcnow

# tolleranza relativa sulle somme (arrotondamenti, voti dispersi)
REL_TOL = 0.02


def quarantine(kind: str, reason: str, payload: dict) -> None:
    get_db()[QUARANTINE].insert_one(
        {"kind": kind, "reason": reason, "payload": payload, "created_at": utcnow(),
         "election_id": payload.get("election_id")}
    )


def validate_turnout(election_id: str) -> List[str]:
    """Affluenza in [0,1] e voters<=eligible."""
    errors: List[str] = []
    for t in get_db()[TURNOUT].find({"election_id": election_id}):
        if not (0.0 <= t.get("turnout", -1) <= 1.0):
            msg = f"turnout fuori range in {t['geo_id']}: {t.get('turnout')}"
            errors.append(msg)
            quarantine("turnout", msg, t)
        if t.get("voters", 0) > t.get("eligible", 0):
            msg = f"voters>eligible in {t['geo_id']}"
            errors.append(msg)
            quarantine("turnout", msg, t)
    return errors


def validate_results_sum(election_id: str) -> List[str]:
    """La somma dei voti di lista per area deve combaciare con i voti validi area."""
    errors: List[str] = []
    pipeline = [
        {"$match": {"election_id": election_id}},
        {"$group": {"_id": "$geo_id",
                    "sum_votes": {"$sum": "$votes"},
                    "valid": {"$max": "$valid_votes_area"}}},
    ]
    for row in get_db()[PARTY_RESULTS].aggregate(pipeline):
        valid = row.get("valid") or 0
        s = row.get("sum_votes") or 0
        if valid <= 0:
            continue
        if abs(s - valid) > REL_TOL * valid:
            msg = (f"somma liste {s} ≠ voti validi {valid} in {row['_id']} "
                   f"(scarto { (s-valid)/valid:.1%})")
            errors.append(msg)
            quarantine("results_sum", msg,
                       {"election_id": election_id, "geo_id": row["_id"],
                        "sum_votes": s, "valid_votes_area": valid})
    return errors


def validate_hierarchy_consistency(election_id: str) -> List[str]:
    """I voti a livello comune devono sommare a quelli di provincia/regione.

    Eseguita solo se esistono più livelli per la stessa elezione.
    """
    errors: List[str] = []
    levels = get_db()[PARTY_RESULTS].distinct("geo_level", {"election_id": election_id})
    if "comune" in levels and "provincia" in levels:
        # confronto aggregato per partito
        com = _sum_by_party(election_id, "comune")
        prov = _sum_by_party(election_id, "provincia")
        for party_id, v_com in com.items():
            v_prov = prov.get(party_id)
            if v_prov and abs(v_com - v_prov) > REL_TOL * v_prov:
                msg = f"comuni({v_com}) ≠ province({v_prov}) per {party_id}"
                errors.append(msg)
                quarantine("hierarchy", msg,
                           {"election_id": election_id, "party_id": party_id})
    return errors


def _sum_by_party(election_id: str, level: str) -> Dict[str, int]:
    pipeline = [
        {"$match": {"election_id": election_id, "geo_level": level}},
        {"$group": {"_id": "$party_id", "v": {"$sum": "$votes"}}},
    ]
    return {r["_id"]: r["v"] for r in get_db()[PARTY_RESULTS].aggregate(pipeline) if r["_id"]}


def validate_election(election_id: str) -> Dict[str, List[str]]:
    """Esegue tutte le validazioni. Dizionario {check: [errori]}."""
    return {
        "turnout": validate_turnout(election_id),
        "results_sum": validate_results_sum(election_id),
        "hierarchy": validate_hierarchy_consistency(election_id),
    }
