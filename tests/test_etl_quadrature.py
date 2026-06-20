"""Le validazioni di quadratura devono intercettare somme incoerenti e affluenze
fuori range, mettendole in quarantine senza scartarle."""
from consenso.db.client import get_db
from consenso.db.schema import PARTY_RESULTS, QUARANTINE, TURNOUT
from consenso.etl import features, validate


def _seed_election(eid="elez:test"):
    db = get_db()
    # comune A coerente, comune B incoerente (somma liste != voti validi)
    db[PARTY_RESULTS].insert_many([
        {"election_id": eid, "geo_id": "ISTAT:000001", "geo_level": "comune",
         "party_id": "party:X", "raw_label": "X", "votes": 600, "valid_votes_area": 1000},
        {"election_id": eid, "geo_id": "ISTAT:000001", "geo_level": "comune",
         "party_id": "party:Y", "raw_label": "Y", "votes": 400, "valid_votes_area": 1000},
        {"election_id": eid, "geo_id": "ISTAT:000002", "geo_level": "comune",
         "party_id": "party:X", "raw_label": "X", "votes": 100, "valid_votes_area": 1000},
    ])
    db[TURNOUT].insert_many([
        {"election_id": eid, "geo_id": "ISTAT:000001", "geo_level": "comune",
         "eligible": 1500, "voters": 1050, "turnout": 0.70},
        {"election_id": eid, "geo_id": "ISTAT:000002", "geo_level": "comune",
         "eligible": 1200, "voters": 2000, "turnout": 1.67},   # incoerente
    ])
    return eid


def test_results_sum_quarantine():
    eid = _seed_election()
    errors = validate.validate_results_sum(eid)
    assert any("000002" in e for e in errors)
    assert get_db()[QUARANTINE].count_documents({"kind": "results_sum"}) == 1


def test_turnout_out_of_range():
    eid = _seed_election()
    errors = validate.validate_turnout(eid)
    assert any("000002" in e for e in errors)
    q = get_db()[QUARANTINE].count_documents({"kind": "turnout"})
    assert q >= 1


def test_compute_shares():
    eid = _seed_election()
    n = features.compute_shares(eid)
    assert n == 3
    doc = get_db()[PARTY_RESULTS].find_one(
        {"election_id": eid, "geo_id": "ISTAT:000001", "party_id": "party:X"})
    assert abs(doc["share"] - 0.6) < 1e-9
