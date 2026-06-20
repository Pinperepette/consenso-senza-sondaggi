"""Test end-to-end: seed di elezioni reali (semplificate) -> run del modello ->
stime nazionali coerenti -> servizio via API."""
import numpy as np

from consenso.db.client import get_db
from consenso.db.schema import ELECTIONS, ESTIMATIONS, PARTY_RESULTS, TURNOUT

SCALE = 100_000

SCENARIO = {
    "elez:2018_politiche": ("politiche", "2018-03-04", 0.729, {
        "party:M5S": 0.327, "party:LEGA": 0.173, "party:PD": 0.187,
        "party:FI": 0.140, "party:FDI": 0.043}),
    "elez:2019_europee": ("europee", "2019-05-26", 0.564, {
        "party:LEGA": 0.341, "party:PD": 0.227, "party:M5S": 0.171,
        "party:FI": 0.089, "party:FDI": 0.063}),
    "elez:2022_politiche": ("politiche", "2022-09-25", 0.639, {
        "party:FDI": 0.260, "party:PD": 0.190, "party:M5S": 0.155,
        "party:LEGA": 0.087, "party:FI": 0.082}),
}


def _seed():
    db = get_db()
    for eid, (etype, date, turn, shares) in SCENARIO.items():
        db[ELECTIONS].insert_one({"_id": eid, "type": etype, "date": date,
                                  "scope": {"level": "nazionale"}})
        rows = []
        total = SCALE
        accounted = 0.0
        for pid, sh in shares.items():
            v = int(sh * total)
            accounted += sh
            rows.append({"election_id": eid, "geo_id": "ISTAT:IT", "geo_level": "nazione",
                         "party_id": pid, "raw_label": pid, "votes": v,
                         "valid_votes_area": total})
        # resto -> Altri
        rows.append({"election_id": eid, "geo_id": "ISTAT:IT", "geo_level": "nazione",
                     "party_id": "party:ALTRI", "raw_label": "ALTRI",
                     "votes": int((1 - accounted) * total), "valid_votes_area": total})
        db[PARTY_RESULTS].insert_many(rows)
        db[TURNOUT].insert_one({"election_id": eid, "geo_id": "ISTAT:IT",
                                "geo_level": "nazione", "eligible": int(total / turn),
                                "voters": total, "turnout": turn})


def test_full_pipeline_and_api():
    _seed()
    from consenso.pipeline.orchestrate import run_model

    res = run_model(include_regional=False, num_warmup=300, num_samples=300)
    run_id = res["run_id"]
    assert res["n_estimations"] > 0
    assert res["n_obs"] == 3                      # tre elezioni nazionali

    # le quote nazionali all'ultimo tempo devono sommare ~1
    last_as_of = sorted(get_db()[ESTIMATIONS].distinct(
        "as_of", {"run_id": run_id, "geo_level": "nazione"}))[-1]
    docs = list(get_db()[ESTIMATIONS].find(
        {"run_id": run_id, "geo_level": "nazione", "as_of": last_as_of}))
    total_mean = sum(d["mean"] for d in docs)
    assert abs(total_mean - 1.0) < 0.05

    # FdI nel 2022 deve risultare il primo partito ed essere ben sopra il 15%
    fdi = next(d for d in docs if d["party_id"] == "party:FDI")
    assert fdi["mean"] > 0.18
    assert fdi["prob_thresholds"][">0.15"] > 0.5
    # intervallo credibile coerente
    assert fdi["ci95"][0] < fdi["mean"] < fdi["ci95"][1]

    # --- serving via API ---
    from consenso.api.app import create_app

    client = create_app().test_client()
    r = client.get(f"/estimate/national?party=party:FDI")
    assert r.status_code == 200
    body = r.get_json()
    assert "ci95" in body and "prob_thresholds" in body

    t = client.get("/trend?party=party:M5S")
    assert t.status_code == 200
    assert len(t.get_json()["series"]) >= 1
