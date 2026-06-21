#!/usr/bin/env python3
"""Backfill dello 'share' per i party_results comunali che ne sono privi
(le tornate storiche caricate prima del fix). share = voti / totale-comune."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import PARTY_RESULTS  # noqa: E402


def main() -> int:
    db = get_db()
    fixed = 0
    for e in db["elections"].find({"type": "comunali"}, {"_id": 1}):
        eid = e["_id"]
        rows = list(db[PARTY_RESULTS].find(
            {"election_id": eid, "geo_level": "comune"},
            {"geo_id": 1, "votes": 1, "share": 1}))
        if not any(r.get("share") is None for r in rows):
            continue
        totals = defaultdict(int)
        for r in rows:
            totals[r["geo_id"]] += r["votes"]
        ops = [UpdateOne({"_id": r["_id"]},
                         {"$set": {"share": (r["votes"] / totals[r["geo_id"]]
                                             if totals[r["geo_id"]] else 0.0),
                                   "valid_votes_area": totals[r["geo_id"]]}})
               for r in rows if r.get("share") is None]
        if ops:
            db[PARTY_RESULTS].bulk_write(ops, ordered=False)
            fixed += len(ops)
            print(f"  {eid}: share calcolato per {len(ops)} righe")
    print(f"TOTALE righe sistemate: {fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
