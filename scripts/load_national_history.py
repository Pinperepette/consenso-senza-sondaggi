#!/usr/bin/env python3
"""Carica le elezioni nazionali STORICHE (2008-2019) come aggregati nazionali,
da una fixture nel repo (data/fixtures/national_history.json).

Servono come ancore di training per i backtest del Track record: senza, il
bootstrap da zero potrebbe valutare solo le elezioni piu' recenti. Sono aggregati
nazionali (il track record nazionale non richiede il dettaglio comunale).

Idempotente: salta le elezioni che hanno gia' risultati (non declassa una fonte
piu' ricca, es. dati comune-level gia' presenti)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import PARTY_RESULTS  # noqa: E402
from consenso.etl.sources.eligendo import register_election  # noqa: E402

FIXTURE = ROOT / "data" / "fixtures" / "national_history.json"
GEO = "ISTAT:IT"


def main() -> int:
    if not FIXTURE.exists():
        print(f"(fixture assente: {FIXTURE})")
        return 0
    db = get_db()
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    loaded = 0
    for e in data:
        eid = e["election_id"]
        if db[PARTY_RESULTS].count_documents({"election_id": eid}) > 0:
            print(f"  {eid}: già presente, salto")
            continue
        register_election(eid, e["type"], e["date"], {"level": "nazionale", "source": "fixture"})
        tot = e["total"] or sum(e["parties"].values()) or 1
        docs = [{"election_id": eid, "geo_id": GEO, "geo_level": "nazione",
                 "party_id": pid, "raw_label": None, "votes": int(v),
                 "valid_votes_area": int(tot), "share": v / tot,
                 "_meta": {"source": "national_history_fixture"}}
                for pid, v in e["parties"].items()]
        db[PARTY_RESULTS].insert_many(docs)
        loaded += 1
        print(f"  {eid}: {len(docs)} partiti (aggregato nazionale)")
    print(f"NAZIONALI STORICHE: {loaded} elezioni caricate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
