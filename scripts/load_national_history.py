#!/usr/bin/env python3
"""Carica le elezioni STORICHE non coperte dagli altri loader, da una fixture
nel repo (data/fixtures/national_history.json): nazionali 2008-2019 (aggregati)
+ regionali 2023/2024 (aggregati per regione).

Sono ANCORE DI TRAINING per i backtest del Track record: senza, il bootstrap da
zero valuta solo le elezioni piu' recenti e il backtest perde precisione (mancano
i punti intermedi 2023/2024). Le righe preservano geo_id/livello reali.

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
        level = (e["rows"][0]["geo_level"] if e.get("rows") else "nazione")
        register_election(eid, e["type"], e["date"], {"level": level, "source": "fixture"})
        docs = [{"election_id": eid, "geo_id": r["geo_id"], "geo_level": r["geo_level"],
                 "party_id": r["party_id"], "raw_label": None, "votes": int(r.get("votes", 0)),
                 "valid_votes_area": int(r.get("valid_votes_area", 0)), "share": r.get("share"),
                 "_meta": {"source": "history_fixture"}}
                for r in e.get("rows", [])]
        if docs:
            db[PARTY_RESULTS].insert_many(docs)
        loaded += 1
        print(f"  {eid}: {len(docs)} righe ({e['type']})")
    print(f"STORICHE (fixture): {loaded} elezioni caricate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
