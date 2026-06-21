#!/usr/bin/env python3
"""Costruisce il TRACK RECORD onesto del modello: per ogni elezione nazionale
passata, allena sui soli dati PRECEDENTI (cutoff ~1 mese prima) e confronta la
previsione col risultato reale. Salva in ``track_record``.

E' il modo piu' onesto di mostrare quanto vale il sistema: include anche le
previsioni sbagliate.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.validation.backtest import run_backtest  # noqa: E402

LEAD_DAYS = 30          # quanto prima del voto "ci fermiamo" (previsione, non eve)
SINCE = "2013-01-01"    # prima c'e' troppo poca storia per allenare


def main() -> int:
    db = get_db()
    nat = list(db["elections"].find(
        {"type": {"$in": ["politiche", "europee"]}, "date": {"$gte": SINCE}},
        {"date": 1, "type": 1}).sort("date", 1))
    print(f"elezioni da valutare: {[e['_id'] for e in nat]}")
    for e in nat:
        cutoff = (date.fromisoformat(e["date"]) - timedelta(days=LEAD_DAYS)).isoformat()
        print(f"\n>> {e['_id']} (cutoff {cutoff})", flush=True)
        try:
            res = run_backtest(cutoff, num_warmup=500, num_samples=500,
                               include_polls=True, trend=True)
        except Exception as exc:  # noqa: BLE001
            print(f"   saltata: {exc}"); continue
        doc = db["backtests"].find_one({"train_run": res["train_run"]})
        tgt = next((t for t in (doc or {}).get("targets", [])
                    if t["target"] == e["_id"]), None)
        if not tgt:
            print("   nessun target trovato"); continue
        pp = [{"party": p["party_id"].replace("party:", ""), "pred": p["pred_mean"],
               "ci95": p.get("ci95"), "actual": p["actual"], "in_ci": p["in_ci95"],
               "err": p["abs_err"]} for p in tgt["per_party"]]
        rec = {"_id": e["_id"], "type": e["type"], "date": e["date"], "cutoff": cutoff,
               "lead_days": LEAD_DAYS, "parties": pp,
               "mae": float(np.mean([p["err"] for p in pp])),
               "coverage": float(np.mean([1.0 if p["in_ci"] else 0.0 for p in pp]))}
        db["track_record"].replace_one({"_id": e["_id"]}, rec, upsert=True)
        print(f"   MAE {rec['mae']*100:.1f} pt, coverage {rec['coverage']*100:.0f}%")
    print("\nTRACK RECORD COSTRUITO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
