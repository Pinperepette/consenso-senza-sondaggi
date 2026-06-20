#!/usr/bin/env python3
"""Precalcola la sintesi spaziale (carta d'identità per partito) e la salva come
JSON statico servito dalla dashboard.

Per ogni partito: quote medie per macro-area, correlazione col reddito,
R² del modello RandomForest. Calcolato una volta (è pesante), letto a ogni
richiesta dalla GUI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.model.spatial_ml import PARTIES, build_dataset, train  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "consenso" / "api" / "static" / "spatial_summary.json"
AREAS = ["NO", "NE", "C", "S", "I"]


def main(election_id: str = "elez:2022_politiche") -> int:
    df = build_dataset(election_id).dropna(subset=["avg_income"])
    rep = train(election_id)
    summary = {"election_id": election_id, "parties": {}}
    for p in PARTIES:
        if p not in df.columns:
            continue
        by_area = (df.groupby("macro")[p].mean() * 100).to_dict()
        r_inc = float(np.corrcoef(df[p].fillna(0), df["avg_income"])[0, 1])
        r_emp = float(np.corrcoef(df[p].fillna(0), df["pct_employee"].fillna(0))[0, 1])
        r2 = rep["parties"].get(p, {}).get("r2_cv_mean")
        drivers = rep["parties"].get(p, {}).get("top_drivers", [])
        summary["parties"][p.replace("party:", "")] = {
            "by_area": {a: round(by_area.get(a, 0.0), 1) for a in AREAS},
            "income_corr": round(r_inc, 2),
            "employee_corr": round(r_emp, 2),
            "r2": round(r2, 2) if r2 is not None else None,
            "drivers": [[n, v] for n, v in drivers[:4]],
        }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("salvato", OUT, "-", len(summary["parties"]), "partiti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
