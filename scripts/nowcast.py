#!/usr/bin/env python3
"""Nowcast CLI: stampa la stima corrente del consenso (default: oggi).

Usa l'unica implementazione in consenso.model.nowcast (ancorata all'ultima
elezione NAZIONALE), così CLI, API e GUI danno gli stessi numeri.

  python scripts/nowcast.py --as-of 2026-06-19
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.model.nowcast import nowcast  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    res = nowcast(ap.parse_args().as_of)
    if "error" in res:
        print(res["error"]); return 1
    lr = res["last_election"]
    print(f"NOWCAST consenso nazionale al {res['as_of']}")
    print(f"(ultimo voto reale: {lr['type']} {lr['date']}; "
          f"proiezione di {res['projection_months']:.0f} mesi)\n")
    print(f"{'partito':12s} {'stima':>7} {'IC 95%':>18} {'P(>10%)':>8} {'P(>20%)':>8}")
    for p in res["parties"]:
        t = p["prob_thresholds"]
        print(f"{p['name']:12s} {p['mean']*100:6.1f}% "
              f"[{p['ci95'][0]*100:5.1f}%,{p['ci95'][1]*100:5.1f}%] "
              f"{t['>0.1']:7.0%} {t['>0.2']:7.0%}")
    print(f"\nP({res['leader']} primo partito) = {res['p_leader_first']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
