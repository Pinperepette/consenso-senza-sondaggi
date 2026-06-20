#!/usr/bin/env python3
"""Esegue un model run completo: assembla i dati, lancia NUTS, materializza le stime.

  python scripts/run_model.py                      # usa tutte le elezioni
  python scripts/run_model.py --up-to 2018-12-31   # solo fino a una data (backtest)
  python scripts/run_model.py --no-regional --warmup 500 --samples 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.pipeline.orchestrate import run_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Esegue un model run")
    ap.add_argument("--up-to", default=None, help="data ISO; usa solo elezioni <= data")
    ap.add_argument("--no-regional", action="store_true")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    res = run_model(up_to_date=args.up_to, include_regional=not args.no_regional,
                    num_warmup=args.warmup, num_samples=args.samples)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
