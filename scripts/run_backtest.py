#!/usr/bin/env python3
"""Validazione temporale: addestra fino a una data e confronta con le elezioni
successive.

  python scripts/run_backtest.py --train-until 2018-12-31
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.validation.backtest import run_backtest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest temporale del modello")
    ap.add_argument("--train-until", required=True, help="data ISO (es. 2018-12-31)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()
    res = run_backtest(args.train_until, num_warmup=args.warmup, num_samples=args.samples)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
