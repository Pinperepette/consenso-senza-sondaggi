#!/usr/bin/env python3
"""Aggancio automatico: scarica e ingerisce le elezioni nuove dall'archivio
del Ministero, poi (salvo --no-rerun) rilancia il modello.

  python scripts/autosync.py                 # sync regionali + rerun
  python scripts/autosync.py --no-rerun      # solo ingestione
  python scripts/autosync.py --types regionali,comunali
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.pipeline.autosync import sync  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="regionali")
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="ingerisci solo le N più recenti")
    args = ap.parse_args()
    res = sync(types=tuple(args.types.split(",")), rerun=not args.no_rerun, limit=args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "rerun"},
                     ensure_ascii=False, indent=2, default=str))
    if res.get("rerun"):
        print("Rerun modello:", res["rerun"].get("run_id"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
