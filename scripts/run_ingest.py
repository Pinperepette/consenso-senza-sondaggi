#!/usr/bin/env python3
"""Ingestione di un'elezione: registra l'elezione, carica risultati e affluenza,
calcola le feature e valida.

Esempi:
  # da URL Eligendo
  python scripts/run_ingest.py --election-id elez:2022_politiche_camera \\
      --type politiche --date 2022-09-25 --geo-level comune \\
      --results-url https://.../scrutiniCI.csv --turnout-url https://.../affluenza.csv

  # da file locale (offline / fixture)
  python scripts/run_ingest.py --election-id elez:test --type europee --date 2024-06-09 \\
      --geo-level comune --results-file results.csv --turnout-file turnout.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.etl.sources.eligendo import EligendoResultsLoader, register_election  # noqa: E402
from consenso.etl.sources.interno import TurnoutLoader  # noqa: E402
from consenso.pipeline.orchestrate import post_ingest  # noqa: E402


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingestione di un'elezione")
    ap.add_argument("--election-id", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--chamber", default=None)
    ap.add_argument("--geo-level", default="comune")
    ap.add_argument("--scope-level", default="nazionale")
    ap.add_argument("--results-url", default=None)
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--turnout-url", default=None)
    ap.add_argument("--turnout-file", default=None)
    ap.add_argument("--delimiter", default=";")
    ap.add_argument("--encoding", default="latin-1")
    args = ap.parse_args()

    register_election(args.election_id, args.type, args.date,
                      scope={"level": args.scope_level}, chamber=args.chamber)

    rl = EligendoResultsLoader()
    kw = dict(election_id=args.election_id, geo_level=args.geo_level,
              delimiter=args.delimiter, encoding=args.encoding)
    if args.results_file:
        r = rl.run(content=_read(args.results_file), **kw)
    elif args.results_url:
        r = rl.run(url=args.results_url, **kw)
    else:
        ap.error("specificare --results-url o --results-file")
    print("Risultati:", json.dumps(r, ensure_ascii=False))

    if args.turnout_file or args.turnout_url:
        tl = TurnoutLoader()
        tkw = dict(election_id=args.election_id, geo_level=args.geo_level,
                   delimiter=args.delimiter, encoding=args.encoding)
        t = (tl.run(content=_read(args.turnout_file), **tkw) if args.turnout_file
             else tl.run(url=args.turnout_url, **tkw))
        print("Affluenza:", json.dumps(t, ensure_ascii=False))

    report = post_ingest(args.election_id)
    print("Validazione:", json.dumps(report["validation"], ensure_ascii=False))
    if report["n_errors"]:
        print(f"ATTENZIONE: {report['n_errors']} anomalie in quarantine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
