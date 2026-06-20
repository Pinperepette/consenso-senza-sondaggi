#!/usr/bin/env python3
"""Inizializza il sistema da zero (utile in Docker, su un DB vuoto):
schema + dati reali (open-data Ministero) + sondaggi + primo run del modello.

Richiede connessione a MongoDB (variabili CONSENSO_MONGO_URI / CONSENSO_DB) e
accesso a internet per scaricare i dati pubblici.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sh(*args: str, optional: bool = False) -> None:
    print("\n>> " + " ".join(args), flush=True)
    try:
        subprocess.run([sys.executable, *args], check=True, cwd=ROOT)
    except subprocess.CalledProcessError:
        if optional:
            print(f"(saltato: {args[0]} ha fallito, proseguo)", flush=True)
        else:
            raise


def main() -> int:
    sh("scripts/init_db.py")
    sh("scripts/load_opendata.py")                 # politiche 2022 + europee 2024
    sh("scripts/load_regionali_2025.py", optional=True)
    sh("scripts/load_polls.py")                    # sondaggi storici + live
    print("\n>> run_model (con sondaggi)…", flush=True)
    from consenso.pipeline.orchestrate import run_model
    r = run_model(include_regional=True, include_polls=True)
    print(f"OK — run {r['run_id']}, {r['n_estimations']} stime, "
          f"{r['n_parties']} partiti.", flush=True)
    print("Apri http://localhost:5057", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
