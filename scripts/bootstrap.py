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
    sh("scripts/load_comunali_catalogo.py", optional=True)   # scheda Comunali (recenti)
    sh("scripts/load_comunali_storico.py", optional=True)    # comunali storiche 2010-2022
    sh("scripts/load_parliament_votes.py", optional=True)    # voti reali Camera (fatti vs parole)
    sh("scripts/load_polls.py")                    # sondaggi storici + live
    # fondamentali: economia (World Bank) per il costo del governare
    try:
        from consenso.etl.sources.economy import fetch_misery
        print(f"\n>> economia: {fetch_misery()} anni di misery index", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(economia saltata: {exc})", flush=True)
    print("\n>> run_model (con sondaggi + trend + fondamentali)…", flush=True)
    from consenso.pipeline.orchestrate import run_model
    r = run_model(include_regional=True, include_polls=True)
    print(f"OK — run {r['run_id']}, {r['n_estimations']} stime, "
          f"{r['n_parties']} partiti.", flush=True)
    sh("scripts/build_spatial_summary.py", optional=True)   # scheda Partiti (identikit)
    sh("scripts/calibrate.py", optional=True)               # auto-calibra phi/kappa sul backtest
    # flussi elettorali (scheda Flussi): inferenza ecologica 2022 -> 2024
    try:
        from consenso.model.flow_pipeline import run_flow_model
        # hierarchical=False: matrice nazionale aggregata (veloce); quella per-comune
        # e' troppo pesante per il bootstrap.
        run_flow_model("elez:2022_politiche", "elez:2024_europee", hierarchical=False)
        print(">> flussi 2022->2024 calcolati", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(flussi saltati: {exc})", flush=True)
    # dimensioni (posizionamento): solo se c'e' la chiave AI
    try:
        from consenso.ai.deepseek import available
        if available():
            from consenso.model.dimensions import PARTY_NAMES, generate_all
            print(f">> dimensioni: {generate_all(list(PARTY_NAMES))} partiti", flush=True)
        else:
            print(">> dimensioni: saltate (nessuna chiave AI)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(dimensioni saltate: {exc})", flush=True)
    print("Apri http://localhost:5057", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
