#!/usr/bin/env python3
"""Scopre automaticamente eventi politici datati per tutti i partiti (AI su
Wikipedia) e li salva in events_auto (marcati 'da verificare'). Pensato per girare
schedulato (es. cron notturno) cosi' la timeline 'Fatti e consenso' si aggiorna da
sola. Richiede la chiave AI; senza, esce senza errori."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.ai.deepseek import available  # noqa: E402
from consenso.ai.events import PARTY_WIKI, discover_party  # noqa: E402


def main() -> int:
    if not available():
        print("AI non disponibile: salto la scoperta eventi.")
        return 0
    only = [a.upper() for a in sys.argv[1:]] or list(PARTY_WIKI)
    for p in only:
        try:
            r = discover_party(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  {p}: errore {str(exc)[:80]}"); continue
        if "error" in r:
            print(f"  {p}: {r['error']}")
        else:
            print(f"  {p}: {r['found']} eventi trovati, {r['saved']} salvati (da verificare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
