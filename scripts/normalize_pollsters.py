#!/usr/bin/env python3
"""Normalizza i nomi degli istituti di sondaggio: lo scraping di Wikipedia ha
salvato lo stesso istituto con varianti ('AnalisiPolitica' vs 'Analisi Politica',
suffissi 'Archived ... Wayback Machine', '– Winpoll', camelCase). Trattati come
istituti distinti falsano gli house effect. Qui li unifico."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402


def normalize(name: str) -> str:
    p = re.sub(r"\s*Archived.*Wayback Machine", "", name, flags=re.I)
    p = re.sub(r"\s*[–-]\s*Winpoll", "", p)
    p = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)   # camelCase -> spazi
    return re.sub(r"\s+", " ", p).strip()


def main() -> int:
    db = get_db()
    fixed = 0
    for name in db["polls"].distinct("pollster"):
        norm = normalize(name)
        if norm and norm != name:
            r = db["polls"].update_many({"pollster": name}, {"$set": {"pollster": norm}})
            fixed += r.modified_count
            print(f"  '{name}' -> '{norm}' ({r.modified_count})")
    print(f"TOTALE record aggiornati: {fixed} | istituti ora: {len(db['polls'].distinct('pollster'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
