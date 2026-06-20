#!/usr/bin/env python3
"""Inizializza il database: crea collection e indici (idempotente).

    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.schema import ALL_COLLECTIONS, ensure_collections  # noqa: E402


def main() -> int:
    created = ensure_collections()
    print(f"Collection garantite: {len(ALL_COLLECTIONS)}")
    if created:
        print("Create ora:", ", ".join(created))
    else:
        print("Tutte già presenti; indici verificati.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
