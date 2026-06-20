"""Aggancio automatico dei dati: interroga l'archivio open-data del Ministero,
rileva le elezioni NON ancora presenti e le ingerisce da solo, poi rilancia il
modello.

Idempotente: ogni download è registrato in ``raw_ingestions`` (source =
"opendata:<path>"); le elezioni già viste vengono saltate.

Tipi gestiti in automatico: regionali (formato SCRUTINI corrente). Gli altri
(comunali/storici/referendum) vengono elencati come disponibili ma non ingeriti
automaticamente, perché richiedono parser dedicati o sono poco rilevanti.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List

from consenso.db.client import get_db
from consenso.db.schema import RAW
from consenso.etl.base import archive_raw, audit, http_get

BASE = "https://elezionistorico.interno.gov.it/daithome/documenti/opendata/"
CATALOG_URL = "https://elezionistorico.interno.gov.it/eligendo/opendata.php"
SYNC_STATUS = "sync_status"
SUPPORTED = ("regionali", "comunali")   # tipi ingeriti automaticamente
MODEL_TYPES = ("regionali",)            # tipi che fanno scattare il re-run del modello
DEFAULT_SINCE_YEAR = 2023               # evita di scaricare l'archivio storico


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def list_catalog() -> List[Dict]:
    """Estrae il catalogo (tipo, path, data) dalla pagina open-data."""
    html = http_get(CATALOG_URL).decode("utf-8", "replace")
    rows = re.findall(
        r'\[\s*"([^"]+)",\s*"(\d{4})",\s*"([^"]+\.(?:csv|zip))",\s*"([^"]+)"', html)
    out = []
    for _cat, yr, path, _fname in rows:
        prefix = path.split("/")[0]
        m = re.search(r'(\d{4})(\d{2})(\d{2})', path)
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else f"{yr}-01-01"
        out.append({"prefix": prefix, "path": path, "date": date, "year": yr})
    # dedup
    seen, uniq = set(), []
    for e in out:
        if e["path"] not in seen:
            seen.add(e["path"]); uniq.append(e)
    return uniq


def _done_paths() -> set:
    """Path già scaricati (registrati in raw_ingestions)."""
    out = set()
    for r in get_db()[RAW].find({"source": {"$regex": "^opendata:"}}, {"source": 1}):
        out.add(r["source"].split("opendata:", 1)[1])
    return out


def sync(types=SUPPORTED, rerun: bool = True, limit: int = 0,
         since_year: int = DEFAULT_SINCE_YEAR, polls: bool = True) -> Dict:
    """Scarica e ingerisce le elezioni nuove dei tipi indicati e, se ``polls``,
    aggiorna anche i sondaggi (statistiche, da Wikipedia live)."""
    from scripts.load_regionali_2025 import (ingest_comunali_zip,
                                             ingest_regionali_zip)

    handlers = {"regionali": ingest_regionali_zip, "comunali": ingest_comunali_zip}
    catalog = list_catalog()
    done = _done_paths()
    new, skipped, errors = [], [], []
    model_relevant = False
    candidates = [e for e in catalog if e["prefix"] in types and e["path"] not in done
                  and int(e["year"]) >= since_year]
    candidates.sort(key=lambda e: e["date"])
    if limit:
        candidates = candidates[-limit:]

    for e in candidates:
        try:
            raw = http_get(BASE + e["path"])
            archive_raw(f"opendata:{e['path']}", raw, {"date": e["date"]})
            res = handlers[e["prefix"]](raw, e["date"])
            ok = [r for r in res if "election_id" in r]
            if ok:
                new.append({"path": e["path"], "date": e["date"], "type": e["prefix"],
                            "elections": [r["election_id"] for r in ok]})
                if e["prefix"] in MODEL_TYPES:
                    model_relevant = True
            else:
                skipped.append({"path": e["path"],
                                "reason": res[0].get("skipped", "schema non gestito")})
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": e["path"], "error": str(exc)[:160]})

    # sondaggi (statistiche): refresh live da Wikipedia; se cambiano, re-stima
    poll_info, polls_changed = None, False
    if polls:
        try:
            from scripts.load_polls import load_all_polls
            pcol = get_db()["polls"]
            prev_n = pcol.count_documents({})
            prev_max = (pcol.find_one(sort=[("date", -1)]) or {}).get("date")
            poll_info = load_all_polls()
            polls_changed = (poll_info["rows"] != prev_n
                             or poll_info["max_date"] != prev_max)
            if polls_changed:
                model_relevant = True
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": "polls", "error": str(exc)[:160]})

    # economia (misery index ISTAT/World Bank) per il "costo del governare"
    econ_years = None
    try:
        from consenso.etl.sources.economy import fetch_misery
        econ_years = fetch_misery()
    except Exception as exc:  # noqa: BLE001
        errors.append({"path": "economy", "error": str(exc)[:160]})

    rerun_info = None
    if model_relevant and rerun:
        from consenso.pipeline.orchestrate import run_model
        rerun_info = run_model(include_regional=True, include_polls=polls)

    # dimensioni: genera il posizionamento dei partiti nuovi (se c'e' la chiave AI)
    dims_added = 0
    try:
        from consenso.ai.deepseek import available
        if available():
            from consenso.model.dimensions import PARTY_NAMES, generate_all
            have = set(get_db()["party_dimensions"].distinct("party_id"))
            missing = [p for p in PARTY_NAMES if p not in have]
            if missing:
                dims_added = generate_all(missing)
    except Exception as exc:  # noqa: BLE001
        errors.append({"path": "dimensions", "error": str(exc)[:160]})

    status = {"_id": "last", "ts": _utcnow(), "new": new, "n_new": len(new),
              "skipped": len(skipped), "errors": errors, "rerun": rerun_info,
              "polls": poll_info, "polls_changed": polls_changed,
              "econ_years": econ_years, "dims_added": dims_added}
    get_db()[SYNC_STATUS].replace_one({"_id": "last"}, status, upsert=True)
    audit("autosync", "sync", {"n_new": len(new), "errors": len(errors)})
    return {"n_new": len(new), "new": new, "skipped": len(skipped),
            "errors": errors, "rerun": rerun_info,
            "polls": poll_info, "polls_changed": polls_changed,
            "econ_years": econ_years, "dims_added": dims_added}


def last_status() -> Dict:
    s = get_db()[SYNC_STATUS].find_one({"_id": "last"})
    if not s:
        return {"ts": None, "n_new": 0}
    s.pop("_id", None)
    if s.get("ts"):
        s["ts"] = s["ts"].isoformat()
    return s
