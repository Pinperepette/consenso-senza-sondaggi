"""Inserimento MANUALE di dati che non abbiamo (sondaggi, risultati elettorali)
e ESTRAZIONE assistita dall'AI a partire da un link.

I dati manuali sono marcati (_manual / source=manuale) cosi' restano distinguibili
dai dati ufficiali e si possono rimuovere. L'AI estrae solo struttura da testo
fornito: e' interpretazione tracciabile lato input, va sempre rivista prima di salvare.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from consenso.db.client import get_db
from consenso.db.schema import PARTY_RESULTS

# riusa la mappa regione->codice del loader storico
from scripts.load_comunali_storico import REG2COD

PARTY_ALIASES = {
    "FDI": "FDI", "FRATELLI D'ITALIA": "FDI", "FRATELLI DITALIA": "FDI",
    "PD": "PD", "PARTITO DEMOCRATICO": "PD",
    "M5S": "M5S", "MOVIMENTO 5 STELLE": "M5S", "5 STELLE": "M5S",
    "LEGA": "LEGA", "FI": "FI", "FORZA ITALIA": "FI",
    "AVS": "AVS", "ALLEANZA VERDI SINISTRA": "AVS", "VERDI": "AVS",
    "FN": "FN", "AZIONE": "AZIONE", "IV": "IV", "ITALIA VIVA": "IV",
}


def _norm_party(k: str) -> str:
    u = re.sub(r"[^A-Z0-9' ]", "", (k or "").upper()).strip()
    return PARTY_ALIASES.get(u, u)


def add_poll(date: str, pollster: str, shares_pct: Dict[str, float]) -> dict:
    """shares_pct: {'FDI': 28.3, ...} in punti percentuali."""
    db = get_db()
    docs = [{"pollster": pollster or "manuale", "date": date[:10],
             "party_id": "party:" + _norm_party(k), "share": float(v) / 100.0,
             "_manual": True}
            for k, v in shares_pct.items() if v not in (None, "")]
    if not docs:
        return {"error": "nessuna quota valida"}
    db["polls"].insert_many(docs)
    return {"inserted": len(docs), "date": date[:10], "pollster": pollster or "manuale"}


def add_result(date: str, etype: str, shares_pct: Dict[str, float], *,
               region: Optional[str] = None, comune: Optional[str] = None,
               eid: Optional[str] = None) -> dict:
    """Risultato elettorale manuale (es. una tornata non ancora nell'open-data).
    region: nome o codice; comune: nome (opzionale). shares in %."""
    db = get_db()
    reg = REG2COD.get((region or "").strip().upper(), (region or "").strip())
    if comune:
        geo = f"COM:{reg}:{comune.strip().upper()}"
        level = "comune"
    elif reg:
        geo = f"REG:{reg}"
        level = "regione"
    else:
        geo = "ITALIA"
        level = "nazione"
    eid = eid or f"elez:{date[:10]}_{etype}_manual"
    from consenso.etl.sources.eligendo import register_election
    register_election(eid, etype, date[:10], {"source": "manuale", "level": level})
    db[PARTY_RESULTS].delete_many({"election_id": eid, "geo_id": geo})
    docs = [{"election_id": eid, "geo_id": geo, "geo_level": level,
             "party_id": "party:" + _norm_party(k), "raw_label": None,
             "share": float(v) / 100.0, "votes": int(float(v) * 1000),
             "valid_votes_area": 100000, "_meta": {"source": "manuale"}}
            for k, v in shares_pct.items() if v not in (None, "")]
    if docs:
        db[PARTY_RESULTS].insert_many(docs)
    return {"election_id": eid, "geo_id": geo, "inserted": len(docs)}


def remove_manual(kind: str) -> dict:
    """Rimuove i dati manuali ('polls' o 'results')."""
    db = get_db()
    if kind == "polls":
        n = db["polls"].delete_many({"_manual": True}).deleted_count
        return {"removed_polls": n}
    n = db[PARTY_RESULTS].delete_many({"_meta.source": "manuale"}).deleted_count
    db["elections"].delete_many({"_id": {"$regex": "_manual$"}})
    return {"removed_results": n}


_EXTRACT = {
    "poll": ("Estrai da questo testo i dati di UN sondaggio elettorale italiano. "
             "Rispondi SOLO JSON: {\"date\":\"YYYY-MM-DD\",\"pollster\":\"...\","
             "\"shares\":{\"FDI\":28.3,\"PD\":21.0,...}} (quote in punti %). "
             "Se un dato manca, ometti la chiave. Non inventare numeri."),
    "result": ("Estrai da questo testo UN risultato elettorale italiano. Rispondi SOLO "
               "JSON: {\"date\":\"YYYY-MM-DD\",\"type\":\"comunali|regionali|politiche\","
               "\"region\":\"...\",\"comune\":\"...\",\"shares\":{\"FDI\":..,...}} "
               "(quote in %). Ometti le chiavi mancanti. Non inventare."),
}


def extract_from_url(url: str, kind: str = "poll") -> dict:
    """Scarica la pagina e usa l'AI per estrarre struttura (da rivedere prima di salvare)."""
    from consenso.ai.deepseek import available, chat_json
    if not available():
        return {"error": "AI non disponibile (nessuna chiave)"}
    if kind not in _EXTRACT:
        return {"error": "tipo non valido"}
    from consenso.etl.base import http_get
    try:
        raw = http_get(url).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"download fallito: {str(exc)[:80]}"}
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)[:9000]
    try:
        data = chat_json(_EXTRACT[kind], text)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"AI errore: {str(exc)[:80]}"}
    data["_source_url"] = url
    return data
