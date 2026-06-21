"""Scoperta AUTOMATICA di eventi politici datati, da fonti testuali REALI
(Wikipedia, notizie). L'AI estrae solo struttura (data, titolo, tipo) dal testo
fornito, con la fonte allegata: e' interpretazione lato input, da VERIFICARE, mai
numeri inventati. Gli eventi auto sono marcati origin='ai', verified=False e tenuti
distinti da quelli curati a mano.
"""
from __future__ import annotations

import re
from typing import List, Optional

PARTY_WIKI = {
    "LEGA": "Lega_(partito_politico_2020)",
    "FDI": "Fratelli_d'Italia",
    "FI": "Forza_Italia_(2013)",
    "PD": "Partito_Democratico_(Italia)",
    "M5S": "Movimento_5_Stelle",
    "AVS": "Alleanza_Verdi_e_Sinistra",
}

SYSTEM = (
    "Sei un analista politico italiano. Dal TESTO fornito (storia di un partito) estrai "
    "gli EVENTI piu rilevanti per capire ascesa/declino del consenso: ingressi/uscite "
    "dai governi, alleanze, scissioni, cambi di linea/giravolte, voti chiave, risultati "
    "elettorali. SOLO eventi con una DATA presente o ricavabile dal testo. NON inventare "
    "date o fatti: se la data manca, ometti l'evento. Rispondi SOLO JSON con questa forma: "
    "{\"events\":[{\"date\":\"YYYY-MM-DD\",\"kind\":\"alleanza|dietrofront|scissione|voto|picco|minimo\","
    "\"title\":\"breve\",\"note\":\"una frase\"}]} . Massimo 12 eventi, dal piu vecchio al piu recente."
)


def _clean_html(raw: str) -> str:
    t = re.sub(r"<(script|style|table)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[\d+\]", " ", t)            # note Wikipedia [12]
    return re.sub(r"\s+", " ", t).strip()


def discover_from_wikipedia(party: str) -> dict:
    """Scarica la voce Wikipedia del partito (API plaintext) e fa estrarre gli eventi all'AI."""
    import json as _json
    from consenso.ai.deepseek import available, chat_json
    if not available():
        return {"error": "AI non disponibile"}
    page = PARTY_WIKI.get(party.upper())
    if not page:
        return {"error": f"nessuna pagina nota per {party}"}
    from consenso.etl.base import http_get
    title = page.replace("_", " ")
    api = ("https://it.wikipedia.org/w/api.php?format=json&action=query&prop=extracts"
           "&explaintext=1&redirects=1&titles=" + title.replace(" ", "%20").replace("'", "%27"))
    url = f"https://it.wikipedia.org/wiki/{page}"
    try:
        data = _json.loads(http_get(api).decode("utf-8", "replace"))
        pages = data["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        text = re.sub(r"\s+", " ", text)[:14000]
        if not text:
            return {"error": "voce Wikipedia vuota"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"download fallito: {str(exc)[:80]}"}
    try:
        data = chat_json(SYSTEM, f"PARTITO: {party}\nTESTO:\n{text}")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"AI errore: {str(exc)[:80]}"}
    evs = data.get("events", []) if isinstance(data, dict) else []
    out = [{"date": e.get("date"), "kind": e.get("kind", "voto"),
            "title": e.get("title", ""), "note": e.get("note", ""),
            "source": url, "origin": "ai", "verified": False}
           for e in evs if e.get("date") and len(str(e.get("date"))) >= 10]
    return {"party": party.upper(), "events": out, "source": url}


def save_events(party: str, events: List[dict]) -> int:
    """Salva/aggiorna gli eventi auto nel DB (collection events_auto), idempotente per (party,date,title)."""
    from consenso.db.client import get_db
    db = get_db()
    n = 0
    for e in events:
        key = {"party": party.upper(), "date": e["date"], "title": e["title"]}
        db["events_auto"].update_one(key, {"$set": {**key, **e}}, upsert=True)
        n += 1
    return n


def discover_party(party: str) -> dict:
    r = discover_from_wikipedia(party)
    if "error" in r:
        return r
    saved = save_events(party, r["events"])
    return {"party": party.upper(), "found": len(r["events"]), "saved": saved}
