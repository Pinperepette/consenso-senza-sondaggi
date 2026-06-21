"""Coerenza tra fatti e parole: confronta cosa un partito DICHIARA con come ha
VOTATO in Parlamento, su alcuni temi, e ne segue l'andamento per legislatura.

ONESTA': i 'fatti' sono i voti parlamentari. Qui, per un MVP, vengono RICOSTRUITI
dall'AI citando legge + anno + come ha votato, con la confidenza dichiarata: e'
un'interpretazione da verificare sulle fonti ufficiali (Camera/Senato), NON un dato
certificato. Layer separato dal consenso. La versione rigorosa userebbe il DB voti
open data del Parlamento.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from consenso.ai.deepseek import chat_json
from consenso.db.client import get_db
from consenso.model.dimensions import PARTY_NAMES
from consenso.model.fundamentals import GOV_TIMELINE

ISSUES = ["immigrazione", "ambiente e clima", "diritti civili",
          "Unione Europea", "fisco e tasse", "giustizia"]
PERIODS = ["2013-2018", "2018-2022", "2022-oggi"]

_SYS = (
    "Sei un analista politico SEVERO e onesto. Valuti la COERENZA tra ciò che un "
    "partito italiano DICHIARA e cosa ha fatto DAVVERO. Principio chiave: la "
    "coerenza si dimostra QUANDO SI E' AL GOVERNO e si puo' attuare; in opposizione "
    "votare e' gratis, quindi NON e' una prova di coerenza.\n"
    "Regole tassative:\n"
    "- Valuta un punteggio (0-10) SOLO per i periodi/temi in cui il partito era al "
    "GOVERNO (te li indico). Se in un periodo era all'opposizione o non c'era, "
    "coherence = null (non valutabile sui fatti).\n"
    "- Sii SEVERO: pesano molto le PROMESSE NON MANTENUTE da chi governava (es. "
    "taglio delle tasse o delle accise promesso e non fatto) e i CAMBI DI POSIZIONE "
    "(es. uscita dall'euro poi abbandonata, posizioni sui vaccini ribaltate). "
    "Questi abbassano nettamente la coerenza.\n"
    "- CITA fatti concreti: misura/legge o promessa, anno, esito (mantenuta/"
    "non mantenuta/ribaltata). Cita SOLO cio' di cui sei sicuro; NON inventare.\n"
    "- Se il partito NON e' MAI stato al governo o non hai elementi affidabili: "
    "overall = null, tutti i periodi null, e spiegalo nel summary.\n"
    "Rispondi SOLO JSON:\n"
    '{"overall": 0-10|null, "summary": "1 frase",'
    ' "by_issue": [{"issue":"...","stated":"...","action":"...","coherence":0-10|null,'
    '"citations":[{"law":"...","year":2018,"vote":"mantenuta|non mantenuta|ribaltata"}],'
    '"confidence":"alta|media|bassa"}],'
    ' "by_period": [{"period":"2013-2018","coherence":0-10|null}]}')


def _gov_periods(party_id: str) -> str:
    per = GOV_TIMELINE.get(party_id, [])
    if not per:
        return "MAI al governo (sempre opposizione) -> coerenza non valutabile sui fatti"
    return ", ".join(f"{a[:4]}-{'oggi' if b > '2026' else b[:4]}" for a, b in per)


def generate_coherence(party_id: str) -> Dict:
    name = PARTY_NAMES.get(party_id, party_id.replace("party:", ""))
    user = (f"PARTITO: {name}\nTEMI: {', '.join(ISSUES)}\n"
            f"LEGISLATURE: {', '.join(PERIODS)}\n"
            f"PERIODI AL GOVERNO di questo partito: {_gov_periods(party_id)}\n\n"
            "Valuta la coerenza SOLO sui periodi di governo, severo su promesse non "
            "mantenute e cambi di posizione, citando i fatti.")
    out = chat_json(_SYS, user)
    out["party_id"] = party_id
    out["name"] = name.split(" (")[0]
    out["source"] = "ai"
    return out


def generate_all(party_ids: Optional[List[str]] = None) -> int:
    db = get_db()
    ids = party_ids or list(PARTY_NAMES)
    n = 0
    for pid in ids:
        try:
            doc = generate_coherence(pid)
        except Exception:  # noqa: BLE001
            continue
        db["coherence"].replace_one({"party_id": pid}, {**doc, "_id": pid}, upsert=True)
        n += 1
    return n
