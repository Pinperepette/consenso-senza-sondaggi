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

ISSUES = ["immigrazione", "ambiente e clima", "diritti civili",
          "Unione Europea", "fisco e tasse", "giustizia"]
PERIODS = ["2013-2018", "2018-2022", "2022-oggi"]

_SYS = (
    "Sei un analista politico rigoroso. Valuti la COERENZA tra ciò che un partito "
    "italiano DICHIARA e come ha effettivamente VOTATO in Parlamento. "
    "Regole tassative:\n"
    "- Per ogni tema dai: posizione dichiarata (1 frase), comportamento legislativo "
    "(1 frase) e un punteggio di coerenza INTERO 0-10 (10 = fatti pienamente "
    "allineati alle parole).\n"
    "- CITA voti concreti: nome/oggetto della legge o misura, anno, e come ha votato "
    "(a favore/contro/astenuto). Cita SOLO ciò di cui sei ragionevolmente sicuro.\n"
    "- Dichiara la confidenza (alta|media|bassa). Se un partito non era in Parlamento "
    "o non hai elementi, metti coherence null e confidence 'bassa'. NON inventare leggi.\n"
    "- Dai anche un andamento per legislatura (coherence 0-10 per periodo) dove hai "
    "elementi, altrimenti null.\n"
    "Rispondi SOLO JSON in questo schema:\n"
    '{"overall": 0-10, "summary": "1 frase",'
    ' "by_issue": [{"issue":"...","stated":"...","action":"...","coherence":0-10,'
    '"citations":[{"law":"...","year":2018,"vote":"a favore|contro|astenuto"}],'
    '"confidence":"alta|media|bassa"}],'
    ' "by_period": [{"period":"2013-2018","coherence":0-10}]}')


def generate_coherence(party_id: str) -> Dict:
    name = PARTY_NAMES.get(party_id, party_id.replace("party:", ""))
    user = (f"PARTITO: {name}\nTEMI: {', '.join(ISSUES)}\n"
            f"LEGISLATURE: {', '.join(PERIODS)}\n\n"
            "Valuta la coerenza fatti/parole secondo le regole, citando i voti.")
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
