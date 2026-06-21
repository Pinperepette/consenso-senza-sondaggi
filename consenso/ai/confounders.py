"""Strato AI (lato INPUT) per la correzione coi voti reali: segnala i CONFONDENTI
per cui il trasferimento locale->nazionale potrebbe non valere.

NON produce numeri ne' stime: e' interpretazione tracciabile sull'input, coerente
con la filosofia del progetto (l'AI propone ipotesi, non genera output). Degrada in
modo pulito se non c'e' la chiave AI.
"""
from __future__ import annotations

import hashlib

SYSTEM = (
    "Sei un analista elettorale italiano scettico. Ti do una correzione che alza o "
    "abbassa la stima di alcuni partiti perche' i VOTI REALI LOCALI recenti divergono "
    "dai sondaggi. Elenca i CONFONDENTI concreti per cui il trasferimento "
    "locale->nazionale potrebbe NON valere (es: un nuovo partito che assorbe voti solo "
    "localmente, effetto-candidato/sindaco, liste civiche, astensione differenziale, "
    "il voto amministrativo che premia persone piu' che simboli). Sii specifico sul "
    "contesto italiano attuale. NON produrre numeri, NON dare nuove stime: solo motivi "
    "di cautela. Rispondi SOLO JSON: "
    '{"confounders":[{"factor":"...","affects":"LEGA|FDI|...","note":"...",'
    '"weakens":true}],"verdict":"una frase di sintesi"}'
)


def _key(ups, downs, level) -> str:
    return hashlib.sha1(f"{sorted(ups)}|{sorted(downs)}|{level}".encode()).hexdigest()[:16]


def confounders(fc: dict, *, use_cache: bool = True) -> dict:
    from consenso.ai.deepseek import available, chat_json
    if not available():
        return {"error": "AI non disponibile (nessuna chiave)"}
    ups = [p["party"] for p in fc["parties"] if (p.get("adjustment") or 0) > 0.1]
    downs = [p["party"] for p in fc["parties"] if (p.get("adjustment") or 0) < -0.1]
    if not ups and not downs:
        return {"confounders": [], "verdict": "Nessuna correzione rilevante da valutare."}
    lvl = fc.get("validity", {}).get("level", "?")
    k = _key(ups, downs, lvl)

    from consenso.db.client import get_db
    cache = get_db()["ai_cache"]
    if use_cache:
        hit = cache.find_one({"_id": f"confounders:{k}"})
        if hit:
            return hit["data"]

    user = (f"Correzione coi voti reali: SU {ups or 'nessuno'}, GIU {downs or 'nessuno'}. "
            f"Validita dai controlli: {lvl}. {fc.get('validity', {}).get('note', '')} "
            "Contesto: e' nato il partito FN di Vannacci, uscito dalla Lega (puo' "
            "assorbire voti leghisti). Elenca i confondenti che indeboliscono il "
            "trasferimento locale->nazionale.")
    try:
        data = chat_json(SYSTEM, user)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"AI errore: {str(exc)[:80]}"}
    if not isinstance(data, dict) or "confounders" not in data:
        data = {"confounders": [], "verdict": str(data)[:200]}
    cache.update_one({"_id": f"confounders:{k}"}, {"$set": {"data": data}}, upsert=True)
    return data
