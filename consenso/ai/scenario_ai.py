"""Genera uno 'scenario spec' a partire da articoli, usando DeepSeek.

REGOLA FONDAMENTALE (nel prompt): l'AI NON inventa le percentuali finali.
Produce solo ASSUNZIONI esplicite (shock in punti, con range e fonte citata) che
il motore applicherà. Output forzato in JSON.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from consenso.ai.deepseek import chat_json
from consenso.model.nowcast import projected_shares, summarize_shares

PARTY_IDS = ["party:FDI", "party:PD", "party:M5S", "party:LEGA", "party:FI",
             "party:AVS", "party:ALTRI"]

SYSTEM = """Sei un analista elettorale prudente. Ti vengono dati: (1) la stima
attuale del consenso dei partiti italiani, (2) uno o più articoli di cronaca.

Il tuo compito NON è dire le nuove percentuali. Il tuo compito è tradurre gli
articoli in ASSUNZIONI esplicite e motivate, che un modello statistico applicherà.

Regole tassative:
- Ogni assunzione deve essere uno SPOSTAMENTO in punti percentuali rispetto al
  baseline (es. -2.0), con un intervallo plausibile (low/high) e una CITAZIONE
  testuale dall'articolo che la giustifica.
- Sii PRUDENTE: uno scandalo o una notizia sposta raramente piu' di 3-5 punti.
  Se l'articolo non giustifica uno spostamento, NON inventarlo.
- Usa SOLO questi identificativi di partito: party:FDI, party:PD, party:M5S,
  party:LEGA, party:FI, party:AVS, party:ALTRI.
- Per un partito NUOVO (mai votato) usa 'new_party' con name e draws_from
  (da quali partiti/astensione pesca, frazioni che sommano ~1).
- Se non hai elementi, restituisci liste vuote. Non inventare.

Rispondi SOLO con JSON valido in questo schema:
{
 "summary": "sintesi in 1-2 frasi di cosa dicono gli articoli",
 "deltas": [
   {"party":"party:FDI","mean":-2.0,"low":-4.0,"high":-0.5,
    "confidence":"bassa|media|alta","rationale":"perche'","source_quote":"frase dall'articolo"}
 ],
 "new_party": null
}
oppure new_party:
 {"name":"Futuro Nazionale","share_mean":4.0,"share_low":2.0,"share_high":6.0,
  "draws_from":{"party:LEGA":0.4,"party:FDI":0.3,"astensione":0.3},
  "confidence":"bassa","rationale":"...","source_quote":"..."}
"""


TRIAGE_SYSTEM = """Sei un analista politico italiano. Ti do un elenco numerato di
titoli di cronaca politica recente. Seleziona SOLO quelli che potrebbero
realisticamente spostare il consenso dei partiti: scandali/inchieste, alleanze e
rotture, leggi divisive, cambi di leadership, nascita di nuovi partiti, exit/risultati.
Scarta cronaca neutra, di colore, o estera irrilevante per il voto italiano.

Rispondi SOLO con JSON: {"selected":[{"i":<indice>,"why":"motivo breve"}]}.
Massimo 8 voci, ordinate per impatto potenziale. Se nulla e' rilevante, lista vuota."""


def generate_spec_from_news(as_of: Optional[str] = None,
                            max_items: int = 25) -> Dict:
    """Agente: legge i feed RSS, fa triage delle notizie rilevanti, poi genera lo
    scenario spec dalle sole notizie selezionate. Restituisce anche news_used."""
    from consenso.ai.news import fetch_political_news

    news = fetch_political_news(max_items)
    if not news:
        return {"error": "nessuna notizia dai feed RSS"}
    listing = "\n".join(
        f"[{i}] ({n['source']}) {n['title']} — {n['summary'][:160]}"
        for i, n in enumerate(news))
    try:
        sel = chat_json(TRIAGE_SYSTEM, "TITOLI:\n" + listing)
    except Exception:  # noqa: BLE001
        sel = {}
    idxs = [s["i"] for s in sel.get("selected", [])
            if isinstance(s.get("i"), int) and 0 <= s["i"] < len(news)]
    why = {s["i"]: s.get("why", "") for s in sel.get("selected", [])
           if isinstance(s.get("i"), int)}
    chosen = []
    for i in (idxs or list(range(min(5, len(news))))):
        item = dict(news[i]); item["why"] = why.get(i, ""); chosen.append(item)
    articles = "\n\n".join(f"({n['source']}) {n['title']}. {n['summary']}"
                           for n in chosen)
    gen = generate_spec(articles, as_of)
    if "error" not in gen:
        gen["news_used"] = chosen
    return gen


def generate_spec(articles: str, as_of: Optional[str] = None) -> Dict:
    parties, base, meta = projected_shares(as_of)
    if parties is None:
        return {"error": meta.get("error", "nessun run")}
    baseline = summarize_shares(parties, base)
    ctx = "\n".join(f"- {p['name']} ({p['party_id']}): {p['mean']*100:.1f}%"
                    for p in baseline)
    user = (f"STIMA ATTUALE (baseline, al {meta['as_of']}):\n{ctx}\n\n"
            f"ARTICOLI:\n\"\"\"\n{articles.strip()[:12000]}\n\"\"\"\n\n"
            "Genera lo scenario spec JSON secondo le regole.")
    spec = chat_json(SYSTEM, user)
    # sanitizza: tieni solo party id validi
    spec["deltas"] = [d for d in spec.get("deltas", [])
                      if d.get("party") in PARTY_IDS]
    return {"spec": spec, "baseline": baseline, "meta": meta}
