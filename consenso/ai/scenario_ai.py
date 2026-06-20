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
