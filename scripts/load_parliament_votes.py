#!/usr/bin/env python3
"""Ingestione dei VOTI PARLAMENTARI REALI (Camera dei deputati, open data SPARQL).

Per ogni voto FINALE di una legislatura prende come ha votato ogni gruppo
(favorevole/contrario/astenuto), lo mappa sui partiti e ne deriva la posizione del
partito (maggioranza dei suoi deputati). Sono FATTI contati, non ricostruzioni AI:
base per la coerenza fatti/parole.

Salva in ``parliament_votes``.
"""
from __future__ import annotations

import html
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402

SPARQL = "https://dati.camera.it/sparql"
SIGLA2PARTY = {"FDI": "party:FDI", "PD-IDP": "party:PD", "LEGA": "party:LEGA",
               "M5S": "party:M5S", "FI-PPE": "party:FI", "AVS": "party:AVS"}
# sigle storiche per legislature precedenti (estendibile)
SIGLA2PARTY.update({"PD": "party:PD", "FI-PDL": "party:FI", "LEGA-SP": "party:LEGA",
                    "LEU": "party:AVS", "FDI-AN": "party:FDI"})


def _q(query: str):
    r = httpx.get(SPARQL, params={"query": query, "format": "json"}, timeout=120,
                  headers={"User-Agent": "Mozilla/5.0",
                           "Accept": "application/sparql-results+json"})
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def load_leg(leg: int = 19) -> int:
    leguri = f"http://dati.camera.it/ocd/legislatura.rdf/repubblica_{leg}"
    meta = _q(f"""PREFIX ocd:<http://dati.camera.it/ocd/>
      PREFIX dc:<http://purl.org/dc/elements/1.1/>
      SELECT ?vt ?d ?app (SAMPLE(?a) as ?atto) (SAMPLE(?bt) as ?billtitle) WHERE {{
        ?vt a ocd:votazione ; ocd:votazioneFinale 1 ; ocd:rif_leg <{leguri}> ;
            dc:date ?d ; ocd:approvato ?app .
        OPTIONAL {{ ?vt ocd:rif_attoCamera ?a . OPTIONAL {{ ?a dc:title ?bt }} }}
      }} GROUP BY ?vt ?d ?app""")
    tall = _q(f"""PREFIX ocd:<http://dati.camera.it/ocd/>
      SELECT ?vt ?sigla ?val (COUNT(*) as ?n) WHERE {{
        ?vt a ocd:votazione ; ocd:votazioneFinale 1 ; ocd:rif_leg <{leguri}> .
        ?v ocd:rif_votazione ?vt ; ocd:siglaGruppo ?sigla ;
           <http://purl.org/dc/elements/1.1/type> ?val .
        FILTER(?sigla IN ("FDI","PD-IDP","LEGA","M5S","FI-PPE","AVS","PD","FI-PDL","LEU"))
      }} GROUP BY ?vt ?sigla ?val""")

    # tallies[vt][party][valore] = n
    tallies = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for b in tall:
        party = SIGLA2PARTY.get(b["sigla"]["value"])
        if not party:
            continue
        val = b["val"]["value"].lower()           # favorevole/contrario/astensione...
        tallies[b["vt"]["value"]][party][val] += int(b["n"]["value"])

    db = get_db()
    db["parliament_votes"].delete_many({"leg": leg})
    docs = {}
    for b in meta:
        vt = b["vt"]["value"]
        by_party = {}
        for party, vals in tallies.get(vt, {}).items():
            fav = vals.get("favorevole", 0)
            contr = vals.get("contrario", 0)
            ast = sum(v for k, v in vals.items() if "asten" in k)
            stance = "favorevole" if fav >= max(contr, ast) and fav > 0 else (
                "contrario" if contr >= ast and contr > 0 else
                ("astenuto" if ast > 0 else "assente"))
            by_party[party] = {"stance": stance, "fav": fav, "contr": contr, "ast": ast}
        if not by_party:
            continue
        d = b["d"]["value"]
        title = html.unescape((b.get("billtitle") or {}).get("value", "")).strip()
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', title)).strip('"“ ') or "Voto finale"
        docs[vt.split("/")[-1]] = {
            "_id": vt.split("/")[-1], "leg": leg,
            "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d,
            "title": title, "bill": (b.get("atto") or {}).get("value", "").split("/")[-1],
            "approved": b["app"]["value"] == "1", "by_party": by_party}
    if docs:
        db["parliament_votes"].insert_many(list(docs.values()))
    db["parliament_votes"].create_index([("leg", 1), ("date", 1)])
    return len(docs)


def main() -> int:
    for leg in (19, 18):
        try:
            n = load_leg(leg)
            print(f"legislatura {leg}: {n} voti finali con dato per-partito")
        except Exception as exc:  # noqa: BLE001
            print(f"legislatura {leg}: errore {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
