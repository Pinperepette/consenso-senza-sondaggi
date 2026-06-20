"""Loader dei risultati elettorali (Eligendo / archivio storico Min. Interno).

Eligendo pubblica dataset scaricabili (CSV) con i risultati per livello
territoriale; l'archivio storico (elezionistorico.interno.gov.it) copre le
tornate passate. I formati variano fra tornate, quindi il parser è guidato da
una *mappa di colonne* configurabile, con default ragionevoli, e lavora in
formato "long" (una riga per area × lista).

Uso tipico:
    EligendoResultsLoader().run(
        url=".../scrutiniCI.csv", election_id="elez:2022_politiche_camera",
        geo_level="comune")
oppure, offline/test, passando ``content=<bytes>``.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional

from consenso.db.client import get_db
from consenso.db.schema import ELECTIONS, PARTY_RESULTS
from consenso.etl.base import BaseLoader, http_get
from consenso.etl.geo import comune_id, province_id, region_id, NATION_ID
from consenso.etl.reconcile import enqueue_unknown, resolve_label

# default: nomi colonna più comuni nei dataset Eligendo (tollerante a varianti)
DEFAULT_COLMAP = {
    "geo_code": ["CODICE_COMUNE", "COD_COMUNE", "CODCOMUNE", "PROCOM", "CODICE", "COD_ISTAT"],
    "list_label": ["LISTA", "DESCR_LISTA", "DENOMINAZIONE_LISTA", "DESCRLISTA", "NOME_LISTA"],
    "votes": ["VOTI_LISTA", "VOTILISTA", "VOTI", "VOTI_VALIDI_LISTA"],
    "valid_votes": ["VOTI_VALIDI", "VOTIVALIDI", "TOTALE_VOTI_VALIDI", "VOTANTI_VALIDI"],
}

LEVEL_ID_FN = {
    "comune": comune_id,
    "provincia": province_id,
    "regione": region_id,
}


def _pick(row: Dict[str, str], keys: List[str]) -> Optional[str]:
    norm = {k.strip().upper(): v for k, v in row.items() if k}
    for k in keys:
        v = norm.get(k.strip().upper())
        if v not in (None, ""):
            return v.strip()
    return None


def _geo_id(level: str, code: str) -> str:
    if level == "nazione":
        return NATION_ID
    fn = LEVEL_ID_FN.get(level)
    return fn(code) if fn else f"ISTAT:{code}"


class EligendoResultsLoader(BaseLoader):
    source_name = "eligendo_results"

    def fetch(self, url: str, **kwargs) -> bytes:  # pragma: no cover - rete
        return http_get(url)

    def parse(self, content: bytes, raw_id: str, *, election_id: str,
              geo_level: str = "comune", colmap: Optional[dict] = None,
              delimiter: str = ";", encoding: str = "latin-1", **kwargs) -> dict:
        cmap = {**DEFAULT_COLMAP, **(colmap or {})}
        elec = get_db()[ELECTIONS].find_one({"_id": election_id}, {"date": 1})
        on_date = elec["date"] if elec else None

        text = content.decode(encoding, errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

        docs: List[dict] = []
        n_unmatched = 0
        for row in reader:
            code = _pick(row, cmap["geo_code"])
            label = _pick(row, cmap["list_label"])
            votes = _pick(row, cmap["votes"])
            valid = _pick(row, cmap["valid_votes"])
            if not (code and label and votes is not None):
                continue
            try:
                votes_i = int(float(votes.replace(".", "").replace(",", ".")))
                valid_i = int(float(valid.replace(".", "").replace(",", "."))) if valid else 0
            except ValueError:
                continue
            party_id = resolve_label(label, on_date)
            if party_id is None:
                enqueue_unknown(label, {"election_id": election_id})
                n_unmatched += 1
            docs.append({
                "election_id": election_id,
                "geo_id": _geo_id(geo_level, code),
                "geo_level": geo_level,
                "party_id": party_id,
                "raw_label": label,
                "votes": votes_i,
                "valid_votes_area": valid_i,
                "share": None,
                "_meta": {"source": self.source_name, "raw_ref": raw_id},
            })

        if docs:
            # rimpiazza i risultati di questa elezione per quel livello (idempotenza)
            get_db()[PARTY_RESULTS].delete_many(
                {"election_id": election_id, "geo_level": geo_level})
            get_db()[PARTY_RESULTS].insert_many(docs)
        return {"rows": len(docs), "unmatched_labels": n_unmatched,
                "election_id": election_id, "geo_level": geo_level}


def register_election(election_id: str, etype: str, date: str, scope: dict,
                      chamber: Optional[str] = None, electoral_system: Optional[str] = None) -> None:
    get_db()[ELECTIONS].update_one(
        {"_id": election_id},
        {"$set": {"type": etype, "date": date, "scope": scope, "chamber": chamber,
                  "electoral_system": electoral_system}},
        upsert=True,
    )
