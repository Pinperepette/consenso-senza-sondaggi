"""Gestione dell'albero territoriale e del remap storico dei comuni.

Le fonti:
  - ISTAT pubblica l'"Elenco dei comuni italiani" (CSV) con i codici di comune,
    provincia e regione: da qui costruiamo l'albero ``geographies``.
  - I comuni nascono, cessano e si fondono: ``geo_remap`` tiene il mapping fra
    codice storico e codice attuale, indispensabile per confrontare elezioni a
    distanza di anni.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, Iterable, List, Optional

from consenso.db.client import get_db
from consenso.db.schema import GEOGRAPHIES, GEO_REMAP
from consenso.etl.base import BaseLoader

NATION_ID = "ISTAT:IT"


def upsert_geography(doc: dict) -> None:
    get_db()[GEOGRAPHIES].update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)


def ensure_nation() -> None:
    upsert_geography({
        "_id": NATION_ID, "level": "nazione", "name": "Italia",
        "parent": None, "region": None, "istat_code": "IT",
    })


def region_id(code: str) -> str:
    return f"ISTAT:R{int(code):02d}"


def province_id(code: str) -> str:
    return f"ISTAT:P{str(code).zfill(3)}"


def comune_id(code: str) -> str:
    return f"ISTAT:{str(code).zfill(6)}"


def resolve_current_code(istat_code: str, year: int) -> str:
    """Restituisce il codice comune valido oggi, seguendo il remap storico."""
    remap = get_db()[GEO_REMAP].find_one(
        {"old_code": istat_code, "year": {"$lte": year}}, sort=[("year", -1)]
    )
    return remap["new_code"] if remap else istat_code


def register_remap(old_code: str, new_code: str, year: int, note: str = "") -> None:
    get_db()[GEO_REMAP].update_one(
        {"old_code": old_code, "year": year},
        {"$set": {"new_code": new_code, "note": note}},
        upsert=True,
    )


def _col(row: Dict[str, str], *candidates: str) -> Optional[str]:
    """Trova un valore di colonna tollerando variazioni di intestazione."""
    norm = {k.strip().lower(): v for k, v in row.items() if k}
    for cand in candidates:
        v = norm.get(cand.strip().lower())
        if v not in (None, ""):
            return v.strip()
    return None


class IstatGeoLoader(BaseLoader):
    """Carica l'elenco comuni ISTAT (CSV ; con encoding latin-1) in ``geographies``."""

    source_name = "istat_comuni"

    def parse(self, content: bytes, raw_id: str, **kwargs) -> dict:
        text = content.decode("latin-1", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=";")
        ensure_nation()
        n_reg = n_prov = n_com = 0
        seen_reg, seen_prov = set(), set()
        for row in reader:
            rcode = _col(row, "Codice Regione", "Codice Regione (numerico)")
            pcode = _col(row, "Codice Provincia", "Codice dell'Unità territoriale sovracomunale",
                         "Codice Provincia (Storico)")
            ccode = _col(row, "Codice Comune formato alfanumerico",
                         "Codice Comune formato numerico", "Progressivo del Comune")
            rname = _col(row, "Denominazione Regione")
            pname = _col(row, "Denominazione dell'Unità territoriale sovracomunale",
                         "Denominazione provincia")
            cname = _col(row, "Denominazione in italiano",
                         "Denominazione (italiana e straniera)")
            if not (rcode and ccode and cname):
                continue
            rid = region_id(rcode)
            pid = province_id(pcode) if pcode else rid
            cid = comune_id(ccode)
            if rid not in seen_reg:
                upsert_geography({"_id": rid, "level": "regione", "name": rname or rid,
                                  "parent": NATION_ID, "region": rid, "istat_code": rcode,
                                  "_meta": {"source": self.source_name, "raw_ref": raw_id}})
                seen_reg.add(rid); n_reg += 1
            if pcode and pid not in seen_prov:
                upsert_geography({"_id": pid, "level": "provincia", "name": pname or pid,
                                  "parent": rid, "region": rid, "istat_code": pcode,
                                  "_meta": {"source": self.source_name, "raw_ref": raw_id}})
                seen_prov.add(pid); n_prov += 1
            upsert_geography({"_id": cid, "level": "comune", "name": cname,
                              "parent": pid, "region": rid, "istat_code": ccode,
                              "_meta": {"source": self.source_name, "raw_ref": raw_id}})
            n_com += 1
        return {"regioni": n_reg, "province": n_prov, "comuni": n_com}


def list_children(geo_id: str, level: Optional[str] = None) -> List[dict]:
    q: Dict = {"parent": geo_id}
    if level:
        q["level"] = level
    return list(get_db()[GEOGRAPHIES].find(q))


def region_of(geo_id: str) -> Optional[str]:
    doc = get_db()[GEOGRAPHIES].find_one({"_id": geo_id}, {"region": 1})
    return doc.get("region") if doc else None
