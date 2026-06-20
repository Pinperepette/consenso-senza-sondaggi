"""Loader dell'affluenza e degli aventi diritto (Ministero dell'Interno).

Formato "long": una riga per area con elettori (aventi diritto) e votanti.
Calcola turnout = votanti / aventi diritto.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional

from pymongo import UpdateOne

from consenso.db.client import get_db
from consenso.db.schema import TURNOUT
from consenso.etl.base import BaseLoader, http_get
from consenso.etl.sources.eligendo import _geo_id, _pick

DEFAULT_COLMAP = {
    "geo_code": ["CODICE_COMUNE", "COD_COMUNE", "PROCOM", "CODICE", "COD_ISTAT"],
    "eligible": ["ELETTORI", "AVENTI_DIRITTO", "ELETTORI_TOTALE", "ISCRITTI"],
    "voters": ["VOTANTI", "VOTANTI_TOTALE", "AFFLUENZA_VOTANTI"],
    "blank": ["SCHEDE_BIANCHE", "BIANCHE"],
    "invalid": ["SCHEDE_NULLE", "NULLE", "VOTI_NULLI"],
}


class TurnoutLoader(BaseLoader):
    source_name = "interno_turnout"

    def fetch(self, url: str, **kwargs) -> bytes:  # pragma: no cover - rete
        return http_get(url)

    def parse(self, content: bytes, raw_id: str, *, election_id: str,
              geo_level: str = "comune", colmap: Optional[dict] = None,
              delimiter: str = ";", encoding: str = "latin-1", **kwargs) -> dict:
        cmap = {**DEFAULT_COLMAP, **(colmap or {})}
        text = content.decode(encoding, errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        ops: List[UpdateOne] = []
        for row in reader:
            code = _pick(row, cmap["geo_code"])
            elig = _pick(row, cmap["eligible"])
            vot = _pick(row, cmap["voters"])
            if not (code and elig and vot):
                continue
            try:
                e = int(float(elig.replace(".", "").replace(",", ".")))
                v = int(float(vot.replace(".", "").replace(",", ".")))
            except ValueError:
                continue
            blank = _to_int(_pick(row, cmap["blank"]))
            invalid = _to_int(_pick(row, cmap["invalid"]))
            gid = _geo_id(geo_level, code)
            ops.append(UpdateOne(
                {"election_id": election_id, "geo_id": gid},
                {"$set": {"election_id": election_id, "geo_id": gid, "geo_level": geo_level,
                          "eligible": e, "voters": v,
                          "turnout": (v / e) if e > 0 else None,
                          "blank": blank, "invalid": invalid,
                          "_meta": {"source": self.source_name, "raw_ref": raw_id}}},
                upsert=True))
        if ops:
            get_db()[TURNOUT].bulk_write(ops)
        return {"areas": len(ops), "election_id": election_id, "geo_level": geo_level}


def _to_int(s: Optional[str]) -> int:
    if not s:
        return 0
    try:
        return int(float(s.replace(".", "").replace(",", ".")))
    except ValueError:
        return 0
