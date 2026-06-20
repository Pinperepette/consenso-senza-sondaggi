"""Loader della demografia ISTAT.

Due modalità:
  - ``fetch_sdmx`` interroga il web service SDMX di ISTAT (REST) per un dataflow;
  - ``DemographicsLoader.parse`` ingerisce un CSV "wide" (una riga per area) con
    gli indicatori socio-demografici usati come covariate dei prior gerarchici.

La demografia è una serie storica: ogni documento porta l'anno, e il modello usa
il valore più recente con anno ≤ data elezione.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional

from pymongo import UpdateOne

from config import CONFIG
from consenso.db.client import get_db
from consenso.db.schema import DEMOGRAPHICS
from consenso.etl.base import BaseLoader, http_get
from consenso.etl.sources.eligendo import _geo_id, _pick

DEFAULT_COLMAP = {
    "geo_code": ["CODICE_COMUNE", "PROCOM", "COD_ISTAT", "ITTER107", "CODICE"],
    "year": ["ANNO", "TIME", "TIME_PERIOD", "YEAR"],
    "population": ["POPOLAZIONE", "POP_TOTALE", "VALUE", "OBS_VALUE"],
    "median_age": ["ETA_MEDIA", "MEDIAN_AGE"],
    "income_avg": ["REDDITO_MEDIO", "INCOME"],
    "employment_rate": ["TASSO_OCCUPAZIONE", "EMPLOYMENT_RATE"],
    "density": ["DENSITA", "DENSITY"],
}


def fetch_sdmx(dataflow: str, key: str = "", params: Optional[dict] = None) -> bytes:  # pragma: no cover
    """Scarica dati da ISTAT SDMX REST.

    Esempio: dataflow='22_289' (popolazione residente). Restituisce il payload
    grezzo (CSV se ``params={'format':'csv'}``).
    """
    base = CONFIG.sources.istat_sdmx_base
    url = f"{base}/data/{dataflow}/{key}"
    return http_get(url, params=params or {"format": "csv"})


class DemographicsLoader(BaseLoader):
    source_name = "istat_demographics"

    def fetch(self, dataflow: str, **kwargs) -> bytes:  # pragma: no cover - rete
        return fetch_sdmx(dataflow, kwargs.get("key", ""), kwargs.get("params"))

    def parse(self, content: bytes, raw_id: str, *, geo_level: str = "comune",
              colmap: Optional[dict] = None, delimiter: str = ";",
              encoding: str = "utf-8", **kwargs) -> dict:
        cmap = {**DEFAULT_COLMAP, **(colmap or {})}
        text = content.decode(encoding, errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        ops: List[UpdateOne] = []
        for row in reader:
            code = _pick(row, cmap["geo_code"])
            year = _pick(row, cmap["year"])
            if not (code and year):
                continue
            try:
                y = int(float(year))
            except ValueError:
                continue
            gid = _geo_id(geo_level, code)
            doc = {"geo_id": gid, "year": y,
                   "_meta": {"source": self.source_name, "raw_ref": raw_id}}
            for field in ("population", "median_age", "income_avg",
                          "employment_rate", "density"):
                v = _pick(row, cmap[field])
                if v is not None:
                    try:
                        doc[field] = float(v.replace(",", "."))
                    except ValueError:
                        pass
            ops.append(UpdateOne({"geo_id": gid, "year": y}, {"$set": doc}, upsert=True))
        if ops:
            get_db()[DEMOGRAPHICS].bulk_write(ops)
        return {"areas": len(ops)}
