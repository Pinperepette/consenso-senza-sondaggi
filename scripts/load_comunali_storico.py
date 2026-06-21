#!/usr/bin/env python3
"""Loader delle COMUNALI STORICHE (Eligendo open-data, formato TXT ';' per
comune x lista). Diverso dal formato recente: un solo .txt con
REGIONE;PROVINCIA;COMUNE;...;COGNOME;NOME;VOTI_CANDIDATO;ELETTO;DESCR_LISTA;VOTI_LISTA;...

Costruisce geo_id 'COM:<cod_reg>:<COMUNE>' (coerente con le comunali recenti),
mappa la lista sul partito (resolve_label) e salva party_results comune-level.
Sblocca anni 2010-2022 -> piu' dati per le analisi (es. swing urne vs sondaggi).
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.db.schema import PARTY_RESULTS  # noqa: E402
from consenso.etl.base import http_get  # noqa: E402
from consenso.etl.reconcile import enqueue_unknown, resolve_label  # noqa: E402
from consenso.etl.sources.eligendo import register_election  # noqa: E402

BASE = "https://elezionistorico.interno.gov.it/daithome/documenti/opendata/comunali/"
REG2COD = {
    "PIEMONTE": "01", "VALLE D'AOSTA": "02", "VALLE D`AOSTA": "02", "LOMBARDIA": "03",
    "TRENTINO-ALTO ADIGE": "04", "TRENTINO ALTO ADIGE": "04", "VENETO": "05",
    "FRIULI-VENEZIA GIULIA": "06", "FRIULI VENEZIA GIULIA": "06", "LIGURIA": "07",
    "EMILIA-ROMAGNA": "08", "EMILIA ROMAGNA": "08", "TOSCANA": "09", "UMBRIA": "10",
    "MARCHE": "11", "LAZIO": "12", "ABRUZZO": "13", "MOLISE": "14", "CAMPANIA": "15",
    "PUGLIA": "16", "BASILICATA": "17", "CALABRIA": "18", "SICILIA": "19", "SARDEGNA": "20",
}


def _kw(label: str):
    """Fallback: mappa la lista sul partito per parole-chiave (cattura le varianti
    locali, es. 'Lega-Liga Veneta', che resolve_label non conosce). Le liste civiche
    restano non mappate (corretto)."""
    u = label.upper()
    if "FRATELLI D" in u or u.strip() in ("FDI", "FD'I"):
        return "party:FDI"
    if "FORZA ITALIA" in u:
        return "party:FI"
    if "MOVIMENTO 5 STELLE" in u or "5 STELLE" in u or "CINQUE STELLE" in u:
        return "party:M5S"
    if "LEGA" in u or "LIGA VENETA" in u:
        return "party:LEGA"
    if "PARTITO DEMOCRATICO" in u:
        return "party:PD"
    if ("ALLEANZA VERDI" in u or "VERDI E SINISTRA" in u or "SINISTRA ITALIANA" in u
            or "EUROPA VERDE" in u or "LIBERI E UGUALI" in u):
        return "party:AVS"
    return None


def _col(cols, *names):
    up = {c.strip().upper(): i for i, c in enumerate(cols)}
    for n in names:
        if n in up:
            return up[n]
    return None


def ingest_storico(date: str) -> dict:
    """date 'YYYYMMDD'. Scarica e ingerisce la tornata comunale storica."""
    raw = http_get(f"{BASE}comunali-{date}.zip")
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = next((n for n in z.namelist() if n.lower().endswith(".txt")), None)
    if not txt:
        return {"skipped": "no txt"}
    rows = list(csv.reader(io.StringIO(z.read(txt).decode("latin-1", "replace")), delimiter=";"))
    if len(rows) < 2:
        return {"skipped": "vuoto"}
    cols = rows[0]
    ci_reg, ci_com = _col(cols, "REGIONE"), _col(cols, "COMUNE")
    ci_lst = _col(cols, "DESCR_LISTA", "LISTA", "DENOMINAZIONE_LISTA", "NOME_LISTA")
    ci_vot = _col(cols, "VOTI_LISTA", "VOTILISTA", "VOTI")
    if None in (ci_reg, ci_com, ci_lst, ci_vot):
        return {"skipped": f"colonne mancanti {cols[:16]}"}

    eid = f"elez:{date[:4]}-{date[4:6]}-{date[6:8]}_comunali"
    on_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    register_election(eid, "comunali", on_date,
                      {"level": "comunale", "source": "storico"})
    # aggrega VOTI_LISTA per (comune, lista)
    agg = defaultdict(lambda: defaultdict(int))   # geo_id -> party/label -> voti
    seen = set()
    n_un = 0
    for r in rows[1:]:
        if len(r) <= max(ci_lst, ci_vot):
            continue
        reg = REG2COD.get(r[ci_reg].strip().upper())
        com = r[ci_com].strip().upper()
        label = r[ci_lst].strip()
        if not (reg and com and label):
            continue
        try:
            v = int(float(r[ci_vot].replace(".", "").replace(",", ".") or 0))
        except ValueError:
            continue
        geo = f"COM:{reg}:{com}"
        key = (geo, label)
        if key in seen:            # stessa lista ripetuta su piu' righe: conta una volta
            continue
        seen.add(key)
        pid = resolve_label(label, on_date) or _kw(label)
        if pid is None:
            n_un += 1
        agg[geo][pid or f"raw:{label}"] += v

    docs = []
    for geo, parties in agg.items():
        for pid, v in parties.items():
            docs.append({"election_id": eid, "geo_id": geo, "geo_level": "comune",
                         "party_id": None if pid.startswith("raw:") else pid,
                         "raw_label": pid[4:] if pid.startswith("raw:") else None,
                         "votes": v, "valid_votes_area": 0, "share": None,
                         "_meta": {"source": "comunali_storico"}})
    db = get_db()
    db[PARTY_RESULTS].delete_many({"election_id": eid, "geo_level": "comune"})
    if docs:
        db[PARTY_RESULTS].insert_many(docs)
    return {"election_id": eid, "comuni": len(agg), "righe": len(docs),
            "unmatched": n_un}


# tornate storiche (open-data) non coperte dal formato recente
DATES = ["20100530", "20101128", "20110515", "20110522", "20111127",
         "20120506", "20120610", "20120617", "20120624", "20121028",
         "20130526", "20131117", "20140525", "20141026", "20150531", "20151115",
         "20160605", "20161113", "20170611", "20171105", "20180610", "20180729",
         "20181021", "20190526", "20190616", "20190623", "20190707", "20190714",
         "20191110", "20200920", "20201025", "20211003", "20211010", "20211107",
         "20220612", "20221127"]


def main() -> int:
    only = sys.argv[1:] or DATES
    tot = 0
    for d in only:
        try:
            r = ingest_storico(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  {d}: errore {str(exc)[:80]}"); continue
        if "election_id" in r:
            tot += r["comuni"]
            print(f"  {d}: {r['comuni']} comuni, {r['righe']} righe ({r['unmatched']} liste non mappate)")
        else:
            print(f"  {d}: saltata ({r.get('skipped')})")
    print(f"TOTALE comuni-tornata: {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
