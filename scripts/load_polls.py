#!/usr/bin/env python3
"""Ingestione di TUTTI i sondaggi politici nazionali, dal 2008 a oggi, nella
collection ``polls``.

Tre fonti pubbliche, unite e deduplicate:
  - ScrapeOpen (dataset da Wikipedia, formato .tab)        2008 -> 2018
  - Wikipedia "Opinion polling for the 2022 ... election"  2019 -> 2022
  - Wikipedia "Opinion polling for the next ... election"  2022 -> oggi

I sondaggi NON sono fatti: entrano nel modello come segnale debole/indicativo,
con varianza calibrata sull'errore reale (~3,2 punti). Qui solo ingestione.
"""
from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.db.client import get_db  # noqa: E402
from consenso.etl.base import http_get  # noqa: E402

TAB = Path(__file__).resolve().parent.parent / "data" / "raw" / "polls.tab"
WIKI = ["https://en.wikipedia.org/wiki/Opinion_polling_for_the_2022_Italian_general_election",
        "https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Italian_general_election"]

# ScrapeOpen: sigle -> party_id
SCRAPE_MAP = {"PD": "party:PD", "LN": "party:LEGA", "M5S": "party:M5S",
              "FdI": "party:FDI", "FI": "party:FI", "PdL": "party:FI",
              "SEL": "party:AVS", "SI": "party:AVS", "SEL/SI": "party:AVS",
              "LeU": "party:AVS"}
# Wikipedia: token colonna -> party_id
# FN = Futuro Nazionale (Vannacci): compare nei sondaggi dal 2026, quindi lo
# tracciamo come partito a se' (alimentato solo dai sondaggi: non ha mai votato).
WIKI_TOK = {"FdI": "party:FDI", "PD": "party:PD", "M5S": "party:M5S",
            "Lega": "party:LEGA", "FI": "party:FI", "AVS": "party:AVS",
            "FN": "party:FN"}
MON = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def _num(x):
    x = re.sub(r"\[[^\]]*\]", "", str(x)).replace(",", ".").strip()
    m = re.match(r"^-?\d+(\.\d+)?", x)
    return float(m.group()) if m else None


def _pdate(s, year):
    ms = list(re.finditer(r"(\d{1,2})\s*([A-Z][a-z]{2})", str(s)))
    if not ms or not year:
        return None
    d, mo = ms[-1].groups()
    return f"{year}-{MON[mo]:02d}-{int(d):02d}" if mo in MON else None


def _parse_wiki_table(t, year):
    cols = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
            for c in t.columns]
    seg = [set(c.split()) for c in cols]
    di = next((i for i, c in enumerate(cols) if "Fieldwork" in c), None)
    fi = next((i for i, c in enumerate(cols) if "firm" in c), None)
    if di is None or fi is None:
        return []
    pcol = {tok: i for tok in WIKI_TOK for i, s in enumerate(seg) if tok in s}
    out = []
    for _, r in t.iterrows():
        d = _pdate(r.iloc[di], year)
        if not d:
            continue
        firm = re.sub(r"\[[^\]]*\]", "", str(r.iloc[fi])).strip()
        for tok, pid in WIKI_TOK.items():
            if tok in pcol:
                v = _num(r.iloc[pcol[tok]])
                if v and 0 < v < 70:
                    out.append({"pollster": firm, "date": d, "party_id": pid,
                                "share": v / 100})
    return out


def _scrape_wiki(url):
    soup = BeautifulSoup(http_get(url).decode("utf-8", "replace"), "lxml")
    year, rows = None, []
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name in ("h2", "h3", "h4"):
            m = re.search(r"(20\d\d)", el.get_text())
            if m:
                year = m.group(1)
        elif "wikitable" in (el.get("class") or []):
            try:
                df = pd.read_html(io.StringIO(str(el)))[0]
            except Exception:
                continue
            rows += _parse_wiki_table(df, year)
    return rows


SCRAPEOPEN_URL = ("https://raw.githubusercontent.com/ScrapeOpen/"
                  "Opinion-polling-for-the-next-Italian-general-election/master/"
                  "open_csv_italian_party_polls.tab")


def _scrapeopen():
    if not TAB.exists():
        try:                                  # scarica il dataset storico se manca
            TAB.parent.mkdir(parents=True, exist_ok=True)
            TAB.write_bytes(http_get(SCRAPEOPEN_URL))
        except Exception:
            return []
    out = []
    for r in csv.DictReader(open(TAB, encoding="utf-8"), delimiter="\t"):
        pid = SCRAPE_MAP.get(r["variable"])
        if pid and r["date"] < "2019-01-01":
            v = _num(r["value"])
            if v:
                out.append({"pollster": r["polling_firm"], "date": r["date"],
                            "party_id": pid, "share": v / 100})
    return out


def load_all_polls(verbose: bool = False) -> dict:
    """Scarica e aggiorna TUTTI i sondaggi (statico + Wikipedia live). Riusabile
    dall'auto-sync. Restituisce {rows, polls, max_date}."""
    rows = _scrapeopen()
    if verbose:
        print(f"ScrapeOpen 2008-2018: {len(rows)} righe")
    for url in WIKI:
        try:
            w = [x for x in _scrape_wiki(url) if x["date"] >= "2019-01-01"]
        except Exception:
            w = []
        if verbose:
            print(f"  {url.split('/')[-1][:40]}: {len(w)} righe")
        rows += w
    from scripts.normalize_pollsters import normalize as _norm_pollster
    seen, uniq = set(), []
    for r in rows:
        r["pollster"] = _norm_pollster(r["pollster"])   # unifica varianti (AnalisiPolitica, Archived, …)
        k = (r["pollster"], r["date"], r["party_id"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    db = get_db()
    db["parties"].update_one({"_id": "party:FN"},
                             {"$set": {"canonical_name": "Futuro Nazionale"}}, upsert=True)
    db["polls"].delete_many({})
    if uniq:
        db["polls"].insert_many(uniq)
    db["polls"].create_index([("date", 1)])
    # piega per istituto (house effect) vs risultati reali
    from consenso.model.inference import compute_house_effects
    n_house = compute_house_effects()
    dates = sorted({r["date"] for r in uniq})
    return {"rows": len(uniq),
            "polls": len({(r["pollster"], r["date"]) for r in uniq}),
            "max_date": dates[-1] if dates else None, "house_effects": n_house}


def main() -> int:
    r = load_all_polls(verbose=True)
    print(f"\nTOTALE: {r['rows']} righe, ~{r['polls']} rilevazioni, fino a {r['max_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
