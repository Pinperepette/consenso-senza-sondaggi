"""Lettura di feed RSS di cronaca politica italiana.

Serve solo a procurare i titoli/sommari grezzi: la selezione e la traduzione in
assunzioni le fanno gli agenti (in scenario_ai). Nessuna percentuale qui.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from consenso.etl.base import http_get

# feed di default (RSS 2.0): usati per inizializzare la collection 'feeds',
# poi gestibili dalla dashboard.
DEFAULT_FEEDS: List[Tuple[str, str]] = [
    ("ANSA", "https://www.ansa.it/sito/notizie/politica/politica_rss.xml"),
    ("Repubblica", "https://www.repubblica.it/rss/politica/rss2.0.xml"),
    ("Il Post", "https://www.ilpost.it/politica/feed/"),
    ("Sky TG24", "https://tg24.sky.it/rss/politica.xml"),
]


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def feeds_list() -> List[Dict]:
    """Feed configurati (dalla collection 'feeds'); al primo avvio la inizializza
    con i default. Gestibile dalla dashboard."""
    from consenso.db.client import get_db
    col = get_db()["feeds"]
    if col.count_documents({}) == 0:
        col.insert_many([{"source": s, "url": u, "enabled": True}
                         for s, u in DEFAULT_FEEDS])
    return [{"source": d["source"], "url": d["url"]}
            for d in col.find({"enabled": {"$ne": False}})]


def fetch_political_news(max_items: int = 25, feeds=None) -> List[Dict]:
    """Scarica gli ultimi titoli dai feed configurati; dedup per titolo, cap."""
    flist = feeds if feeds is not None else [(f["source"], f["url"])
                                             for f in feeds_list()]
    out: List[Dict] = []
    for source, url in flist:
        try:
            root = ET.fromstring(http_get(url))
        except Exception:
            continue
        for it in root.iter("item"):
            title = _strip(it.findtext("title") or "")
            if not title:
                continue
            out.append({"title": title,
                        "summary": _strip(it.findtext("description") or "")[:320],
                        "source": source,
                        "date": (it.findtext("pubDate") or "").strip(),
                        "link": (it.findtext("link") or "").strip()})
    seen, uniq = set(), []
    for n in out:
        k = n["title"].lower()
        if k not in seen:
            seen.add(k); uniq.append(n)
    return uniq[:max_items]
