"""Infrastruttura ETL comune: fetch HTTP resiliente, archiviazione del raw
immutabile (idempotente via hash) e lineage verso il curated.

Principi (doc §4):
  - il raw non si modifica mai;
  - re-ingestione della stessa risorsa non duplica (hash del contenuto);
  - ogni documento curated referenzia il raw da cui deriva.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import CONFIG
from consenso.db.client import get_db
from consenso.db.schema import AUDIT_LOG, RAW


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def audit(actor: str, action: str, detail: Optional[dict] = None) -> None:
    get_db()[AUDIT_LOG].insert_one(
        {"ts": utcnow(), "actor": actor, "action": action, "detail": detail or {}}
    )


def http_get(url: str, params: Optional[dict] = None) -> bytes:
    """GET con retry/backoff. Salva anche una copia su disco (raw_cache_dir)."""
    cfg = CONFIG.sources
    headers = {"User-Agent": cfg.user_agent}
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.http_retries):
        try:
            with httpx.Client(timeout=cfg.http_timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:  # noqa: BLE001 - retry su qualsiasi errore di rete
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"GET fallito dopo {cfg.http_retries} tentativi: {url}") from last_exc


def _cache_path(source: str, source_hash: str) -> str:
    safe = source.replace("/", "_").replace(":", "_")
    os.makedirs(CONFIG.sources.raw_cache_dir, exist_ok=True)
    return os.path.join(CONFIG.sources.raw_cache_dir, f"{safe}__{source_hash[:16]}.bin")


def archive_raw(source: str, content: bytes, meta: Optional[dict] = None) -> str:
    """Archivia un payload grezzo in modo idempotente.

    Restituisce l'``_id`` del documento raw (esistente o appena creato).
    """
    h = sha256_bytes(content)
    coll = get_db()[RAW]
    existing = coll.find_one({"source_hash": h}, {"_id": 1})
    if existing:
        return str(existing["_id"])

    # copia su disco per riproducibilità offline
    path = _cache_path(source, h)
    with open(path, "wb") as fh:
        fh.write(content)

    doc = {
        "source": source,
        "source_hash": h,
        "size": len(content),
        "cache_path": path,
        "ingested_at": utcnow(),
        "meta": meta or {},
    }
    res = coll.insert_one(doc)
    audit("etl", "archive_raw", {"source": source, "hash": h, "size": len(content)})
    return str(res.inserted_id)


def load_raw_content(raw_id: str) -> bytes:
    doc = get_db()[RAW].find_one({"_id": _as_object_id(raw_id)})
    if not doc:
        raise KeyError(f"raw non trovato: {raw_id}")
    with open(doc["cache_path"], "rb") as fh:
        return fh.read()


def _as_object_id(raw_id):
    from bson import ObjectId

    return ObjectId(raw_id) if not isinstance(raw_id, ObjectId) else raw_id


class BaseLoader:
    """Classe base per i loader di fonte.

    Sottoclassi implementano:
      - ``fetch(**kwargs) -> bytes``   (scarica il payload grezzo)
      - ``parse(content: bytes, raw_id: str) -> None``  (normalizza in curated)

    Il metodo ``run`` orchestra fetch -> archive_raw -> parse, con la possibilità
    di passare ``content`` già scaricato (utile per test/fixture offline).
    """

    source_name: str = "base"

    def fetch(self, **kwargs) -> bytes:  # pragma: no cover - override
        raise NotImplementedError

    def parse(self, content: bytes, raw_id: str, **kwargs) -> dict:  # pragma: no cover
        raise NotImplementedError

    def run(self, content: Optional[bytes] = None, **kwargs) -> dict:
        if content is None:
            content = self.fetch(**kwargs)
        raw_id = archive_raw(self.source_name, content, meta=kwargs)
        result = self.parse(content, raw_id, **kwargs)
        audit("etl", "load", {"source": self.source_name, "result": result})
        return result
