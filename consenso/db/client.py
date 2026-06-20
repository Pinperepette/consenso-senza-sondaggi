"""Connessione MongoDB (singleton lazy) e accesso a GridFS."""
from __future__ import annotations

from functools import lru_cache

import gridfs
from pymongo import MongoClient
from pymongo.database import Database

from config import CONFIG


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(CONFIG.mongo.uri, tz_aware=True)


def get_db() -> Database:
    return get_client()[CONFIG.mongo.db_name]


@lru_cache(maxsize=1)
def get_gridfs() -> gridfs.GridFS:
    """GridFS per i posterior samples (payload pesanti, fuori dai documenti)."""
    return gridfs.GridFS(get_db(), collection="posterior_samples")
