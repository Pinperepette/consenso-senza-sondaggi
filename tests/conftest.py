"""Configurazione pytest: isola un database Mongo dedicato ai test.

Imposta ``CONSENSO_DB`` PRIMA di importare qualsiasi modulo del pacchetto, così
``config.CONFIG`` punta al db di test. Le collection vengono ricreate pulite ad
ogni sessione.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CONSENSO_DB", "consenso_test")
os.environ.setdefault("CONSENSO_NUM_CHAINS", "1")   # test più rapidi
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _clean_db():
    from consenso.db.client import get_client
    from config import CONFIG
    from consenso.db.schema import ensure_collections

    client = get_client()
    client.drop_database(CONFIG.mongo.db_name)
    ensure_collections()
    yield
    client.drop_database(CONFIG.mongo.db_name)


@pytest.fixture(autouse=True)
def _clear_collections():
    """Svuota le collection fra un test e l'altro (mantiene indici)."""
    from consenso.db.client import get_db
    from consenso.db.schema import ALL_COLLECTIONS

    db = get_db()
    yield
    for c in ALL_COLLECTIONS:
        db[c].delete_many({})
