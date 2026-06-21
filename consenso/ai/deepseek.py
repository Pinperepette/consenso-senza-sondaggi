"""Client minimale per l'API DeepSeek (compatibile OpenAI).

La chiave è letta da file (default ~/.server/deepseek.txt) e non viene mai
loggata. Usato solo per TRADURRE articoli in assunzioni strutturate, mai per
produrre direttamente le percentuali finali.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

KEY_PATH = os.environ.get("DEEPSEEK_KEY_FILE",
                          os.path.expanduser("~/.server/deepseek.txt"))
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _key_from_db() -> Optional[str]:
    """Chiave salvata dall'interfaccia (rotellina). Mai loggata."""
    try:
        from consenso.db.client import get_db
        d = get_db()["model_config"].find_one({"_id": "ai_key"})
        v = (d or {}).get("value")
        return v.strip() if v else None
    except Exception:  # noqa: BLE001
        return None


def key_source() -> Optional[str]:
    """Da dove arriva la chiave: 'env' | 'file' | 'db' | None (per la UI)."""
    if (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        return "env"
    try:
        if open(KEY_PATH, encoding="utf-8").read().strip():
            return "file"
    except OSError:
        pass
    return "db" if _key_from_db() else None


def set_api_key(key: str) -> None:
    """Salva la chiave nel DB (usata se non c'è in env/file)."""
    from consenso.db.client import get_db
    get_db()["model_config"].update_one(
        {"_id": "ai_key"}, {"$set": {"value": (key or "").strip()}}, upsert=True)


def _api_key() -> str:
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env and env.strip():            # comodo per Docker: chiave via variabile
        return env.strip()
    dbk = _key_from_db()               # salvata dalla rotellina
    if dbk:
        return dbk
    with open(KEY_PATH, encoding="utf-8") as fh:   # default: file locale, mai nel codice
        return fh.read().strip()


def chat_json(system: str, user: str, *, model: Optional[str] = None,
              temperature: float = 0.2, timeout: float = 120.0) -> dict:
    """Chiamata chat che forza output JSON. Restituisce il dict parseato."""
    payload = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def available() -> bool:
    try:
        return bool(_api_key())
    except OSError:
        return False
