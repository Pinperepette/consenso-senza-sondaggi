"""Client minimale per l'API DeepSeek (compatibile OpenAI).

La chiave è letta da file (default ~/.server/deepseek.txt) e non viene mai
loggata. Usato solo per TRADURRE articoli in assunzioni strutturate, mai per
produrre direttamente le percentuali finali.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import httpx

KEY_PATH = os.environ.get("DEEPSEEK_KEY_FILE",
                          os.path.expanduser("~/.server/deepseek.txt"))
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


@lru_cache(maxsize=1)
def _api_key() -> str:
    with open(KEY_PATH, encoding="utf-8") as fh:
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
