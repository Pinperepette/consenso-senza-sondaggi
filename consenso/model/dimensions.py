"""Spazio dimensionale dei partiti: ogni partito come vettore di ~20 descrittori
politico-valoriali (immigrazione, sicurezza, sovranismo, UE, ...).

REGOLA DI ONESTA': questo NON e' il consenso (quanti voti). E' il *posizionamento*
(cosa rappresenta un partito), un layer separato. Ogni punteggio porta la sua
FONTE: 'ai' (LLM dai programmi/dichiarazioni), oppure 'ches'/'manifesto' quando si
agganciano i dati degli esperti. I due piani non si mescolano mai nei numeri del
consenso.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from consenso.ai.deepseek import chat_json
from consenso.db.client import get_db

# 20 descrittori per la politica italiana: (chiave, etichetta, polo-alto)
AXES: List[Tuple[str, str, str]] = [
    ("immigrazione", "Immigrazione restrittiva", "linea dura su immigrazione"),
    ("sicurezza", "Sicurezza e ordine", "enfasi su ordine pubblico e pene"),
    ("sovranismo", "Sovranismo / nazione", "primato dell'interesse nazionale"),
    ("euroscetticismo", "Euroscetticismo", "critico verso l'UE"),
    ("atlantismo", "Atlantismo (NATO/USA)", "filo NATO e Stati Uniti"),
    ("russia", "Apertura alla Russia", "dialogo/ammorbidimento verso Mosca"),
    ("liberismo", "Liberismo di mercato", "meno stato, piu' mercato"),
    ("taglio_tasse", "Taglio delle tasse", "riduzione della pressione fiscale"),
    ("welfare", "Welfare e redistribuzione", "stato sociale forte"),
    ("assistenzialismo", "Assistenzialismo", "sussidi tipo reddito di cittadinanza"),
    ("diritti_civili", "Diritti civili / LGBTQ+", "progressista sui diritti"),
    ("ambiente", "Ambiente / transizione", "priorita' ecologica"),
    ("valori_tradizionali", "Valori tradizionali / religione", "famiglia e radici cristiane"),
    ("populismo", "Populismo / anti-establishment", "contro le elite"),
    ("autonomismo", "Autonomismo regionale", "piu' poteri alle regioni"),
    ("giustizialismo", "Giustizialismo", "linea dura su magistratura e corruzione"),
    ("statalismo", "Interventismo statale", "stato nell'economia"),
    ("pacifismo", "Pacifismo / anti-riarmo", "contrario a spese militari e invio armi"),
    ("protezionismo", "Protezionismo / Made in Italy", "difesa produzione nazionale"),
    ("presidenzialismo", "Riforme / presidenzialismo", "esecutivo forte, riforme istituzionali"),
]

PARTY_NAMES = {
    "party:FDI": "Fratelli d'Italia", "party:PD": "Partito Democratico",
    "party:M5S": "Movimento 5 Stelle", "party:LEGA": "Lega",
    "party:FI": "Forza Italia", "party:AVS": "Alleanza Verdi e Sinistra",
    "party:FN": "Futuro Nazionale (Vannacci)",
}

_SYS = ("Sei un politologo italiano. Valuti il POSIZIONAMENTO di un partito su una "
        "serie di assi, dando a ciascuno un punteggio INTERO da 0 a 10 (0 = per "
        "niente, 10 = moltissimo), in base a programmi e linea pubblica nota. "
        "Non e' un giudizio di valore, e' una collocazione. Rispondi SOLO JSON: "
        '{"scores":{"chiave":voto,...},"summary":"1 frase sul profilo"}. '
        "Usa esattamente le chiavi date.")


def score_party(party_id: str) -> Dict:
    """Punteggi 0-10 di un partito sui 20 assi, via LLM (fonte 'ai')."""
    name = PARTY_NAMES.get(party_id, party_id.replace("party:", ""))
    axes_txt = "\n".join(f"- {k}: {label} ({pole})" for k, label, pole in AXES)
    user = (f"PARTITO: {name}\n\nASSI (chiave: descrizione):\n{axes_txt}\n\n"
            "Dai un voto 0-10 a ogni chiave secondo la linea del partito.")
    out = chat_json(_SYS, user)
    scores = {k: float(max(0, min(10, out.get("scores", {}).get(k, 5))))
              for k, _, _ in AXES}
    return {"party_id": party_id, "name": name.split(" (")[0],
            "scores": scores, "summary": out.get("summary", ""),
            "source": "ai"}


def generate_all(party_ids: List[str]) -> int:
    """Calcola e salva i vettori per i partiti dati (collection party_dimensions)."""
    db = get_db()
    n = 0
    for pid in party_ids:
        doc = score_party(pid)
        db["party_dimensions"].update_one({"party_id": pid}, {"$set": doc}, upsert=True)
        n += 1
    return n


def load_vectors() -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Carica la matrice partiti x assi. Restituisce (ids, nomi, matrice, chiavi)."""
    keys = [k for k, _, _ in AXES]
    docs = list(get_db()["party_dimensions"].find({}))
    order = list(PARTY_NAMES)
    docs.sort(key=lambda d: order.index(d["party_id"]) if d["party_id"] in order else 99)
    ids = [d["party_id"] for d in docs]
    names = [d.get("name", d["party_id"].replace("party:", "")) for d in docs]
    mat = np.array([[d["scores"].get(k, 5.0) for k in keys] for d in docs], dtype=float)
    return ids, names, mat, keys


def analysis() -> Dict:
    """PCA 2D + cluster + vicini piu' prossimi (prossimita' = vicinanza ideologica)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ids, names, mat, keys = load_vectors()
    if len(ids) < 2:
        return {"error": "servono almeno 2 partiti con i descrittori"}
    Z = StandardScaler().fit_transform(mat)
    xy = PCA(n_components=2, random_state=0).fit_transform(Z)
    k = min(3, len(ids))
    clusters = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    # distanze (sullo spazio standardizzato) e vicino piu' prossimo
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    nearest = {}
    for i, pid in enumerate(ids):
        order = [j for j in np.argsort(D[i]) if j != i]
        nearest[pid] = [{"party_id": ids[j], "name": names[j],
                         "dist": round(float(D[i, j]), 2)} for j in order[:3]]
    points = [{"party_id": ids[i], "name": names[i],
               "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
               "cluster": int(clusters[i])} for i in range(len(ids))]
    return {"points": points, "nearest": nearest}
