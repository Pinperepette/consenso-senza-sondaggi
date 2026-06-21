"""Previsione corretta coi VOTI REALI RECENTI.

Idea (l'argomento di Borghi): se i voti reali recenti divergono dai sondaggi
(es. la Lega cala molto meno nelle urne di quanto dicano i sondaggi), la stima
del presente — che si appoggia ai sondaggi — va corretta in quella direzione.

Onesto: le comunali NON sono il dato nazionale, quindi la correzione e':
- proporzionale alla DIVERGENZA urne-vs-sondaggi (scheda swings),
- SMORZATA (le comunali sottostimano i livelli nazionali) e CAPPATA (max pochi punti),
- dichiarata come IPOTESI ("se il segnale locale vale anche sul nazionale"),
- con incertezza alta. Non e' un numero certo, e' una lettura alternativa.
"""
from __future__ import annotations

from typing import Optional

DAMP = 0.35      # quanto del segnale locale si trasferisce al nazionale (prudente)
CAP = 4.0        # correzione massima in punti, in valore assoluto


def forecast_adjusted(as_of: Optional[str] = None) -> dict:
    from consenso.model.nowcast import nowcast
    from consenso.model.swings import swings

    nc = nowcast(as_of)
    if "error" in nc:
        return nc
    disc = {p["party"]: p["discrepancy"] for p in swings()["parties"]}  # poll - urne
    out = []
    for p in nc["parties"]:
        name = p["name"]
        model = p["mean"] * 100
        d = disc.get(name)            # divergenza in punti di swing (urne vs sondaggi)
        # d>0: i sondaggi salgono piu' delle urne -> partito sovrastimato -> aggiusta GIU'
        # d<0: i sondaggi calano piu' delle urne -> partito sottostimato -> aggiusta SU'
        adj = max(-CAP, min(CAP, -(d or 0.0) * DAMP))
        out.append({"party": name, "model": round(model, 1),
                    "signal": (round(d, 1) if d is not None else None),
                    "adjustment": round(adj, 1),
                    "adjusted": round(model + adj, 1)})
    return {"as_of": nc["as_of"], "damping": DAMP, "cap": CAP, "parties": out}
