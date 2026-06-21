"""Sondaggi vs Urne: confronto fra lo swing REALE nei comuni che hanno votato
(comunali 2020 vs 2026, voto omogeneo) e lo swing implicito dai SONDAGGI nello
stesso periodo. Dati reali (paniere di 15 comuni) + sondaggio La7.

Punto chiave: FI e M5S fungono da CONTROLLO (urne e sondaggi concordano); se
Lega e FdI divergono molto mentre i controlli no, la divergenza non e' un mero
artefatto locale<->nazionale ma indica un bias specifico dei sondaggi.

Cautele oneste: le comunali non sono il dato nazionale (le liste civiche drenano
voti a tutti); quindi si legge la DIREZIONE e la magnitudine RELATIVA dello swing,
non un numero nazionale. Paniere e periodo sono quelli dell'analisi.
"""
from __future__ import annotations

from typing import Dict

# per comune: partito -> (% 2020, % 2026). Comunali, voto omogeneo.
BASKET: Dict[str, Dict[str, tuple]] = {
    "Venezia":         {"LEGA": (12.4, 4.7), "FDI": (6.6, 12.9), "FI": (2.7, 2.5), "M5S": (3.9, 2.6)},
    "Reggio Calabria": {"LEGA": (4.7, 6.8), "FDI": (7.7, 11.2), "FI": (11.0, 12.5), "M5S": (2.0, 0.0)},
    "Messina":         {"LEGA": (5.4, 5.1), "FDI": (8.8, 7.6), "FI": (5.1, 3.5), "M5S": (4.2, 3.0)},
    "Prato":           {"LEGA": (3.0, 2.8), "FDI": (18.0, 21.5), "FI": (4.0, 4.2), "M5S": (3.9, 2.3)},
    "Arezzo":          {"LEGA": (13.9, 4.8), "FDI": (12.5, 16.5), "FI": (6.1, 7.7), "M5S": (3.8, 2.0)},
    "Vigevano":        {"LEGA": (27.0, 9.6), "FDI": (10.5, 11.4), "FI": (9.7, 13.9), "M5S": (6.3, 2.7)},
    "Lecco":           {"LEGA": (13.7, 10.6), "FDI": (9.0, 18.6), "FI": (14.1, 5.9), "M5S": (3.6, 0.0)},
    "Chieti":          {"LEGA": (17.0, 11.0), "FDI": (13.8, 11.6), "FI": (7.8, 14.2), "M5S": (6.2, 4.5)},
    "Termini Imerese": {"LEGA": (6.0, 9.1), "FDI": (6.3, 9.5), "FI": (7.5, 7.1), "M5S": (4.4, 1.9)},
    "Macerata":        {"LEGA": (19.5, 12.9), "FDI": (10.7, 18.2), "FI": (5.8, 7.5), "M5S": (5.8, 4.3)},
    "Legnano":         {"LEGA": (15.5, 6.1), "FDI": (8.6, 22.6), "FI": (6.4, 8.0), "M5S": (4.6, 1.6)},
    "Afragola":        {"LEGA": (7.1, 5.8), "FDI": (8.4, 3.9), "FI": (7.5, 7.1), "M5S": (4.4, 1.9)},
    "Voghera":         {"LEGA": (24.0, 21.6), "FDI": (7.0, 12.4), "FI": (11.0, 14.0), "M5S": (6.4, 4.3)},
    "Marsala":         {"LEGA": (2.7, 5.5), "FDI": (7.3, 6.1), "FI": (8.3, 5.9), "M5S": (4.6, 2.0)},
    "Moncalieri":      {"LEGA": (10.7, 7.1), "FDI": (6.9, 16.0), "FI": (3.0, 9.1), "M5S": (4.8, 0.0)},
}
# sondaggio La7: partito -> (% 2020, % 2026)
POLL_LA7 = {"LEGA": (26.3, 6.0), "FDI": (14.4, 28.1), "FI": (6.3, 7.4), "M5S": (15.8, 12.7)}
CONTROL = {"FI", "M5S"}     # partiti di controllo (urne e sondaggi dovrebbero concordare)
PARTY_ORDER = ["LEGA", "FDI", "FI", "M5S"]
SOURCE = ("Paniere di 15 comuni · comunali 2020 vs 2026 (risultati reali, voto "
          "omogeneo) · swing sondaggi: La7")


def _real_2020() -> dict:
    """% 2020 calcolate dai NOSTRI dati (comunali storiche in DB), per citta'."""
    import re
    from consenso.db.client import get_db
    db = get_db()
    elections = ["elez:2020-09-20_comunali", "elez:2020-10-25_comunali"]
    out = {}
    for city in BASKET:
        for e in elections:
            rows = list(db["party_results"].find(
                {"election_id": e, "geo_id": {"$regex": f":{re.escape(city.upper())}$"}},
                {"party_id": 1, "votes": 1}))
            if not rows:
                continue
            tot = sum(r["votes"] for r in rows)
            if tot <= 0:
                break
            pct = {}
            for r in rows:
                pid = (r.get("party_id") or "").replace("party:", "")
                if pid in PARTY_ORDER:
                    pct[pid] = pct.get(pid, 0.0) + r["votes"] / tot * 100
            out[city] = pct
            break
    return out


def swings() -> dict:
    try:
        real20 = _real_2020()
    except Exception:  # noqa: BLE001
        real20 = {}
    computed = sum(1 for c in real20 if real20.get(c))

    def y20(city, p):
        return round(real20.get(city, {}).get(p, BASKET[city][p][0]), 1)

    out = []
    for p in PARTY_ORDER:
        deltas = [v[p][1] - y20(c, p) for c, v in BASKET.items() if p in v]
        real = sum(deltas) / len(deltas)
        p20, p26 = POLL_LA7[p]
        poll = p26 - p20
        out.append({"party": p, "real_swing": round(real, 1),
                    "poll_swing": round(poll, 1),
                    "discrepancy": round(poll - real, 1),   # quanto il sondaggio si muove PIU' della realta'
                    "control": p in CONTROL})
    cities = [{"city": c, "src20": "dati" if real20.get(c) else "articolo",
               **{p: {"y20": y20(c, p), "y26": v[p][1], "d": round(v[p][1] - y20(c, p), 1)}
                  for p in PARTY_ORDER if p in v}}
              for c, v in BASKET.items()]
    src = SOURCE + (f" · 2020 calcolato dai nostri dati per {computed}/{len(BASKET)} comuni"
                    if computed else "")
    return {"source": src, "n_cities": len(BASKET), "computed_2020": computed,
            "parties": out,
            "poll_la7": {p: {"y20": POLL_LA7[p][0], "y26": POLL_LA7[p][1]} for p in PARTY_ORDER},
            "cities": cities}
