"""Motore di swing GENERALE e data-driven.

Per una qualunque elezione 'target' (comunali o regionali) calcola, sulle unita'
che hanno gia' votato in passato (stesso tipo), lo SWING REALE per partito e lo
confronta con lo SWING DEI SONDAGGI nello stesso periodo, ricavato dai NOSTRI
dati (collection polls). Niente panieri scritti a mano.

Validita' data-driven: i 'controlli' sono i partiti con discrepanza piccola
(<CONTROL_TOL); se ce ne sono almeno 2 e qualche partito diverge molto, la
divergenza non e' artefatto locale-vs-nazionale ma segnale reale.

Il caso Borghi (comunali 2020->2026, sondaggi La7) resta in swings.py come
validazione di riferimento; qui il segnale e' calcolato su qualunque dato reale.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date as _date
from statistics import mean
from typing import Optional

GP = ["LEGA", "FDI", "FI", "M5S", "PD", "AVS"]
CONTROL_TOL = 2.5      # |discrepanza| sotto cui un partito fa da controllo
SIGNAL_MIN = 3.0       # |discrepanza| sopra cui e' "segnale" forte


def _d(s: str) -> str:
    return (s or "")[:10]


def poll_value(db, party: str, day: str, window: int = 90) -> Optional[float]:
    """Media nazionale dei sondaggi per un partito in una finestra attorno a 'day'."""
    pid = "party:" + party
    try:
        d0 = _date.fromisoformat(_d(day))
    except ValueError:
        return None
    for w in (window, window * 2, window * 4):     # allarga se pochi dati
        lo = _date.fromordinal(d0.toordinal() - w).isoformat()
        hi = _date.fromordinal(d0.toordinal() + w).isoformat()
        vals = [r["share"] for r in db["polls"].find(
            {"party_id": pid, "date": {"$gte": lo, "$lte": hi}}, {"share": 1})]
        if len(vals) >= 3:
            return 100 * mean(vals)
    return None


def _shares_by_geo(db, election_ids, geos=None) -> dict:
    """Per ciascun geo, la tornata piu' RECENTE tra election_ids: voti per partito + totale.
    election_ids: dict {id: date}. Se geos dato, filtra a quei geo."""
    q = {"election_id": {"$in": list(election_ids)}}
    if geos is not None:
        q["geo_id"] = {"$in": list(geos)}
    best = {}
    for r in db["party_results"].find(q, {"geo_id": 1, "party_id": 1, "votes": 1, "election_id": 1}):
        d = election_ids.get(r["election_id"], "")
        g = best.get(r["geo_id"])
        if g is None:
            g = {"date": d, "tot": 0, "p": defaultdict(int)}
            best[r["geo_id"]] = g
        if d < g["date"]:
            continue
        if d > g["date"]:
            g["date"] = d
            g["tot"] = 0
            g["p"] = defaultdict(int)
        g["tot"] += r["votes"]
        if r.get("party_id"):
            g["p"][r["party_id"].replace("party:", "")] += r["votes"]
    return best


def swing_signal(target_eid: str) -> dict:
    """Segnale di swing reale-vs-sondaggi per l'elezione target."""
    from consenso.db.client import get_db
    db = get_db()
    tgt = db["elections"].find_one({"_id": target_eid}, {"type": 1, "date": 1})
    if not tgt:
        return {"error": "elezione non trovata"}
    etype, tdate = tgt["type"], _d(tgt["date"])
    target = _shares_by_geo(db, {target_eid: tdate})
    geos = set(target)
    priors = {e["_id"]: _d(e["date"]) for e in db["elections"].find(
        {"type": etype, "date": {"$lt": tgt["date"]}}, {"date": 1})}
    if not priors:
        return {"error": f"nessuna tornata {etype} precedente"}
    prior = _shares_by_geo(db, priors, geos)

    sw, prior_dates = defaultdict(list), []
    for g in geos:
        pb, tb = prior.get(g), target.get(g)
        if not pb or pb["tot"] <= 0 or tb["tot"] <= 0:
            continue
        prior_dates.append(pb["date"])
        for p in GP:
            a = 100 * pb["p"].get(p, 0) / pb["tot"]
            b = 100 * tb["p"].get(p, 0) / tb["tot"]
            if a > 0 or b > 0:
                sw[p].append(b - a)
    if not prior_dates:
        return {"error": "nessuna unita' comparabile con voto precedente"}
    prior_date = sorted(prior_dates)[len(prior_dates) // 2]   # data mediana di partenza

    parties = []
    for p in GP:
        if not sw[p]:
            continue
        real = mean(sw[p])
        pv0, pv1 = poll_value(db, p, prior_date), poll_value(db, p, tdate)
        poll = (pv1 - pv0) if (pv0 is not None and pv1 is not None) else None
        parties.append({"party": p, "n_units": len(sw[p]),
                        "real_swing": round(real, 1),
                        "poll_swing": (round(poll, 1) if poll is not None else None),
                        "discrepancy": (round(poll - real, 1) if poll is not None else None)})
    # controlli data-driven: discrepanza piccola
    withd = [p for p in parties if p["discrepancy"] is not None]
    controls = [p["party"] for p in withd if abs(p["discrepancy"]) < CONTROL_TOL]
    signal = [p["party"] for p in withd if abs(p["discrepancy"]) >= SIGNAL_MIN]
    noise = max((abs(p["discrepancy"]) for p in withd if p["party"] in controls), default=0.0)
    n_units = len(prior_dates)
    valid = len(controls) >= 2 and len(signal) >= 1 and n_units >= 30
    for p in parties:
        p["role"] = ("controllo" if p["party"] in controls else
                     "segnale" if p["party"] in signal else "—")
    return {"target": target_eid, "type": etype, "target_date": tdate,
            "prior_date": prior_date, "n_units": n_units,
            "parties": parties, "controls": controls, "signal": signal,
            "control_noise": round(noise, 1), "valid": valid}
