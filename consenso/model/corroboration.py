"""Corroborazione REGIONALE della divergenza urne-vs-sondaggi.

Confronta lo swing reale 2020->2025 nelle regioni che hanno rivotato
(Veneto, Campania, Puglia) con lo swing dei sondaggi. Se i controlli (FI, M5S)
tornano e Lega/FdI divergono nella STESSA direzione delle comunali, la
correzione e' corroborata da una fonte indipendente.

2020: archivio storico Eligendo (auto-cache nella collection regional_2020).
2025: party_results gia' nel DB. Onesto: il Veneto gonfia la Lega (effetto Zaia),
ma Campania/Puglia confermano comunque.
"""
from __future__ import annotations

from collections import defaultdict

PARTIES = ["LEGA", "FDI", "FI", "M5S"]
CONTROL = {"FI", "M5S"}
POLL = {"LEGA": -20.3, "FDI": 13.7, "FI": 1.1, "M5S": -3.1}   # swing La7 2020->2026
REGIONS_2025 = {"VENETO": "elez:2025_reg_veneto",
                "CAMPANIA": "elez:2025_reg_campania",
                "PUGLIA": "elez:2025_reg_puglia"}
_HIST = ("https://elezionistorico.interno.gov.it/daithome/documenti/opendata/"
         "regionali/regionali-20200920.zip")


def _ensure_2020(db) -> dict:
    """Totali regionali 2020 per partito (cache in collection regional_2020)."""
    cached = {d["_id"]: d["parties"] for d in db["regional_2020"].find()}
    if all(r in cached for r in REGIONS_2025):
        return cached
    import csv
    import io
    import zipfile
    from consenso.etl.base import http_get
    from consenso.etl.reconcile import resolve_label
    from scripts.load_comunali_storico import _kw
    z = zipfile.ZipFile(io.BytesIO(http_get(_HIST)))
    txt = next(n for n in z.namelist() if n.endswith("candidDiLista.txt"))
    rows = list(csv.reader(io.StringIO(z.read(txt).decode("latin-1", "replace")), delimiter=";"))
    H = {c.strip().upper(): i for i, c in enumerate(rows[0])}
    ir, ic, il, iv = H["REGIONE"], H["COMUNE"], H["DESCRLISTA"], H["VOTILISTA"]
    seen, agg = set(), defaultdict(lambda: defaultdict(int))
    for r in rows[1:]:
        if len(r) <= iv:
            continue
        reg, com, lab = r[ir].strip().upper(), r[ic].strip().upper(), r[il].strip()
        if reg not in REGIONS_2025 or (reg, com, lab) in seen:
            continue
        seen.add((reg, com, lab))
        try:
            v = int(float(r[iv] or 0))
        except ValueError:
            continue
        pid = resolve_label(lab, "2020-09-20") or _kw(lab)
        agg[reg][pid.replace("party:", "") if pid else "_civ"] += v
    out = {}
    for reg, parties in agg.items():
        tot = sum(parties.values())
        sh = {p: 100 * parties.get(p, 0) / tot for p in PARTIES} if tot else {}
        db["regional_2020"].update_one({"_id": reg}, {"$set": {"parties": sh}}, upsert=True)
        out[reg] = sh
    return out


def _share_2025(db, eid) -> dict:
    agg, tot = defaultdict(int), 0
    for x in db["party_results"].find({"election_id": eid}, {"party_id": 1, "votes": 1}):
        tot += x["votes"]
        if x.get("party_id"):
            agg[x["party_id"].replace("party:", "")] += x["votes"]
    return {p: 100 * agg.get(p, 0) / tot for p in PARTIES} if tot else {}


def regional_corroboration() -> dict:
    from consenso.db.client import get_db
    db = get_db()
    try:
        s20 = _ensure_2020(db)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)[:80]}
    swings = defaultdict(list)
    for reg, eid in REGIONS_2025.items():
        a, b = s20.get(reg, {}), _share_2025(db, eid)
        for p in PARTIES:
            if p in a and p in b:
                swings[p].append(b[p] - a[p])
    if not swings:
        return {"available": False, "error": "nessuna regione confrontabile"}
    parties = []
    for p in PARTIES:
        if not swings[p]:
            continue
        real = sum(swings[p]) / len(swings[p])
        parties.append({"party": p, "real_swing": round(real, 1),
                        "poll_swing": POLL[p], "discrepancy": round(POLL[p] - real, 1),
                        "control": p in CONTROL})
    noise = max((abs(x["discrepancy"]) for x in parties if x["control"]), default=0.0)
    controls_ok = noise < 2.5
    # corrobora se i controlli tornano e Lega(disc<0)/FdI(disc>0) divergono nello stesso verso
    by = {x["party"]: x["discrepancy"] for x in parties}
    same_dir = by.get("LEGA", 0) < -2 and by.get("FDI", 0) > 2
    return {"available": True, "regions": list(REGIONS_2025), "parties": parties,
            "control_noise": round(noise, 1), "controls_ok": controls_ok,
            "corroborates": bool(controls_ok and same_dir)}
