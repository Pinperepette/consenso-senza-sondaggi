"""Endpoint REST per servire le stime probabilistiche (doc §OUTPUT).

Tutte le risposte usano l'ultimo run "completed" salvo ``?run=`` esplicito, così
le stime restano versionate e riproducibili.
"""
from __future__ import annotations

from typing import List, Optional

from flask import Blueprint, jsonify, render_template, request

from consenso.db.client import get_db
from consenso.db.schema import (ESTIMATIONS, FLOW_MODELS, GEOGRAPHIES,
                                MODEL_RUNS, PARTY_RESULTS)

api = Blueprint("api", __name__)


def _latest_run() -> Optional[str]:
    run = get_db()[MODEL_RUNS].find_one(
        {"status": {"$in": ["completed", "completed_low_quality"]}},
        sort=[("created_at", -1)])
    return run["_id"] if run else None


def _resolve_run() -> Optional[str]:
    return request.args.get("run") or _latest_run()


def _clean(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc


def _latest_estimate(run_id: str, party: str, geo_id: str) -> Optional[dict]:
    return get_db()[ESTIMATIONS].find_one(
        {"run_id": run_id, "party_id": party, "geo_id": geo_id},
        sort=[("as_of", -1)])


@api.get("/health")
def health():
    return jsonify({"status": "ok", "latest_run": _latest_run()})


@api.get("/runs")
def runs():
    out = []
    for r in get_db()[MODEL_RUNS].find(sort=[("created_at", -1)]).limit(50):
        out.append({"run_id": r["_id"], "created_at": r["created_at"].isoformat(),
                    "status": r["status"], "inference": r.get("inference", {}),
                    "n_parties": len(r.get("hyperparams", {}).get("parties", []))})
    return jsonify(out)


@api.get("/estimate/national")
def estimate_national():
    run_id = _resolve_run()
    party = request.args.get("party")
    if not run_id or not party:
        return jsonify({"error": "param 'party' richiesto (e serve un run)"}), 400
    as_of = request.args.get("as_of")
    q = {"run_id": run_id, "party_id": party, "geo_id": "ISTAT:IT"}
    if as_of:
        q["as_of"] = as_of
        doc = get_db()[ESTIMATIONS].find_one(q)
    else:
        doc = _latest_estimate(run_id, party, "ISTAT:IT")
    if not doc:
        return jsonify({"error": "stima non trovata"}), 404
    return jsonify(_clean(doc))


@api.get("/estimate/regional")
def estimate_regional():
    run_id = _resolve_run()
    party = request.args.get("party")
    region = request.args.get("region")
    if not (run_id and party and region):
        return jsonify({"error": "param 'party' e 'region' richiesti"}), 400
    as_of = request.args.get("as_of")
    if as_of:
        doc = get_db()[ESTIMATIONS].find_one(
            {"run_id": run_id, "party_id": party, "geo_id": region, "as_of": as_of})
    else:
        doc = _latest_estimate(run_id, party, region)
    if not doc:
        return jsonify({"error": "stima non trovata"}), 404
    return jsonify(_clean(doc))


@api.get("/trend")
def trend():
    run_id = _resolve_run()
    party = request.args.get("party")
    geo = request.args.get("geo", "ISTAT:IT")
    if not (run_id and party):
        return jsonify({"error": "param 'party' richiesto"}), 400
    series = []
    for d in get_db()[ESTIMATIONS].find(
            {"run_id": run_id, "party_id": party, "geo_id": geo}).sort("as_of", 1):
        series.append({"as_of": d["as_of"], "mean": d["mean"],
                       "ci95": d["ci95"], "prob_growth_6m": d.get("prob_growth_6m")})
    return jsonify({"party": party, "geo": geo, "run_id": run_id, "series": series})


@api.get("/polls")
def polls_cloud():
    """Sondaggi grezzi per un partito (la 'nuvola' dietro la linea del trend)."""
    party = request.args.get("party")
    if not party:
        return jsonify({"error": "param 'party' richiesto"}), 400
    mx = int(request.args.get("max", 600))
    rows = list(get_db()["polls"].find({"party_id": party},
                                       {"_id": 0, "date": 1, "share": 1}).sort("date", 1))
    if len(rows) > mx:                       # campiona uniformemente per leggibilita'
        step = len(rows) / mx
        rows = [rows[int(i * step)] for i in range(mx)]
    return jsonify({"party": party, "points": rows})


@api.get("/pollsters")
def pollsters_house():
    """Piega per istituto (house effect) vs i risultati reali."""
    docs = list(get_db()["pollster_house"].find({}, {"_id": 0}))
    docs.sort(key=lambda d: -d.get("abs_pts", 0))
    return jsonify({"house": docs})


@api.get("/map")
def map_view():
    """GeoJSON: ultima stima per regione di un partito."""
    run_id = _resolve_run()
    party = request.args.get("party")
    if not (run_id and party):
        return jsonify({"error": "param 'party' richiesto"}), 400
    features = []
    regions = get_db()[ESTIMATIONS].distinct(
        "geo_id", {"run_id": run_id, "party_id": party, "geo_level": "regione"})
    for reg in regions:
        doc = _latest_estimate(run_id, party, reg)
        geo = get_db()[GEOGRAPHIES].find_one({"_id": reg}, {"name": 1, "centroid": 1})
        if not doc:
            continue
        features.append({
            "type": "Feature",
            "geometry": (geo or {}).get("centroid"),
            "properties": {"region": reg, "name": (geo or {}).get("name"),
                           "mean": doc["mean"], "ci95": doc["ci95"]},
        })
    return jsonify({"type": "FeatureCollection", "party": party, "features": features})


@api.get("/flows")
def flows():
    qfrom = request.args.get("from")
    qto = request.args.get("to")
    scope = request.args.get("scope", "ISTAT:IT")
    q = {"geo_scope": scope}
    if qfrom:
        q["from_election"] = qfrom
    if qto:
        q["to_election"] = qto
    doc = get_db()[FLOW_MODELS].find_one(q, sort=[("_id", -1)])
    if not doc:
        return jsonify({"error": "matrice flussi non trovata"}), 404
    return jsonify(_clean(doc))


@api.get("/flows/latest")
def flows_latest():
    doc = get_db()[FLOW_MODELS].find_one(sort=[("_id", -1)])
    if not doc:
        return jsonify({"error": "nessun flusso stimato"}), 404
    return jsonify(_clean(doc))


@api.get("/nowcast")
def nowcast_route():
    from consenso.model.nowcast import nowcast

    res = nowcast(as_of=request.args.get("as_of"), run_id=request.args.get("run"))
    code = 404 if "error" in res else 200
    return jsonify(res), code


@api.get("/regions")
def regions_latest():
    """Ultima stima per regione di tutti i partiti (per la tabella/mappa)."""
    run_id = _resolve_run()
    if not run_id:
        return jsonify({"error": "nessun run"}), 404
    names = {g["_id"]: g["name"] for g in get_db()[GEOGRAPHIES].find(
        {"level": "regione"}, {"name": 1})}
    out = {}
    cur = get_db()[ESTIMATIONS].find({"run_id": run_id, "geo_level": "regione"})
    for d in cur:
        reg = d["geo_id"]
        rec = out.setdefault(reg, {"region": reg, "name": names.get(reg, reg),
                                   "parties": {}})
        # tieni l'ultima as_of per partito
        prev = rec["parties"].get(d["party_id"])
        if not prev or d["as_of"] > prev["as_of"]:
            rec["parties"][d["party_id"]] = {"mean": d["mean"], "as_of": d["as_of"]}
    return jsonify(list(out.values()))


_coord_cache = {}


def _comune_coords():
    """Lookup geo_id -> (lat, long) costruito una volta da main.csv (ISTAT)."""
    if _coord_cache:
        return _coord_cache
    import csv
    import os
    import re
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return re.sub(r"[^A-Z0-9 ]+", " ", s.upper()).strip()

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "main.csv")
    try:
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    reg2 = f"{int(r['cod_reg']):02d}"
                    gid = f"COM:{reg2}:{norm(r['comune'])}"
                    _coord_cache[gid] = (float(r["lat"]), float(r["long"]))
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return _coord_cache


@api.get("/comunali/elections")
def comunali_elections():
    out = []
    for e in get_db()["elections"].find({"type": "comunali"}):
        n = len(get_db()[PARTY_RESULTS].distinct("geo_id", {"election_id": e["_id"]}))
        out.append({"id": e["_id"], "date": e["date"], "n_comuni": n})
    out.sort(key=lambda x: -x["n_comuni"])     # la tornata più ricca per prima
    return jsonify(out)


@api.get("/comunali/map")
def comunali_map():
    election = request.args.get("election")
    party = request.args.get("party")
    if not (election and party):
        return jsonify({"error": "param 'election' e 'party' richiesti"}), 400
    coords = _comune_coords()
    GP = ["party:FDI", "party:PD", "party:M5S", "party:LEGA", "party:FI", "party:AVS"]
    allp = party in ("all", "vincente")
    proj = {"geo_id": 1, "share": 1, "votes": 1, "election_id": 1, "party_id": 1}
    dates = {e["_id"]: e["date"] for e in
             get_db()["elections"].find({"type": "comunali"}, {"date": 1})}
    q = {"party_id": {"$in": GP} if allp else party, "share": {"$gt": 0}}
    q["election_id"] = {"$in": list(dates)} if election == "all" else election
    # per comune: la tornata piu' recente (se 'all') e le quote dei partiti
    bygeo = {}
    for r in get_db()[PARTY_RESULTS].find(q, proj):
        d = dates.get(r["election_id"], "")
        g = bygeo.setdefault(r["geo_id"], {"date": d, "parties": {}})
        if election == "all":
            if d < g["date"]:
                continue
            if d > g["date"]:
                g["date"] = d
                g["parties"] = {}
        g["parties"][r["party_id"]] = (r["share"], r.get("votes", 0))
    pts = []
    for geo, g in bygeo.items():
        c = coords.get(geo)
        if not c or not g["parties"]:
            continue
        pid = max(g["parties"], key=lambda k: g["parties"][k][0]) if allp else party
        if pid not in g["parties"]:
            continue
        sh, vt = g["parties"][pid]
        pts.append({"lat": c[0], "lng": c[1], "name": geo.split(":", 2)[-1].title(),
                    "party": pid.replace("party:", ""), "share": sh, "votes": vt})
    return jsonify({"election": election, "party": party, "points": pts})


_sync_running = {"on": False}


@api.post("/sync")
def trigger_sync():
    """Avvia l'aggancio automatico in background (download nuove elezioni + rerun)."""
    import threading

    from consenso.pipeline.autosync import sync

    if _sync_running["on"]:
        return jsonify({"status": "already_running"}), 202

    def _run():
        _sync_running["on"] = True
        try:
            sync(rerun=True)
        finally:
            _sync_running["on"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"}), 202


@api.get("/sync/status")
def sync_status():
    from consenso.pipeline.autosync import last_status

    s = last_status()
    s["running"] = _sync_running["on"]
    return jsonify(s)


@api.get("/ai/status")
def ai_status():
    from consenso.ai.deepseek import MODEL, available
    return jsonify({"available": available(), "model": MODEL})


@api.post("/scenario/ai")
def scenario_ai():
    body = request.get_json(force=True, silent=True) or {}
    articles = (body.get("articles") or "").strip()
    as_of = body.get("as_of")
    if not articles:
        return jsonify({"error": "incolla almeno un articolo"}), 400
    from consenso.ai.scenario_ai import generate_spec
    try:
        gen = generate_spec(articles, as_of)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"AI non disponibile: {exc}"}), 502
    if "error" in gen:
        return jsonify(gen), 404
    from consenso.model.scenario import run_scenario
    res = run_scenario(gen["spec"], as_of)
    res["spec"] = gen["spec"]
    return jsonify(res)


@api.get("/scenario/feeds")
def list_feeds():
    """Elenca i feed RSS configurati."""
    from consenso.ai.news import feeds_list
    return jsonify({"feeds": feeds_list()})


@api.post("/scenario/feeds")
def add_feed():
    """Aggiunge (o riattiva) un feed RSS."""
    body = request.get_json(force=True, silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url.startswith("http"):
        return jsonify({"error": "URL non valido"}), 400
    source = (body.get("source") or url.split("/")[2]).strip()
    get_db()["feeds"].update_one(
        {"url": url}, {"$set": {"url": url, "source": source, "enabled": True}},
        upsert=True)
    from consenso.ai.news import feeds_list
    return jsonify({"feeds": feeds_list()})


@api.post("/scenario/feeds/remove")
def remove_feed():
    """Rimuove un feed RSS."""
    body = request.get_json(force=True, silent=True) or {}
    get_db()["feeds"].delete_one({"url": (body.get("url") or "").strip()})
    from consenso.ai.news import feeds_list
    return jsonify({"feeds": feeds_list()})


def _coal_arg():
    """Composizione coalizioni: dal body POST se presente, altrimenti i default."""
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        c = body.get("coalitions")
        return (c or None), body.get("as_of")
    return None, request.args.get("as_of")


@api.get("/settings/ai")
def settings_ai_get():
    """Stato della chiave AI (non restituisce mai la chiave)."""
    from consenso.ai.deepseek import key_source, available
    src = key_source()
    return jsonify({"configured": bool(src), "source": src, "available": available()})


@api.post("/settings/ai")
def settings_ai_set():
    """Salva la chiave AI nel DB (solo se non già in env/file)."""
    from consenso.ai.deepseek import set_api_key, key_source, available
    b = request.get_json(force=True, silent=True) or {}
    key = (b.get("key") or "").strip()
    src = key_source()
    if src in ("env", "file"):
        return jsonify({"error": f"chiave già configurata via {src}"}), 400
    if not key:
        set_api_key("")
        return jsonify({"configured": False, "source": None, "available": False})
    set_api_key(key)
    return jsonify({"configured": True, "source": "db", "available": available()})


@api.get("/forecast")
def forecast_route():
    """Previsione: stima dei sondaggi vs stima corretta coi voti reali recenti."""
    from consenso.model.forecast import forecast_adjusted
    res = forecast_adjusted(request.args.get("as_of"))
    return jsonify(res), (404 if "error" in res else 200)


@api.get("/data/overview")
def data_overview():
    """Riepilogo visivo di cosa c'è nel DB."""
    db = get_db()
    from consenso.db.schema import PARTY_RESULTS
    by_type = {}
    for e in db["elections"].find({}, {"type": 1}):
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    polls = db["polls"].count_documents({})
    dates = [d["date"] for d in db["polls"].find({}, {"date": 1}).sort("date", 1).limit(1)]
    dates += [d["date"] for d in db["polls"].find({}, {"date": 1}).sort("date", -1).limit(1)]
    return jsonify({
        "polls": polls, "polls_manual": db["polls"].count_documents({"_manual": True}),
        "polls_span": dates, "pollsters": len(db["polls"].distinct("pollster")),
        "elections": db["elections"].count_documents({}), "elections_by_type": by_type,
        "party_results": db[PARTY_RESULTS].count_documents({}),
        "results_manual": db[PARTY_RESULTS].count_documents({"_meta.source": "manuale"}),
    })


def _quarter_avg(rows):
    buckets = {}
    for r in rows:
        d = r.get("date", "")
        if len(d) < 7:
            continue
        q = (int(d[5:7]) - 1) // 3 + 1
        b = buckets.setdefault(f"{d[:4]}-Q{q}", [0.0, 0])
        b[0] += r["share"]; b[1] += 1
    return [{"period": k, "value": round(100 * v[0] / v[1], 1)}
            for k, v in sorted(buckets.items()) if v[1]]


@api.get("/timeline")
def timeline():
    """Trend REALE del consenso di un partito (sondaggi trimestrali + risultati
    elettorali) annotato con EVENTI datati e verificabili. Il legame evento->consenso
    è una lettura cronologica, non una prova di causa-effetto."""
    import json as _json
    from pathlib import Path
    db = get_db()
    party = (request.args.get("party") or "LEGA").replace("party:", "")
    pid = "party:" + party
    # serie trimestrale dai sondaggi
    series = _quarter_avg(db["polls"].find({"party_id": pid}, {"date": 1, "share": 1}))
    sidx = {s["period"]: s["value"] for s in series}

    def consensus_at(day: str):
        if not day or len(day) < 7:
            return None
        q = (int(day[5:7]) - 1) // 3 + 1
        return sidx.get(f"{day[:4]}-Q{q}")
    # risultati elettorali reali (nazionali) come punti
    elec = []
    for e in db["elections"].find({"type": {"$in": ["politiche", "europee"]}},
                                  {"date": 1, "type": 1}).sort("date", 1):
        rows = list(db[PARTY_RESULTS].find({"election_id": e["_id"]}, {"party_id": 1, "votes": 1}))
        tot = sum(r.get("votes", 0) for r in rows)
        v = sum(r.get("votes", 0) for r in rows if r.get("party_id") == pid)
        if tot and v:
            elec.append({"date": e["date"], "type": e["type"], "value": round(100 * v / tot, 1)})
    # eventi curati
    fx = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures" / "political_events.json"
    events, seen = [], set()
    if fx.exists():
        for ev in _json.loads(fx.read_text(encoding="utf-8")).get(party, []):
            seen.add((ev["date"], ev["title"]))
            events.append({**ev, "origin": "curato", "consensus": consensus_at(ev["date"])})
    # eventi scoperti dall'AI (da verificare), esclusi i duplicati dei curati
    for ev in db["events_auto"].find({"party": party}, {"_id": 0}):
        if (ev.get("date"), ev.get("title")) in seen:
            continue
        events.append({**ev, "consensus": consensus_at(ev["date"])})
    events.sort(key=lambda e: e.get("date") or "")
    return jsonify({"party": party, "series": series, "elections": elec,
                    "events": events,
                    "parties_available": [k for k in (_json.loads(fx.read_text(encoding="utf-8")).keys()
                    if fx.exists() else []) if not k.startswith("_")]})


@api.post("/timeline/discover")
def timeline_discover():
    """Scopre automaticamente eventi datati per un partito (AI su Wikipedia), da verificare."""
    b = request.get_json(force=True, silent=True) or {}
    from consenso.ai.events import discover_party
    r = discover_party((b.get("party") or "LEGA").replace("party:", ""))
    return jsonify(r), (502 if "error" in r else 200)


@api.get("/data/poll_trend")
def data_poll_trend():
    """Media trimestrale dei sondaggi per partito; opzionalmente la serie di un singolo istituto."""
    db = get_db()
    party = "party:" + (request.args.get("party") or "FDI").replace("party:", "")
    pollster = request.args.get("pollster") or None
    proj = {"date": 1, "share": 1}
    series = _quarter_avg(db["polls"].find({"party_id": party}, proj))
    out = {"party": party.replace("party:", ""), "series": series}
    if pollster:
        ps = _quarter_avg(db["polls"].find({"party_id": party, "pollster": pollster}, proj))
        out["pollster"] = pollster
        out["pollster_series"] = ps
    return jsonify(out)


@api.get("/data/polls")
def data_polls():
    """Sondaggi raggruppati per (istituto, data), piu' recenti prima."""
    db = get_db()
    limit = min(int(request.args.get("limit", 40)), 200)
    pollster = request.args.get("pollster")
    match = {"pollster": pollster} if pollster else {}
    pipe = [{"$match": match},
            {"$group": {"_id": {"d": "$date", "p": "$pollster"},
                        "shares": {"$push": {"k": "$party_id", "v": "$share"}},
                        "manual": {"$max": {"$ifNull": ["$_manual", False]}}}},
            {"$sort": {"_id.d": -1}}, {"$limit": limit}]
    out = []
    for g in db["polls"].aggregate(pipe):
        sh = {s["k"].replace("party:", ""): round(s["v"] * 100, 1) for s in g["shares"]}
        out.append({"date": g["_id"]["d"], "pollster": g["_id"]["p"],
                    "manual": bool(g["manual"]), "shares": sh})
    return jsonify({"polls": out,
                    "pollsters": sorted(db["polls"].distinct("pollster"))})


@api.get("/data/elections")
def data_elections():
    """Elenco elezioni con n. unità e top partito (nazionale o aggregato)."""
    db = get_db()
    from consenso.db.schema import PARTY_RESULTS
    etype = request.args.get("type")
    q = {"type": etype} if etype else {}
    out = []
    for e in db["elections"].find(q, {"type": 1, "date": 1}).sort("date", -1):
        n = len(db[PARTY_RESULTS].distinct("geo_id", {"election_id": e["_id"]}))
        out.append({"id": e["_id"], "type": e["type"], "date": e["date"], "n_units": n,
                    "manual": e["_id"].endswith("_manual")})
    return jsonify({"elections": out})


@api.get("/data/election")
def data_election():
    """Dettaglio di una elezione: quote aggregate per partito."""
    db = get_db()
    from consenso.db.schema import PARTY_RESULTS
    eid = request.args.get("id")
    if not eid:
        return jsonify({"error": "param 'id' richiesto"}), 400
    agg, tot = {}, 0
    for r in db[PARTY_RESULTS].find({"election_id": eid}, {"party_id": 1, "votes": 1}):
        tot += r.get("votes", 0)
        if r.get("party_id"):
            agg[r["party_id"].replace("party:", "")] = agg.get(r["party_id"].replace("party:", ""), 0) + r["votes"]
    shares = {p: round(100 * v / tot, 1) for p, v in agg.items()} if tot else {}
    return jsonify({"id": eid, "shares": dict(sorted(shares.items(), key=lambda x: -x[1]))})


@api.post("/data/poll")
def data_poll():
    b = request.get_json(force=True, silent=True) or {}
    from consenso.ingest_manual import add_poll
    r = add_poll(b.get("date", ""), b.get("pollster", ""), b.get("shares") or {})
    return jsonify(r), (400 if "error" in r else 200)


@api.post("/data/result")
def data_result():
    b = request.get_json(force=True, silent=True) or {}
    from consenso.ingest_manual import add_result
    r = add_result(b.get("date", ""), b.get("type", "comunali"), b.get("shares") or {},
                   region=b.get("region"), comune=b.get("comune"))
    return jsonify(r), (400 if "error" in r else 200)


@api.post("/data/extract")
def data_extract():
    b = request.get_json(force=True, silent=True) or {}
    from consenso.ingest_manual import extract_from_url
    r = extract_from_url(b.get("url", ""), b.get("kind", "poll"))
    return jsonify(r), (502 if "error" in r else 200)


@api.post("/data/remove")
def data_remove():
    b = request.get_json(force=True, silent=True) or {}
    from consenso.ingest_manual import remove_manual
    return jsonify(remove_manual(b.get("kind", "polls")))


@api.get("/swing/signal")
def swing_signal_route():
    """Motore di swing generale: swing reale vs sondaggi per una elezione target."""
    eid = request.args.get("election")
    if not eid:
        return jsonify({"error": "param 'election' richiesto"}), 400
    from consenso.model.swing_engine import swing_signal
    res = swing_signal(eid)
    return jsonify(res), (404 if "error" in res else 200)


@api.get("/forecast/confounders")
def forecast_confounders():
    """Strato AI (input-side): confondenti che indeboliscono la correzione locale->nazionale."""
    from consenso.model.forecast import forecast_adjusted
    from consenso.ai.confounders import confounders
    fc = forecast_adjusted(request.args.get("as_of"))
    if "error" in fc:
        return jsonify(fc), 404
    return jsonify(confounders(fc))


@api.get("/swings")
def swings_route():
    """Sondaggi vs Urne: swing reale nei comuni vs swing dei sondaggi."""
    from consenso.model.swings import swings
    return jsonify(swings())


@api.get("/flows/sankey")
def flows_sankey():
    """Flussi 2022->2024 come bande 'da->verso' per un partito (per il Sankey).
    dir=out: dove sono andati i voti del partito (proporzioni della sua riga).
    dir=in: da dove arrivano i suoi voti 2024 (composizione, pesata sui risultati 2022)."""
    db = get_db()
    f = db["flow_models"].find_one(sort=[("_id", -1)])
    if not f:
        return jsonify({"error": "nessun flusso"}), 404
    pf, pt = f["parties_from"], f["parties_to"]
    M = f["transfer_matrix_mean"]
    party = "party:" + (request.args.get("party") or "FDI").replace("party:", "")
    direction = request.args.get("dir", "out")
    nm = lambda x: x.replace("party:", "").replace("astensione", "Astensione")

    if direction == "out":
        if party not in pf:
            return jsonify({"party": nm(party), "dir": "out", "ribbons": []})
        i = pf.index(party)
        ribbons = [{"label": nm(pt[j]), "key": pt[j], "value": round(M[i][j], 4)}
                   for j in range(len(pt)) if M[i][j] > 0.005]
    else:
        # pesa per i risultati reali 2022 (composizione onesta dei voti 2024)
        agg, tot = {}, 0
        for r in db[PARTY_RESULTS].find({"election_id": f["from_election"]},
                                        {"party_id": 1, "votes": 1}):
            tot += r.get("votes", 0)
            if r.get("party_id"):
                agg[r["party_id"]] = agg.get(r["party_id"], 0) + r["votes"]
        w = {p: agg.get(p, 0) / tot for p in pf} if tot else {p: 1.0 for p in pf}
        if party not in pt:
            return jsonify({"party": nm(party), "dir": "in", "ribbons": []})
        j = pt.index(party)
        raw = [(pf[i], w.get(pf[i], 0.0) * M[i][j]) for i in range(len(pf))]
        s = sum(v for _, v in raw) or 1.0
        ribbons = [{"label": nm(k), "key": k, "value": round(v / s, 4)}
                   for k, v in raw if v / s > 0.005]
    ribbons.sort(key=lambda r: -r["value"])
    return jsonify({"party": nm(party), "dir": direction,
                    "from_election": f["from_election"], "to_election": f["to_election"],
                    "ribbons": ribbons})


@api.get("/parliament/discipline")
def parliament_discipline():
    """Cosa mostrano DAVVERO i voti finali della Camera (leg. 19): quanto ogni
    partito vota compatto col proprio blocco e le RARE rotture. Niente 'coerenza'
    parole-vs-fatti (i voti finali non la misurano), solo disciplina di voto reale."""
    db = get_db()
    GOV = ["party:FDI", "party:FI", "party:LEGA"]
    OPP = ["party:PD", "party:M5S", "party:AVS"]
    bloc = {p: "Maggioranza" for p in GOV}
    bloc.update({p: "Opposizione" for p in OPP})

    def majority(stances, members):
        fav = sum(1 for p in members if stances.get(p) == "favorevole")
        con = sum(1 for p in members if stances.get(p) == "contrario")
        if fav == 0 and con == 0:
            return None
        return "favorevole" if fav > con else "contrario" if con > fav else None

    stats = {p: {"party": p.replace("party:", ""), "bloc": bloc[p],
                 "n": 0, "aligned": 0, "breaks": []} for p in bloc}
    wide = 0
    total = 0
    for x in db["parliament_votes"].find({"leg": 19}, {"date": 1, "title": 1, "by_party": 1, "approved": 1}):
        st = {p: d.get("stance") for p, d in x.get("by_party", {}).items()
              if d.get("stance") in ("favorevole", "contrario")}
        if not st:
            continue
        total += 1
        gm, om = majority(st, GOV), majority(st, OPP)
        if gm and om and gm == om:
            wide += 1
        for p in bloc:
            if p not in st:
                continue
            bm = gm if bloc[p] == "Maggioranza" else om
            if not bm:
                continue
            stats[p]["n"] += 1
            if st[p] == bm:
                stats[p]["aligned"] += 1
            else:
                stats[p]["breaks"].append({"date": x.get("date"), "title": x.get("title"),
                                           "stance": st[p], "approved": x.get("approved")})
    out = []
    for p in bloc:
        s = stats[p]
        s["discipline"] = round(100 * s["aligned"] / s["n"], 1) if s["n"] else None
        s["n_breaks"] = len(s["breaks"])
        s["breaks"] = s["breaks"][:20]
        out.append(s)
    out.sort(key=lambda r: (r["bloc"], -(r["discipline"] or 0)))
    return jsonify({"leg": 19, "n_votes": total, "wide_consensus": wide, "parties": out})


@api.get("/parliament")
def parliament():
    """Voti finali REALI della Camera (dati aperti) per un partito, per legislatura.
    Fatti contati, non ricostruzione AI."""
    party = request.args.get("party")
    if not party:
        return jsonify({"error": "param 'party' richiesto"}), 400
    q = {f"by_party.{party}": {"$exists": True}, "title": {"$ne": "Voto finale"}}
    rows = list(get_db()["parliament_votes"].find(
        q, {"date": 1, "leg": 1, "title": 1, "approved": 1, "by_party": 1}
    ).sort("date", -1))
    votes, tally = [], {"favorevole": 0, "contrario": 0, "astenuto": 0}
    for r in rows:
        st = r["by_party"].get(party, {}).get("stance")
        if st in tally:
            tally[st] += 1
        votes.append({"date": r["date"], "leg": r["leg"], "title": r["title"],
                      "approved": r["approved"], "stance": st})
    return jsonify({"party": party.replace("party:", ""), "tally": tally,
                    "n": len(votes), "votes": votes[:80]})


@api.get("/coherence")
def coherence():
    """Coerenza fatti/parole per partito (ricostruzione AI da verificare)."""
    from consenso.model.dimensions import PARTY_NAMES
    order = list(PARTY_NAMES)
    recs = list(get_db()["coherence"].find({}, {"_id": 0}))
    recs.sort(key=lambda r: order.index(r["party_id"]) if r["party_id"] in order else 99)
    return jsonify({"parties": recs})


@api.get("/trackrecord")
def trackrecord():
    """Storico onesto: previsione del modello (allenato sul prima) vs risultato reale,
    per ogni elezione nazionale passata."""
    recs = list(get_db()["track_record"].find({}, {"_id": 0}))
    recs.sort(key=lambda r: r["date"])
    overall = None
    if recs:
        import numpy as np
        errs = [p["err"] for r in recs for p in r["parties"]]
        cov = [1.0 if p["in_ci"] else 0.0 for r in recs for p in r["parties"]]
        overall = {"mae": float(np.mean(errs)), "coverage": float(np.mean(cov)),
                   "n_elections": len(recs)}
    return jsonify({"elections": recs, "overall": overall})


@api.route("/coalitions", methods=["GET", "POST"])
def coalitions():
    """Quote delle coalizioni (somma membri) con IC95 propagato. Composizione
    modificabile via POST {coalitions:{nome:[party_id...]}}."""
    from consenso.model.coalitions import coalition_shares
    coal, as_of = _coal_arg()
    res = coalition_shares(as_of, coalitions=coal)
    return jsonify(res), (404 if "error" in res else 200)


@api.route("/seats", methods=["GET", "POST"])
def seats():
    """Proiezione semplificata dei seggi (Camera) con IC95 propagato.
    Composizione coalizioni modificabile via POST."""
    from consenso.model.coalitions import seat_projection
    coal, as_of = _coal_arg()
    res = seat_projection(as_of, coalitions=coal)
    return jsonify(res), (404 if "error" in res else 200)


@api.get("/dimensions")
def dimensions():
    """Spazio dimensionale dei partiti: vettori 20D, mappa 2D (PCA), cluster,
    vicini (prossimita' ideologica). Layer separato dal consenso."""
    from consenso.model.dimensions import AXES, analysis, load_vectors
    ids, _names, _mat, _keys = load_vectors()
    if not ids:
        return jsonify({"error": "dimensioni non ancora generate"}), 404
    docs = {d["party_id"]: d for d in get_db()["party_dimensions"].find({})}
    parties = [{"party_id": pid, "name": docs[pid].get("name"),
                "scores": docs[pid]["scores"], "summary": docs[pid].get("summary", ""),
                "source": docs[pid].get("source", "ai")} for pid in ids]
    an = analysis()
    cons = {}
    try:
        from consenso.model.nowcast import nowcast
        cons = {p["party_id"]: p["mean"] for p in nowcast().get("parties", [])}
    except Exception:  # noqa: BLE001
        pass
    for pt in an.get("points", []):
        pt["consensus"] = cons.get(pt["party_id"], 0.0)
    return jsonify({"axes": [{"key": k, "label": lbl} for k, lbl, _ in AXES],
                    "parties": parties, **an})


@api.get("/scenario/news")
def scenario_news():
    """Ultimi titoli di politica dai feed RSS (anteprima)."""
    from consenso.ai.news import fetch_political_news
    try:
        return jsonify({"news": fetch_political_news(25)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502


@api.post("/scenario/ai/auto")
def scenario_ai_auto():
    """Agente: legge i feed RSS, sceglie le notizie rilevanti e genera lo scenario."""
    body = request.get_json(force=True, silent=True) or {}
    as_of = body.get("as_of")
    from consenso.ai.scenario_ai import generate_spec_from_news
    try:
        gen = generate_spec_from_news(as_of)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"AI non disponibile: {exc}"}), 502
    if "error" in gen:
        return jsonify(gen), 404
    from consenso.model.scenario import run_scenario
    res = run_scenario(gen["spec"], as_of)
    res["spec"] = gen["spec"]
    res["news_used"] = gen.get("news_used", [])
    return jsonify(res)


@api.post("/scenario/apply")
def scenario_apply():
    body = request.get_json(force=True, silent=True) or {}
    spec = body.get("spec") or {}
    as_of = body.get("as_of")
    from consenso.model.scenario import run_scenario
    return jsonify(run_scenario(spec, as_of))


@api.post("/scenario/save")
def scenario_save():
    from datetime import datetime, timezone
    body = request.get_json(force=True, silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "nome richiesto"}), 400
    doc = {"name": name, "spec": body.get("spec") or {}, "as_of": body.get("as_of"),
           "created_at": datetime.now(timezone.utc)}
    res = get_db()["scenarios"].insert_one(doc)
    return jsonify({"id": str(res.inserted_id), "name": name})


@api.get("/scenario/list")
def scenario_list():
    out = []
    for s in get_db()["scenarios"].find().sort("created_at", -1):
        out.append({"id": str(s["_id"]), "name": s["name"], "as_of": s.get("as_of"),
                    "summary": (s.get("spec", {}).get("summary") or "")[:90]})
    return jsonify(out)


@api.delete("/scenario/<sid>")
def scenario_delete(sid):
    from bson import ObjectId
    get_db()["scenarios"].delete_one({"_id": ObjectId(sid)})
    return jsonify({"ok": True})


@api.post("/scenario/compare")
def scenario_compare():
    from bson import ObjectId

    from consenso.model.nowcast import projected_shares, summarize_shares
    from consenso.model.scenario import run_scenario

    body = request.get_json(force=True, silent=True) or {}
    ids = body.get("ids") or []
    as_of = body.get("as_of")
    parties, base, meta = projected_shares(as_of)
    if parties is None:
        return jsonify({"error": "nessun run"}), 404
    columns = [{"name": "Baseline", "parties": summarize_shares(parties, base)}]
    for sid in ids:
        s = get_db()["scenarios"].find_one({"_id": ObjectId(sid)})
        if not s:
            continue
        r = run_scenario(s.get("spec", {}), as_of)
        columns.append({"name": s["name"], "parties": r.get("scenario", [])})
    return jsonify({"as_of": meta.get("as_of"), "columns": columns})


@api.get("/")
def dashboard():
    return render_template("dashboard.html")
