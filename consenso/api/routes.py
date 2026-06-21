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
    proj = {"geo_id": 1, "share": 1, "votes": 1, "election_id": 1}
    if election == "all":
        # per ogni comune, il risultato piu' RECENTE del partito su tutte le tornate
        dates = {e["_id"]: e["date"] for e in
                 get_db()["elections"].find({"type": "comunali"}, {"date": 1})}
        best = {}
        for r in get_db()[PARTY_RESULTS].find(
                {"election_id": {"$in": list(dates)}, "party_id": party,
                 "share": {"$gt": 0}}, proj):
            d = dates.get(r["election_id"], "")
            if r["geo_id"] not in best or d > best[r["geo_id"]][0]:
                best[r["geo_id"]] = (d, r)
        rows = [r for _, r in best.values()]
    else:
        rows = get_db()[PARTY_RESULTS].find(
            {"election_id": election, "party_id": party, "share": {"$gt": 0}}, proj)
    pts = []
    for r in rows:
        c = coords.get(r["geo_id"])
        if not c:
            continue
        pts.append({"lat": c[0], "lng": c[1], "name": r["geo_id"].split(":", 2)[-1].title(),
                    "share": r["share"], "votes": r.get("votes", 0)})
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


@api.get("/swings")
def swings_route():
    """Sondaggi vs Urne: swing reale nei comuni vs swing dei sondaggi."""
    from consenso.model.swings import swings
    return jsonify(swings())


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
