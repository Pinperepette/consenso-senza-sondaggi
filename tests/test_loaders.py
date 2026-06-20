"""Il parsing dei CSV (formato Eligendo/Min. Interno) deve popolare i risultati
e l'affluenza, riconciliando le etichette di lista e mettendo le sconosciute in
coda di revisione."""
from consenso.db.client import get_db
from consenso.db.schema import (PARTY_RESULTS, RAW, RECONCILE_QUEUE, TURNOUT)
from consenso.etl.reconcile import register_alias, register_party
from consenso.etl.sources.eligendo import EligendoResultsLoader, register_election
from consenso.etl.sources.interno import TurnoutLoader

RESULTS_CSV = b"""CODICE_COMUNE;DESCR_LISTA;VOTI_LISTA;VOTI_VALIDI
058091;FRATELLI D'ITALIA;26000;100000
058091;PARTITO DEMOCRATICO;19000;100000
058091;LISTA SCONOSCIUTA;5000;100000
"""

TURNOUT_CSV = b"""CODICE_COMUNE;ELETTORI;VOTANTI;SCHEDE_BIANCHE;SCHEDE_NULLE
058091;160000;100000;1200;900
"""


def test_results_loader_reconciles_and_queues_unknown():
    register_party("party:FDI", "Fratelli d'Italia")
    register_party("party:PD", "Partito Democratico")
    register_alias("FRATELLI D'ITALIA", "party:FDI")
    register_alias("PARTITO DEMOCRATICO", "party:PD")
    register_election("elez:x", "politiche", "2022-09-25", {"level": "nazionale"})

    res = EligendoResultsLoader().run(
        content=RESULTS_CSV, election_id="elez:x", geo_level="comune")
    assert res["rows"] == 3
    assert res["unmatched_labels"] == 1

    db = get_db()
    fdi = db[PARTY_RESULTS].find_one({"election_id": "elez:x", "party_id": "party:FDI"})
    assert fdi and fdi["votes"] == 26000 and fdi["geo_id"] == "ISTAT:058091"
    # l'etichetta sconosciuta è in coda di revisione (non assegnata a caso)
    assert db[RECONCILE_QUEUE].count_documents({"status": "pending"}) == 1
    # il payload grezzo è stato archiviato (idempotenza)
    assert db[RAW].count_documents({"source": "eligendo_results"}) == 1


def test_results_loader_idempotent():
    register_election("elez:x", "politiche", "2022-09-25", {"level": "nazionale"})
    loader = EligendoResultsLoader()
    loader.run(content=RESULTS_CSV, election_id="elez:x", geo_level="comune")
    loader.run(content=RESULTS_CSV, election_id="elez:x", geo_level="comune")
    db = get_db()
    # stesso payload -> un solo raw, e i risultati non si duplicano
    assert db[RAW].count_documents({"source": "eligendo_results"}) == 1
    assert db[PARTY_RESULTS].count_documents({"election_id": "elez:x"}) == 3


def test_turnout_loader():
    register_election("elez:x", "politiche", "2022-09-25", {"level": "nazionale"})
    res = TurnoutLoader().run(content=TURNOUT_CSV, election_id="elez:x", geo_level="comune")
    assert res["areas"] == 1
    t = get_db()[TURNOUT].find_one({"election_id": "elez:x", "geo_id": "ISTAT:058091"})
    assert abs(t["turnout"] - 0.625) < 1e-6
    assert t["blank"] == 1200 and t["invalid"] == 900
