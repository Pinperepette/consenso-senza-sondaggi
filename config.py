"""Configurazione centrale del motore di stima del consenso.

Tutti i valori sono sovrascrivibili via variabili d'ambiente, così lo stesso
codice gira in locale, in test e in produzione senza modifiche.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class MongoConfig:
    uri: str = _env("CONSENSO_MONGO_URI", "mongodb://localhost:27017")
    db_name: str = _env("CONSENSO_DB", "consenso")


@dataclass(frozen=True)
class SourceConfig:
    """Endpoint reali delle fonti pubbliche italiane.

    Gli archivi storici del Ministero dell'Interno e l'open-data di Eligendo
    espongono i risultati per livello territoriale; ISTAT espone la demografia
    via SDMX. Gli URL base sono qui per essere aggiornati senza toccare i loader.
    """

    # Archivio storico delle elezioni (Min. Interno)
    eligendo_storico_base: str = _env(
        "CONSENSO_ELIGENDO_BASE", "https://elezionistorico.interno.gov.it"
    )
    # Portale Eligendo (open data / dataset scaricabili)
    eligendo_opendata_base: str = _env(
        "CONSENSO_ELIGENDO_OPENDATA", "https://eligendo.interno.gov.it"
    )
    # ISTAT SDMX
    istat_sdmx_base: str = _env(
        "CONSENSO_ISTAT_SDMX", "https://esploradati.istat.it/SDMXWS/rest"
    )
    # Cartella locale dove archiviare i payload grezzi scaricati (raw immutabile)
    raw_cache_dir: str = _env(
        "CONSENSO_RAW_CACHE", os.path.join(os.path.dirname(__file__), "data", "raw")
    )
    http_timeout: float = float(_env("CONSENSO_HTTP_TIMEOUT", "60"))
    http_retries: int = int(_env("CONSENSO_HTTP_RETRIES", "3"))
    user_agent: str = _env(
        "CONSENSO_UA",
        "consenso-research/1.0 (+public-data ETL; contact: research@example.org)",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Iperparametri di default del modello state-space gerarchico."""

    # Partito di riferimento per la trasformazione log-ratio (ALR).
    # Conviene una categoria stabile e sempre presente ("Altri").
    reference_party: str = _env("CONSENSO_REF_PARTY", "party:ALTRI")
    # Scala dell'innovazione del random walk per mese (deviazione std su scala log-ratio).
    rw_scale_per_month: float = float(_env("CONSENSO_RW_SCALE", "0.05"))
    # Iperparametro Dirichlet dei flussi (omogeneità territoriale dei trasferimenti).
    flow_concentration: float = float(_env("CONSENSO_FLOW_ALPHA", "20.0"))
    # Campionamento NUTS
    num_warmup: int = int(_env("CONSENSO_NUM_WARMUP", "1000"))
    num_samples: int = int(_env("CONSENSO_NUM_SAMPLES", "1000"))
    num_chains: int = int(_env("CONSENSO_NUM_CHAINS", "4"))
    rhat_threshold: float = float(_env("CONSENSO_RHAT_MAX", "1.05"))
    # Soglie per il calcolo di P(p > soglia) negli output.
    default_thresholds: tuple = (0.03, 0.04, 0.10, 0.15, 0.20, 0.30, 0.40)


# Tipi di elezione riconosciuti. Le "politiche" sono l'ancora (bias = 0).
ELECTION_TYPES = ("politiche", "europee", "regionali", "comunali", "referendum")
ANCHOR_ELECTION_TYPE = "politiche"

# Livelli territoriali (dal più fine al più aggregato).
GEO_LEVELS = ("sezione", "comune", "provincia", "regione", "nazione")


@dataclass(frozen=True)
class Config:
    mongo: MongoConfig = field(default_factory=MongoConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


CONFIG = Config()
