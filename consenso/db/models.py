"""Modelli pydantic per validare i documenti prima della scrittura su Mongo.

Non sono ORM: servono come contratto/validazione. I loader costruiscono questi
modelli e poi chiamano ``.model_dump(by_alias=True)`` per ottenere il dict da
inserire.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from config import ELECTION_TYPES, GEO_LEVELS


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Meta(BaseModel):
    source: str
    ingested_at: datetime = Field(default_factory=_utcnow)
    source_hash: Optional[str] = None
    raw_ref: Optional[str] = None  # _id del documento raw da cui deriva (lineage)


class Geography(BaseModel):
    id: str = Field(alias="_id")            # es. "ISTAT:058091"
    level: str
    name: str
    parent: Optional[str] = None
    region: Optional[str] = None
    istat_code: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    centroid: Optional[Dict] = None          # GeoJSON Point
    meta: Optional[Meta] = Field(default=None, alias="_meta")

    @field_validator("level")
    @classmethod
    def _level_ok(cls, v: str) -> str:
        if v not in GEO_LEVELS:
            raise ValueError(f"livello geografico non valido: {v}")
        return v

    model_config = {"populate_by_name": True}


class Party(BaseModel):
    id: str = Field(alias="_id")            # es. "party:M5S"
    canonical_name: str
    family: Optional[str] = None
    active_from: Optional[str] = None
    active_to: Optional[str] = None
    model_config = {"populate_by_name": True}


class PartyAlias(BaseModel):
    raw_label: str
    party_id: str
    relation: str = "alias"                  # alias|merge|split|coalition_member|civic
    weight: float = 1.0
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: str = "manual"
    model_config = {"populate_by_name": True}


class Election(BaseModel):
    id: str = Field(alias="_id")
    type: str
    chamber: Optional[str] = None            # camera|senato|na
    date: str                                # ISO date
    scope: Dict                              # {"level": ..., "geo_ids": [...]}
    electoral_system: Optional[str] = None
    n_sezioni: Optional[int] = None
    meta: Optional[Meta] = Field(default=None, alias="_meta")

    @field_validator("type")
    @classmethod
    def _type_ok(cls, v: str) -> str:
        if v not in ELECTION_TYPES:
            raise ValueError(f"tipo elezione non valido: {v}")
        return v

    model_config = {"populate_by_name": True}


class PartyResult(BaseModel):
    election_id: str
    geo_id: str
    geo_level: str
    party_id: Optional[str] = None           # None finché non riconciliato
    raw_label: str
    votes: int
    valid_votes_area: int
    share: Optional[float] = None
    meta: Optional[Meta] = Field(default=None, alias="_meta")
    model_config = {"populate_by_name": True}


class Turnout(BaseModel):
    election_id: str
    geo_id: str
    geo_level: str
    eligible: int
    voters: int
    turnout: float
    blank: int = 0
    invalid: int = 0
    meta: Optional[Meta] = Field(default=None, alias="_meta")
    model_config = {"populate_by_name": True}


class Demographics(BaseModel):
    geo_id: str
    year: int
    population: Optional[int] = None
    median_age: Optional[float] = None
    age_bands: Optional[Dict[str, float]] = None
    income_avg: Optional[float] = None
    employment_rate: Optional[float] = None
    education: Optional[Dict[str, float]] = None
    urbanization: Optional[str] = None
    density: Optional[float] = None
    meta: Optional[Meta] = Field(default=None, alias="_meta")
    model_config = {"populate_by_name": True}


class ModelRun(BaseModel):
    id: str = Field(alias="_id")
    model_version: str
    code_hash: str
    input_data_hash: str
    hyperparams: Dict
    elections_used: List[str]
    inference: Dict = Field(default_factory=dict)   # method, draws, rhat_max, ess
    created_at: datetime = Field(default_factory=_utcnow)
    status: str = "running"                          # running|completed|failed
    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class Estimation(BaseModel):
    run_id: str
    party_id: str
    geo_id: str
    geo_level: str
    as_of: str
    mean: float
    median: float
    sd: float
    ci80: List[float]
    ci95: List[float]
    quantiles: Dict[str, float]
    prob_thresholds: Dict[str, float]
    prob_growth_6m: Optional[float] = None
    posterior_samples_ref: Optional[str] = None
    model_config = {"populate_by_name": True}
