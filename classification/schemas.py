from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class Species(str, Enum):
    birch = "birch"
    cedar = "cedar"
    fir = "fir"
    larch = "larch"
    pine = "pine"
    spruce = "spruce"
    unknown = "unknown"


class SampleRecord(BaseModel):
    path: str
    species: Species
    num_points: int


