from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel


class ClusteringConfig(BaseModel):
    dataset_root: Path
    points_per_cloud: int = 1024
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    exclude_unknown: bool = True
    min_class_count: int = 5
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    num_clusters: int = 6
    embedding_dim: int = 256


class HeadParams(BaseModel):
    dropout: float = 0.1
