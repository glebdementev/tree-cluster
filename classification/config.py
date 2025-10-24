from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from training.schemas import Visualizer


class TrainingConfig(BaseModel):
    dataset_root: Path
    points_per_cloud: int = 1024
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-3
    train_fraction: float = 0.85
    exclude_unknown: bool = True
    min_class_count: int = 5
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    visualize_one_sample_per_class: bool = False
    visualizer: Visualizer = Visualizer.plotly


class PointNetParams(BaseModel):
    conv_channels: tuple[int, int, int] = (64, 128, 256)
    fc_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.2


