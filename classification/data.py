from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
import laspy

from classification.schemas import Species, SampleRecord


LAS_SUFFIX = ".las"
SPECIES_NAMES = {s.value for s in Species}


def infer_species(path: Path) -> Species:
    parent = path.parent.name.lower()
    if parent in SPECIES_NAMES:
        return Species(parent)  # type: ignore[arg-type]
    return Species.unknown


def scan_records(root: Path) -> List[SampleRecord]:
    results: List[SampleRecord] = []
    for p in root.rglob(f"*{LAS_SUFFIX}"):
        if not p.is_file():
            continue
        with laspy.open(p) as las_reader:
            point_count = int(las_reader.header.point_count)
        if point_count <= 0:
            continue
        results.append(
            SampleRecord(
                path=str(p),
                species=infer_species(p),
                num_points=point_count,
            )
        )
    return results


def filter_records(records: List[SampleRecord], min_per_class: int, exclude_unknown: bool) -> List[SampleRecord]:
    counts = Counter(r.species for r in records)
    allowed_species = {
        sp
        for sp, cnt in counts.items()
        if cnt >= min_per_class and (sp != Species.unknown if exclude_unknown else True)
    }
    return [r for r in records if r.species in allowed_species]


def build_label_mapping(records: List[SampleRecord]) -> Dict[Species, int]:
    present = sorted({r.species for r in records}, key=lambda s: s.value)
    mapping: Dict[Species, int] = {sp: i for i, sp in enumerate(present)}
    return mapping


def read_points_xyz(file_path: Path) -> np.ndarray:
    with laspy.open(file_path) as las_reader:
        las = las_reader.read()
        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
    pts = np.stack([x, y, z], axis=1)
    return pts


def sample_points(points_xyz: np.ndarray, points_per_cloud: int, rng: np.random.RandomState) -> np.ndarray:
    n = int(points_xyz.shape[0])
    assert n > 0, "point cloud must contain at least one point"
    if n >= points_per_cloud:
        idx = rng.choice(n, size=points_per_cloud, replace=False)
    else:
        idx = rng.choice(n, size=points_per_cloud, replace=True)
    return points_xyz[idx]


def center_points(points_xyz: np.ndarray) -> np.ndarray:
    centroid = points_xyz.mean(axis=0, keepdims=True)
    return points_xyz - centroid


class LasPointCloudDataset(Dataset):
    def __init__(
        self,
        records: List[SampleRecord],
        points_per_cloud: int,
        species_to_index: Dict[Species, int],
        seed: int,
    ) -> None:
        self.records = records
        self.points_per_cloud = points_per_cloud
        self.species_to_index = species_to_index
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[index]
        pts = read_points_xyz(Path(rec.path))
        pts_sampled = sample_points(pts, self.points_per_cloud, self.rng)
        pts_centered = center_points(pts_sampled)
        x = torch.from_numpy(pts_centered.astype(np.float32))  # (P,3)
        y = torch.tensor(self.species_to_index[rec.species], dtype=torch.long)
        return x, y


def train_val_split(records: List[SampleRecord], train_fraction: float, seed: int) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    n = len(records)
    assert n > 0, "no records after filtering"
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * train_fraction)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records


