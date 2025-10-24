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


def center_points_xy(points_xyz: np.ndarray) -> np.ndarray:
    # Compute centering in float64 to avoid precision loss with large UTM coordinates
    orig_dtype = points_xyz.dtype
    pts64 = points_xyz.astype(np.float64, copy=False)
    centroid_xy = pts64[:, :2].mean(axis=0, keepdims=True)
    # Debug prints to inspect centroid and raw XY means
    print(
        f"center_points_xy: raw_xy_mean=({float(centroid_xy[0,0]):.6f}, {float(centroid_xy[0,1]):.6f})",
        flush=True,
    )
    centered64 = pts64.copy()
    centered64[:, 0] = centered64[:, 0] - centroid_xy[0, 0]
    centered64[:, 1] = centered64[:, 1] - centroid_xy[0, 1]
    centered_xy_mean = centered64[:, :2].mean(axis=0, keepdims=True)
    print(
        f"center_points_xy: centered_xy_mean=({float(centered_xy_mean[0,0]):.6f}, {float(centered_xy_mean[0,1]):.6f})",
        flush=True,
    )
    return centered64.astype(orig_dtype, copy=False)


def has_points_in_all_quadrants(points_xyz: np.ndarray) -> bool:
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    q1 = np.any((x > 0.0) & (y > 0.0))
    q2 = np.any((x > 0.0) & (y < 0.0))
    q3 = np.any((x < 0.0) & (y > 0.0))
    q4 = np.any((x < 0.0) & (y < 0.0))
    return bool(q1 and q2 and q3 and q4)


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
        # Debug prints: raw stats before centering
        raw_mean = pts.mean(axis=0)
        raw_min = pts.min(axis=0)
        raw_max = pts.max(axis=0)
        print(
            f"sample path={rec.path}\n"
            f"raw_xyz_mean=({float(raw_mean[0]):.6f}, {float(raw_mean[1]):.6f}, {float(raw_mean[2]):.6f})\n"
            f"raw_xyz_min=({float(raw_min[0]):.6f}, {float(raw_min[1]):.6f}, {float(raw_min[2]):.6f})\n"
            f"raw_xyz_max=({float(raw_max[0]):.6f}, {float(raw_max[1]):.6f}, {float(raw_max[2]):.6f})",
            flush=True,
        )
        pts_centered_full = center_points_xy(pts)
        # Debug prints: stats after XY centering
        cen_mean = pts_centered_full.mean(axis=0)
        cen_min = pts_centered_full.min(axis=0)
        cen_max = pts_centered_full.max(axis=0)
        print(
            f"after_center_xy_xyz_mean=({float(cen_mean[0]):.6f}, {float(cen_mean[1]):.6f}, {float(cen_mean[2]):.6f})\n"
            f"after_center_xy_xyz_min=({float(cen_min[0]):.6f}, {float(cen_min[1]):.6f}, {float(cen_min[2]):.6f})\n"
            f"after_center_xy_xyz_max=({float(cen_max[0]):.6f}, {float(cen_max[1]):.6f}, {float(cen_max[2]):.6f})",
            flush=True,
        )
        assert has_points_in_all_quadrants(pts_centered_full), f"points lack all quadrants: {rec.path}"
        pts_sampled = sample_points(pts_centered_full, self.points_per_cloud, self.rng)
        pts_centered = pts_sampled
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


