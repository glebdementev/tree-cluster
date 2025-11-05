from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import sys

import laspy
import numpy as np


@dataclass(frozen=True)
class VoxelizeConfig:
    repo_root: Path
    input_root: Path
    output_root: Path
    voxel_size_m: float


def list_all_las(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.las")]


def render_progress(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return "[{}] 0/0 0%".format(" " * width)
    ratio = current / total
    if ratio < 0:
        ratio = 0.0
    if ratio > 1:
        ratio = 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    return f"[{bar}] {current}/{total} {percent}%"


def voxel_downsample(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.stack([xs, ys, zs], axis=1).astype(np.float64)
    mins = coords.min(axis=0)
    ijk = np.floor((coords - mins) / voxel_size).astype(np.int64)

    unique_cells, inverse, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    num_vox = unique_cells.shape[0]

    sum_x = np.bincount(inverse, weights=coords[:, 0], minlength=num_vox)
    sum_y = np.bincount(inverse, weights=coords[:, 1], minlength=num_vox)
    sum_z = np.bincount(inverse, weights=coords[:, 2], minlength=num_vox)

    mean_x = sum_x / counts
    mean_y = sum_y / counts
    mean_z = sum_z / counts

    return mean_x.astype(np.float64), mean_y.astype(np.float64), mean_z.astype(np.float64)


def voxelize_file(input_path: Path, output_path: Path, voxel_size: float) -> Tuple[int, int]:
    source = laspy.read(input_path)
    xs = np.asarray(source.x, dtype=np.float64)
    ys = np.asarray(source.y, dtype=np.float64)
    zs = np.asarray(source.z, dtype=np.float64)

    vx, vy, vz = voxel_downsample(xs, ys, zs, voxel_size)

    header = laspy.LasHeader(
        point_format=source.header.point_format,
        version=source.header.version,
    )
    header.scales = source.header.scales
    header.offsets = source.header.offsets

    out = laspy.LasData(header)
    out.x = vx
    out.y = vy
    out.z = vz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_path))

    return xs.size, vx.size


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    input_root = repo_root / "las_cylinders_5m"
    output_root = repo_root / "las_cylinders_5m_voxel"
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = VoxelizeConfig(
        repo_root=repo_root,
        input_root=input_root,
        output_root=output_root,
        voxel_size_m=0.1,
    )

    if not cfg.input_root.exists():
        print(f"Input directory not found: {cfg.input_root}")
        return

    files = list_all_las(cfg.input_root)
    if not files:
        print("No LAS files found in", cfg.input_root)
        return

    outputs: List[Path] = []
    for f in files:
        rel = f.relative_to(cfg.input_root)
        outputs.append(cfg.output_root / rel)

    total = len(files)
    processed = 0
    total_in = 0
    total_out = 0

    sys.stdout.write(render_progress(processed, total))
    sys.stdout.flush()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(voxelize_file, inp, outp, cfg.voxel_size_m) for inp, outp in zip(files, outputs)]
        for fut in as_completed(futures):
            in_count, out_count = fut.result()
            total_in += in_count
            total_out += out_count
            processed += 1
            sys.stdout.write("\r" + render_progress(processed, total))
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    print("Input points:", int(total_in))
    print("Output points:", int(total_out))
    print("Wrote voxelized point clouds to:", cfg.output_root)


if __name__ == "__main__":
    main()


