from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Callable
import sys

import laspy
from pydantic import BaseModel


class PointGeometry(BaseModel):
    type: Literal["Point"]
    coordinates: List[float]


class TreeProperties(BaseModel):
    id: int
    x: float
    y: float


class Feature(BaseModel):
    type: Literal["Feature"]
    properties: TreeProperties
    geometry: PointGeometry


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    features: List[Feature]


def find_single_las_in_directory(directory: Path) -> Path | None:
    las_files = [p for p in directory.glob("*.las")]
    return las_files[0] if len(las_files) == 1 else None


def extract_cylinders_for_geojson(
    geojson_path: Path,
    output_root: Path,
    dataset_root: Path,
    on_feature_processed: Callable[[int], None] | None = None,
) -> int:
    las_path = find_single_las_in_directory(geojson_path.parent)
    if las_path is None:
        print(f"Skipped (need exactly one .las in folder): {geojson_path}")
        return 0

    with geojson_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    fc = FeatureCollection.parse_obj(raw)

    source_las = laspy.read(las_path)
    xs = source_las.x
    ys = source_las.y

    rel_dir = geojson_path.parent.relative_to(dataset_root)
    base_out_dir = output_root / rel_dir / geojson_path.stem
    base_out_dir.mkdir(parents=True, exist_ok=True)

    radius_sq = 25.0

    processed_count = 0
    for feature in fc.features:
        x0 = feature.properties.x
        y0 = feature.properties.y

        dx = xs - x0
        dy = ys - y0
        mask = (dx * dx + dy * dy) <= radius_sq

        header = laspy.LasHeader(
            point_format=source_las.header.point_format,
            version=source_las.header.version,
        )
        header.scales = source_las.header.scales
        header.offsets = source_las.header.offsets
        subset = laspy.LasData(header)
        subset.points = source_las.points[mask]

        out_path = base_out_dir / f"id_{feature.properties.id}.las"
        subset.write(str(out_path))
        processed_count += 1
        if on_feature_processed is not None:
            on_feature_processed(1)
    return processed_count


def count_features_if_has_single_las(geojson_path: Path) -> int:
    las_path = find_single_las_in_directory(geojson_path.parent)
    if las_path is None:
        return 0
    with geojson_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    fc = FeatureCollection.parse_obj(raw)
    return len(fc.features)


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


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset"
    output_root = repo_root / "las_cylinders_5m"
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        print(f"Dataset directory not found: {dataset_root}")
        return

    geojson_files: List[Path] = []
    for path in dataset_root.rglob("*.geojson"):
        full_lower = str(path).lower()
        if "partial" in full_lower:
            continue
        geojson_files.append(path)

    total_features = 0
    for gj in geojson_files:
        total_features += count_features_if_has_single_las(gj)

    if total_features == 0:
        print("No eligible features found (no folders with exactly one .las).")
        return

    processed = 0
    sys.stdout.write(render_progress(processed, total_features))
    sys.stdout.flush()

    def on_progress(delta: int) -> None:
        nonlocal processed
        processed += delta
        sys.stdout.write("\r" + render_progress(processed, total_features))
        sys.stdout.flush()

    for gj in geojson_files:
        extract_cylinders_for_geojson(gj, output_root, dataset_root, on_feature_processed=on_progress)

    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()


