from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import laspy
import numpy as np
from pydantic import BaseModel
from typing_extensions import Literal
from scipy.spatial import cKDTree


class PointGeometry(BaseModel):
    type: Literal["Point"]
    coordinates: Tuple[float, float, float]


class FeatureProperties(BaseModel):
    id: int
    x: float
    y: float
    z: float | None = None
    radius: float


class Feature(BaseModel):
    type: Literal["Feature"]
    properties: FeatureProperties
    geometry: PointGeometry


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    features: List[Feature]


def _list_treeiso_files(root: Path) -> List[Path]:
    all_files = sorted(root.rglob("*_treeiso.las"))
    result: List[Path] = []
    for p in all_files:
        if any("complete" in parent.name for parent in p.parents):
            result.append(p)
    return result


def _find_geojson_for_las(las_path: Path) -> Path | None:
    candidates = sorted(las_path.parent.glob("*.geojson"))
    if len(candidates) == 1:
        return candidates[0]
    markup = [p for p in candidates if "markup" in p.name.lower()]
    if len(markup) == 1:
        return markup[0]
    return None


def _load_feature_collection(path: Path) -> FeatureCollection:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return FeatureCollection.model_validate(data)


def _build_feature_to_seg_map(las: laspy.LasData, features: List[Feature]) -> Dict[int, int]:
    assert "final_segs" in las.point_format.dimension_names, "Missing final_segs in LAS"

    points_xy = np.stack([np.asarray(las.x, dtype=float), np.asarray(las.y, dtype=float)], axis=1)
    tree = cKDTree(points_xy)
    final_segs = np.asarray(las.final_segs, dtype=np.int64)

    mapping: Dict[int, int] = {}
    for feat in features:
        props = feat.properties
        center = np.array([props.x, props.y], dtype=float)
        radius = float(props.radius)
        indices = tree.query_ball_point(center, radius)
        idx = np.asarray(indices, dtype=np.int64)
        assert idx.size > 0, f"No points inside cylinder for feature {props.id}"
        segs = final_segs[idx]
        labels, counts = np.unique(segs, return_counts=True)
        winner = int(labels[int(np.argmax(counts))])
        mapping[props.id] = winner

    values = list(mapping.values())
    assert len(set(values)) == len(values), "Duplicate segment ids detected in mapping"
    return mapping


def _apply_mapping_inplace(las: laspy.LasData, feature_to_seg: Dict[int, int]) -> None:
    old_to_new: Dict[int, int] = {seg: feat_id for (feat_id, seg) in feature_to_seg.items()}
    old = np.asarray(las.final_segs, dtype=np.int64)
    new = old.copy()
    for seg, feat_id in old_to_new.items():
        mask = old == np.int64(seg)
        if np.any(mask):
            new[mask] = np.int64(feat_id)
    las.final_segs = new.astype(np.int32)


def _output_path_for(las_path: Path) -> Path:
    return las_path.with_name(f"{las_path.stem}_ids.las")


def process_pair(las_path: Path, geojson_path: Path) -> Path:
    logging.info("Processing pair: %s | %s", las_path, geojson_path)

    fc = _load_feature_collection(geojson_path)
    las = laspy.read(str(las_path))
    mapping = _build_feature_to_seg_map(las, fc.features)

    out_path = _output_path_for(las_path)
    if out_path.exists():
        logging.info("Skipping (exists): %s", out_path)
        return out_path

    _apply_mapping_inplace(las, mapping)
    las.write(str(out_path))
    logging.info("Wrote: %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).resolve().parent
    dataset_root = repo_root / "dataset"
    if not dataset_root.exists():
        print(f"Dataset directory not found: {dataset_root}")
        return

    las_files = _list_treeiso_files(dataset_root)
    print(f"Found {len(las_files)} *_treeiso.las file(s) under {dataset_root}")

    for las_path in las_files:
        print(f"Processing input: {las_path}")
        geojson_path = _find_geojson_for_las(las_path)
        if geojson_path is None:
            print(f"Skipping (no matching .geojson in directory): {las_path.parent}")
            continue
        process_pair(las_path, geojson_path)


if __name__ == "__main__":
    main()


