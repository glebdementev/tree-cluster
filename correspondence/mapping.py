from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import laspy
import numpy as np
from scipy.spatial import cKDTree


#
# Configuration (edit these before running the script)
#

# Accepts absolute paths or glob patterns, relative to the repository root.
LAS_GLOB = r"dataset/435_tree_itc.laz"
GEOJSON_GLOB = (
    r"dataset/summer_irkutsk2025/complete_CH435/"
    r"complete_BUR1_CH435_PROB1_markup_GIRLS.geojson"
)

# Name of the LAS point attribute to use for matching (e.g. "final_segs").
FIELD_NAME = "pred_itc"


@dataclass
class PairCandidate:
    feature_id: int
    field_value: int
    count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_single_path(pattern: str) -> Path:
    """
    Resolve a single path from a glob pattern or direct path.
    Patterns are treated as relative to the repository root if not absolute.
    """
    root = _repo_root()
    pattern_path = Path(pattern)
    if not pattern_path.is_absolute():
        pattern_path = root / pattern_path

    # Use pathlib's glob on the parent if pattern contains wildcards
    if any(ch in pattern_path.name for ch in ["*", "?", "["]):
        parent = pattern_path.parent
        name_pattern = pattern_path.name
        matches = sorted(parent.glob(name_pattern))
    else:
        matches = [pattern_path] if pattern_path.exists() else []

    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    if len(matches) > 1:
        joined = "\n  ".join(str(m) for m in matches)
        raise RuntimeError(
            f"Expected a single file for pattern '{pattern}', "
            f"but found {len(matches)}:\n  {joined}"
        )
    return matches[0]


def _load_features(path: Path) -> List[dict]:
    """
    Load raw GeoJSON and return the list of feature dicts.
    Assumes a FeatureCollection with Point geometries and properties
    containing at least id, x, y, radius, and optional z.
    """
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"Unsupported GeoJSON structure in {path}")
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON 'features' must be a list in {path}")
    return features


def _load_las_field(las: laspy.LasData, field_name: str) -> np.ndarray:
    available = list(las.point_format.dimension_names)
    if field_name not in available:
        raise ValueError(
            f"Field '{field_name}' not found in LAS file. "
            f"Available dimensions: {', '.join(available)}"
        )
    arr = np.asarray(las[field_name])
    if arr.ndim != 1:
        raise ValueError(f"Field '{field_name}' must be 1D, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _build_cylinder_candidates(
    las: laspy.LasData,
    field_values: np.ndarray,
    features: Iterable[dict],
) -> Tuple[List[PairCandidate], Set[int], Set[int]]:
    """
    For each feature, build a vertical cylinder (radius from properties.radius,
    z from 0 to properties.z if present) and count points per unique field value.
    Returns all (feature_id, field_value, count) candidates with count > 0,
    along with the set of feature_ids and field_values that appear in at least
    one candidate.
    """
    xs = np.asarray(las.x, dtype=float)
    ys = np.asarray(las.y, dtype=float)
    zs = np.asarray(las.z, dtype=float)

    points_xy = np.stack([xs, ys], axis=1)
    tree = cKDTree(points_xy)

    candidates: List[PairCandidate] = []
    feature_ids: Set[int] = set()
    field_values_seen: Set[int] = set()

    for feat in features:
        props = feat.get("properties") or {}
        feature_id = int(props["id"])
        feature_ids.add(feature_id)

        center = np.array([float(props["x"]), float(props["y"])], dtype=float)
        radius = float(props["radius"])
        if not math.isfinite(radius) or radius <= 0:
            continue

        indices = tree.query_ball_point(center, radius)
        if not indices:
            continue
        idx = np.asarray(indices, dtype=np.int64)

        # Apply z filter: from 0 to feature["z"] (if present and not None)
        z_val = props.get("z", None)
        if z_val is not None:
            z_top = float(z_val)
            z_vals = zs[idx]
            mask_z = (z_vals >= 0.0) & (z_vals <= z_top)
            idx = idx[mask_z]
            if idx.size == 0:
                continue

        vals = field_values[idx]
        labels, counts = np.unique(vals, return_counts=True)
        for v, c in zip(labels, counts):
            iv = int(v)
            ic = int(c)
            if ic <= 0:
                continue
            candidates.append(PairCandidate(feature_id=feature_id, field_value=iv, count=ic))
            field_values_seen.add(iv)

    return candidates, feature_ids, field_values_seen


def _greedy_match(candidates: List[PairCandidate]) -> Tuple[List[PairCandidate], Set[int], Set[int]]:
    """
    Greedy 1:1 matching between feature_ids and field_values based on counts.
    Sorts all candidates by descending count and takes the first available pair
    where both feature_id and field_value are unused.
    """
    sorted_candidates = sorted(candidates, key=lambda c: c.count, reverse=True)

    used_features: Set[int] = set()
    used_field_values: Set[int] = set()
    assignments: List[PairCandidate] = []

    for cand in sorted_candidates:
        if cand.feature_id in used_features:
            continue
        if cand.field_value in used_field_values:
            continue
        assignments.append(cand)
        used_features.add(cand.feature_id)
        used_field_values.add(cand.field_value)

    return assignments, used_features, used_field_values


def main() -> None:
    repo_root = _repo_root()

    las_path = _resolve_single_path(LAS_GLOB)
    geojson_path = _resolve_single_path(GEOJSON_GLOB)

    print(f"Using LAS: {las_path}")
    print(f"Using GeoJSON: {geojson_path}")
    print(f"Using field: {FIELD_NAME}")

    las = laspy.read(str(las_path))
    field_values = _load_las_field(las, FIELD_NAME)

    features = _load_features(geojson_path)

    candidates, feature_ids, field_values_seen = _build_cylinder_candidates(
        las, field_values, features
    )

    if not candidates:
        print("No candidate feature/field pairs with points inside cylinders.")
        return

    assignments, used_features, used_field_values = _greedy_match(candidates)

    unmatched_features = sorted(fid for fid in feature_ids if fid not in used_features)
    unmatched_values = sorted(
        int(v) for v in field_values_seen if v not in used_field_values
    )

    # Prepare output
    output = {
        "las_path": str(las_path.relative_to(repo_root)) if las_path.is_relative_to(repo_root) else str(las_path),
        "geojson_path": str(geojson_path.relative_to(repo_root)) if geojson_path.is_relative_to(repo_root) else str(geojson_path),
        "field_name": FIELD_NAME,
        "assignments": [
            {
                "feature_id": a.feature_id,
                "field_value": a.field_value,
                "points_in_cylinder": a.count,
            }
            for a in assignments
        ],
        "unmatched_features": unmatched_features,
        "unmatched_field_values": unmatched_values,
    }

    out_path = repo_root / "correspondence" / "feature_field_pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote assignments to: {out_path}")
    print(f"Total assignments: {len(assignments)}")
    print(f"Unmatched features: {len(unmatched_features)}")
    print(f"Unmatched field values: {len(unmatched_values)}")


if __name__ == "__main__":
    main()


