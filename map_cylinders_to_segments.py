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
    winner_counts_by_feature: Dict[int, int] = {}
    for feat in features:
        props = feat.properties
        center = np.array([props.x, props.y], dtype=float)
        radius = float(props.radius)
        indices = tree.query_ball_point(center, radius)
        idx = np.asarray(indices, dtype=np.int64)
        assert idx.size > 0, f"No points inside cylinder for feature {props.id}"
        segs = final_segs[idx]
        labels, counts = np.unique(segs, return_counts=True)
        argmax_idx = int(np.argmax(counts))
        winner = int(labels[argmax_idx])
        winner_count = int(counts[argmax_idx])
        mapping[props.id] = winner
        winner_counts_by_feature[props.id] = winner_count

    values = list(mapping.values())
    if len(set(values)) != len(values):
        seg_to_features: Dict[int, List[Tuple[int, int]]] = {}
        for feat_id, seg_id in mapping.items():
            count = winner_counts_by_feature[feat_id]
            if seg_id not in seg_to_features:
                seg_to_features[seg_id] = []
            seg_to_features[seg_id].append((feat_id, count))

        violating = {seg_id: pairs for seg_id, pairs in seg_to_features.items() if len(pairs) > 1}
        if violating:
            lines: List[str] = []
            for seg_id, pairs in sorted(violating.items(), key=lambda kv: kv[0]):
                pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)
                desc = ", ".join(f"feature {fid} count {cnt}" for (fid, cnt) in pairs_sorted)
                lines.append(f"segment {seg_id}: {desc}")
            logging.error("Duplicate segment ids detected in mapping:\n%s", "\n".join(lines))

    assert len(set(values)) == len(values), "Duplicate segment ids detected in mapping"
    return mapping


def _drop_non_winner_points_mask(
    las: laspy.LasData,
    features: List[Feature],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    For each feature's cylinder, keep points belonging to the most frequent final_segs label
    within that cylinder and drop the rest. Returns a boolean mask of points to keep and a map
    from dropped segment id to count of dropped points (unique points, no double counting).
    """
    assert "final_segs" in las.point_format.dimension_names, "Missing final_segs in LAS"

    points_xy = np.stack([np.asarray(las.x, dtype=float), np.asarray(las.y, dtype=float)], axis=1)
    tree = cKDTree(points_xy)
    final_segs = np.asarray(las.final_segs, dtype=np.int64)

    keep = np.ones(final_segs.shape[0], dtype=bool)
    dropped_by_seg: Dict[int, int] = {}

    for feat in features:
        props = feat.properties
        center = np.array([props.x, props.y], dtype=float)
        radius = float(props.radius)
        indices = tree.query_ball_point(center, radius)
        idx = np.asarray(indices, dtype=np.int64)
        assert idx.size > 0, f"No points inside cylinder for feature {props.id}"

        segs_in_cyl = final_segs[idx]
        labels, counts = np.unique(segs_in_cyl, return_counts=True)
        winner_label = int(labels[int(np.argmax(counts))])

        losers_local_mask = segs_in_cyl != np.int64(winner_label)
        if np.any(losers_local_mask):
            losers_idx = idx[losers_local_mask]
            # Avoid double counting across overlapping cylinders
            fresh_losers_idx = losers_idx[keep[losers_idx]]
            if fresh_losers_idx.size > 0:
                keep[fresh_losers_idx] = False
                losers_segs = final_segs[fresh_losers_idx]
                loser_labels, loser_counts = np.unique(losers_segs, return_counts=True)
                for lab, cnt in zip(loser_labels, loser_counts):
                    key = int(lab)
                    value = int(cnt)
                    if key in dropped_by_seg:
                        dropped_by_seg[key] = dropped_by_seg[key] + value
                    else:
                        dropped_by_seg[key] = value

    return keep, dropped_by_seg


def _output_filtered_path_for(las_path: Path) -> Path:
    return las_path.with_name(f"{las_path.stem}_filtered.las")


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

    # Drop non-winner points within each cylinder and write filtered output
    keep_mask, dropped = _drop_non_winner_points_mask(las, fc.features)
    total_dropped = int(np.count_nonzero(~keep_mask))

    out_path = _output_filtered_path_for(las_path)
    if out_path.exists():
        logging.info("Skipping (exists): %s", out_path)
        return out_path

    filtered = laspy.LasData(las.header)
    filtered.points = las.points[keep_mask]
    # Report dropped segments and totals
    if dropped:
        lines: List[str] = []
        for seg_id in sorted(dropped.keys()):
            lines.append(f"seg {seg_id}: dropped {dropped[seg_id]}")
        logging.info("Dropped points inside cylinders:\n%s", "\n".join(lines))
    logging.info("Total dropped points: %d", total_dropped)

    filtered.write(str(out_path))
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


