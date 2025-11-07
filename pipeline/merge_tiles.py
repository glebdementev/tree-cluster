import os
import logging
from typing import List, Tuple

import numpy as np
import laspy
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


logger = logging.getLogger(__name__)


def _stack_points_and_labels(paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, laspy.LasHeader]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    tile_ids: List[np.ndarray] = []
    template_header: laspy.LasHeader | None = None
    for tile_idx, p in enumerate(paths):
        las = laspy.read(p)
        if template_header is None:
            template_header = las.header
        if not hasattr(las, "final_segs"):
            raise ValueError(f"Segmented tile missing 'final_segs': {p}")
        xs.append(las.x.astype(float))
        ys.append(las.y.astype(float))
        zs.append(las.z.astype(float))
        labels.append(las.final_segs.astype(np.int64))
        tile_ids.append(np.full(len(las.x), tile_idx, dtype=np.int32))
    assert template_header is not None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    label = np.concatenate(labels)
    tile_id = np.concatenate(tile_ids)
    return np.stack([x, y, z], axis=1), label, tile_id, template_header


def _merge_label_overlaps(points_xyz: np.ndarray, labels: np.ndarray, tile_ids: np.ndarray, radius_m: float) -> np.ndarray:
    # Normalize labels to start at 0 and be contiguous, then offset is identity
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    if num_labels == 0:
        return labels

    tree = cKDTree(points_xyz[:, :2])
    # Find close point pairs (2D proximity) to detect duplicate trees across overlapping tiles
    pairs = tree.query_pairs(r=radius_m)
    if len(pairs) == 0:
        # Nothing to merge
        return inverse.astype(np.int64)

    pairs_arr = np.fromiter((i for ij in pairs for i in ij), dtype=np.int64)
    pairs_arr = pairs_arr.reshape(-1, 2)
    # Only consider pairs from different tiles (overlap zones)
    diff_tile = tile_ids[pairs_arr[:, 0]] != tile_ids[pairs_arr[:, 1]]
    if not np.any(diff_tile):
        return inverse.astype(np.int64)
    pairs_arr = pairs_arr[diff_tile]

    li = inverse[pairs_arr[:, 0]]
    lj = inverse[pairs_arr[:, 1]]
    diff_label = li != lj
    if not np.any(diff_label):
        return inverse.astype(np.int64)
    li = li[diff_label]
    lj = lj[diff_label]

    # Build label adjacency and compute connected components
    data = np.ones(len(li), dtype=np.int8)
    adj = csr_matrix((data, (li, lj)), shape=(num_labels, num_labels))
    # Make it undirected
    adj = adj + adj.T
    _, comp_ids = connected_components(csgraph=adj, directed=False, connection='weak', return_labels=True)

    # Map components to contiguous ids
    comp_unique, comp_new = np.unique(comp_ids, return_inverse=True)
    merged_labels = comp_new[inverse]
    return merged_labels.astype(np.int64)


def merge_segmented_tiles(segmented_tiles: List[str], output_path: str, merge_radius_m: float) -> str:
    if len(segmented_tiles) == 0:
        raise ValueError("No segmented tiles provided for merging")
    logger.info("Merging %d segmented tile(s) into: %s", len(segmented_tiles), output_path)

    points_xyz, labels, tile_ids, template_header = _stack_points_and_labels(segmented_tiles)
    logger.info("Stacked points: %d | initial unique labels: %d", len(points_xyz), len(np.unique(labels)))

    merged_labels = _merge_label_overlaps(points_xyz, labels, tile_ids, merge_radius_m)
    num_merged = len(np.unique(merged_labels))
    logger.info("Merged unique labels: %d (radius=%.3f m)", num_merged, merge_radius_m)

    # Scramble label ids to avoid spatially monotonic ids
    if num_merged > 0:
        rng = np.random.default_rng()
        permutation = rng.permutation(num_merged)
        merged_labels = permutation[merged_labels]
        logger.info("Scrambled label ids with random permutation")

    available_backends = list(laspy.LazBackend.detect_available())
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    las = laspy.create(file_version=template_header.version, point_format=template_header.point_format)
    las.header.scales = template_header.scales
    las.header.offsets = template_header.offsets
    if len(points_xyz) > 0:
        las.x = points_xyz[:, 0]
        las.y = points_xyz[:, 1]
        las.z = points_xyz[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
    las.final_segs = merged_labels.astype(np.int32)

    if output_path.lower().endswith('.laz') and len(available_backends) > 0:
        las.write(output_path, do_compress=True, laz_backend=available_backends[0])
        return output_path
    else:
        out_path = output_path[:-4] + ".las" if output_path.lower().endswith('.laz') else output_path
        las.write(out_path)
        return out_path


