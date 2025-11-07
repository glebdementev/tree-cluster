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
    label_offset = 0
    for tile_idx, p in enumerate(paths):
        las = laspy.read(p)
        if template_header is None:
            template_header = las.header
        xs.append(np.asarray(las.x, dtype=float))
        ys.append(np.asarray(las.y, dtype=float))
        zs.append(np.asarray(las.z, dtype=float))
        tile_labels = np.asarray(las.final_segs, dtype=np.int64)
        unique_tile_labels, inverse_tile = np.unique(tile_labels, return_inverse=True)
        labels.append(inverse_tile.astype(np.int64) + label_offset)
        label_offset += unique_tile_labels.size
        tile_ids.append(np.full(len(las.x), tile_idx, dtype=np.int32))
    assert template_header is not None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    label = np.concatenate(labels)
    tile_id = np.concatenate(tile_ids)
    return np.stack([x, y, z], axis=1), label, tile_id, template_header


def _merge_label_overlaps(points_xyz: np.ndarray, labels: np.ndarray, tile_ids: np.ndarray, header: laspy.LasHeader) -> np.ndarray:
    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    if num_labels == 0 or points_xyz.shape[0] == 0:
        return labels

    # Convert to integer LAS coordinates for exact matching
    scales = np.asarray(header.scales, dtype=np.float64)
    offsets = np.asarray(header.offsets, dtype=np.float64)
    xyz_int = np.rint((points_xyz - offsets) / scales).astype(np.int64)

    xi = xyz_int[:, 0]
    yi = xyz_int[:, 1]
    zi = xyz_int[:, 2]

    order = np.lexsort((zi, yi, xi))
    xi = xi[order]
    yi = yi[order]
    zi = zi[order]
    tiles_sorted = tile_ids[order]
    labels_sorted = inverse_labels[order]

    if xi.size <= 1:
        return inverse_labels.astype(np.int64)

    eq_adj = (xi[1:] == xi[:-1]) & (yi[1:] == yi[:-1]) & (zi[1:] == zi[:-1])
    if not np.any(eq_adj):
        return inverse_labels.astype(np.int64)

    starts = np.empty(xi.size, dtype=bool)
    starts[0] = True
    starts[1:] = ~eq_adj
    start_idx = np.flatnonzero(starts)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = xi.size

    rows: list[int] = []
    cols: list[int] = []
    for s, e in zip(start_idx, end_idx):
        if e - s <= 1:
            continue
        # Require duplicates across different tiles
        tiles_here = tiles_sorted[s:e]
        if np.unique(tiles_here).size < 2:
            continue
        labs = labels_sorted[s:e]
        labs_unique = np.unique(labs)
        if labs_unique.size <= 1:
            continue
        base = labs_unique[0]
        others = labs_unique[1:]
        rows.extend(np.broadcast_to(base, others.shape).tolist())
        cols.extend(others.tolist())

    if len(rows) == 0:
        return inverse_labels.astype(np.int64)

    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(num_labels, num_labels))
    adj = adj + adj.T
    _, comp_ids = connected_components(csgraph=adj, directed=False, connection='weak', return_labels=True)
    _, comp_new = np.unique(comp_ids, return_inverse=True)
    merged_labels = comp_new[inverse_labels]
    return merged_labels.astype(np.int64)


def merge_segmented_tiles(segmented_tiles: List[str], output_path: str, merge_radius_m: float) -> str:
    if len(segmented_tiles) == 0:
        raise ValueError("No segmented tiles provided for merging")
    logger.info("Merging %d segmented tile(s) into: %s", len(segmented_tiles), output_path)

    points_xyz, labels, tile_ids, template_header = _stack_points_and_labels(segmented_tiles)
    logger.info("Stacked points: %d | initial unique labels: %d", len(points_xyz), len(np.unique(labels)))

    merged_labels = _merge_label_overlaps(points_xyz, labels, tile_ids, template_header)
    num_merged = len(np.unique(merged_labels))
    logger.info("Merged unique labels: %d (by identical coordinates)", num_merged)

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
    if "final_segs" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
    las.final_segs = merged_labels.astype(np.int32)

    if output_path.lower().endswith('.laz') and len(available_backends) > 0:
        las.write(output_path, do_compress=True, laz_backend=available_backends[0])
        return output_path
    else:
        out_path = output_path[:-4] + ".las" if output_path.lower().endswith('.laz') else output_path
        las.write(out_path)
        return out_path


