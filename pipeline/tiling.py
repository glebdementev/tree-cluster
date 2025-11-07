from typing import Generator, Iterable, List, Tuple
import os
import numpy as np
import laspy


BBox = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def header_xy_bounds(header: laspy.LasHeader) -> BBox:
    return (float(header.mins[0]), float(header.maxs[0]), float(header.mins[1]), float(header.maxs[1]))


def generate_tiles(bounds_xy: BBox, tile_size_m: float, overlap_m: float) -> List[BBox]:
    xmin, xmax, ymin, ymax = bounds_xy
    step = tile_size_m - overlap_m
    nx = int(np.ceil((xmax - xmin) / step))
    ny = int(np.ceil((ymax - ymin) / step))
    tiles: List[BBox] = []
    for iy in range(ny):
        y0 = ymin + iy * step
        y1 = y0 + tile_size_m
        for ix in range(nx):
            x0 = xmin + ix * step
            x1 = x0 + tile_size_m
            tiles.append((x0, x1, y0, y1))
    return tiles


def mask_bbox_xy(xs: np.ndarray, ys: np.ndarray, bbox: BBox) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)


def collect_points_in_bbox(reader: laspy.LasReader, bbox: BBox) -> np.ndarray:
    points_list: List[np.ndarray] = []
    for chunk in reader.chunk_iterator(5_000_000):
        xs = chunk.x
        ys = chunk.y
        zs = chunk.z
        mask = mask_bbox_xy(xs, ys, bbox)
        if np.any(mask):
            pts = np.stack([xs[mask], ys[mask], zs[mask]], axis=1)
            points_list.append(pts)
    if len(points_list) == 0:
        return np.empty((0, 3), dtype=float)
    return np.vstack(points_list)


def write_points_las_like(template_header: laspy.LasHeader, output_path: str, points_xyz: np.ndarray) -> str:
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
    if output_path.lower().endswith(".laz") and len(available_backends) > 0:
        las.write(output_path, do_compress=True, laz_backend=available_backends[0])
        return output_path
    else:
        out_path = output_path[:-4] + ".las" if output_path.lower().endswith(".laz") else output_path
        las.write(out_path)
        return out_path


