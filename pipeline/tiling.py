from typing import Generator, Iterable, List, Tuple
import os
import logging
import numpy as np
import laspy

BBox = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax

logger = logging.getLogger(__name__)


def header_xy_bounds(header: laspy.LasHeader) -> BBox:
    xmin = float(header.mins[0])
    xmax = float(header.maxs[0])
    ymin = float(header.mins[1])
    ymax = float(header.maxs[1])
    logger.info(
        "Header XY bounds computed | xmin=%.6f xmax=%.6f ymin=%.6f ymax=%.6f | scales=(%.9f, %.9f) offsets=(%.6f, %.6f)",
        xmin,
        xmax,
        ymin,
        ymax,
        float(header.scales[0]),
        float(header.scales[1]),
        float(header.offsets[0]),
        float(header.offsets[1]),
    )
    return (xmin, xmax, ymin, ymax)


def data_xy_bounds(reader: laspy.LasReader) -> BBox:
    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")
    total = 0
    for chunk in reader.chunk_iterator(5_000_000):
        if len(chunk.x) == 0:
            continue
        total += len(chunk.x)
        cxmin = float(np.min(chunk.x))
        cxmax = float(np.max(chunk.x))
        cymin = float(np.min(chunk.y))
        cymax = float(np.max(chunk.y))
        if cxmin < xmin:
            xmin = cxmin
        if cxmax > xmax:
            xmax = cxmax
        if cymin < ymin:
            ymin = cymin
        if cymax > ymax:
            ymax = cymax
    logger.info(
        "Data XY bounds scanned over %d points | xmin=%.6f xmax=%.6f ymin=%.6f ymax=%.6f",
        total,
        xmin,
        xmax,
        ymin,
        ymax,
    )
    return (xmin, xmax, ymin, ymax)


def generate_tiles(bounds_xy: BBox, tile_size_m: float, overlap_m: float) -> List[BBox]:
    xmin, xmax, ymin, ymax = bounds_xy
    step = tile_size_m - overlap_m
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    if step <= 0:
        logger.warning(
            "Invalid tiling parameters: step<=0 | tile_size=%.6f overlap=%.6f",
            tile_size_m,
            overlap_m,
        )
        return [(xmin, xmin + tile_size_m, ymin, ymin + tile_size_m)]
    nx = int(np.ceil(dx / step))
    ny = int(np.ceil(dy / step))
    logger.info(
        "Tiling grid: dx=%.6f dy=%.6f | tile_size=%.6f overlap=%.6f step=%.6f | nx=%d ny=%d",
        dx,
        dy,
        tile_size_m,
        overlap_m,
        step,
        nx,
        ny,
    )
    tiles: List[BBox] = []
    for iy in range(ny):
        y0 = ymin + iy * step
        y1 = y0 + tile_size_m
        for ix in range(nx):
            x0 = xmin + ix * step
            x1 = x0 + tile_size_m
            tile = (x0, x1, y0, y1)
            tiles.append(tile)
            logger.info(
                "Tile[%d,%d]: x0=%.6f x1=%.6f y0=%.6f y1=%.6f",
                ix,
                iy,
                x0,
                x1,
                y0,
                y1,
            )
    return tiles


def mask_bbox_xy(xs: np.ndarray, ys: np.ndarray, bbox: BBox) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)


def collect_points_and_records_in_bbox(reader: laspy.LasReader, bbox: BBox) -> Tuple[np.ndarray, np.ndarray]:
    points_list: List[np.ndarray] = []
    records_list: List[np.ndarray] = []
    for chunk in reader.chunk_iterator(5_000_000):
        xs = chunk.x
        ys = chunk.y
        zs = chunk.z
        mask = mask_bbox_xy(xs, ys, bbox)
        if np.any(mask):
            pts = np.stack([xs[mask], ys[mask], zs[mask]], axis=1)
            points_list.append(pts)
            records_list.append(chunk.points[mask])
    if len(points_list) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.float64)
    return np.vstack(points_list), np.concatenate(records_list)


 


def write_points_with_attrs(
    template_header: laspy.LasHeader,
    output_path: str,
    points_structured: np.ndarray,
    xyz_override: np.ndarray | None = None,
) -> str:
    available_backends = list(laspy.LazBackend.detect_available())
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    las = laspy.create(file_version=template_header.version, point_format=template_header.point_format)
    las.header.scales = template_header.scales
    las.header.offsets = template_header.offsets
    if len(points_structured) > 0:
        las.points = points_structured
        if xyz_override is not None and len(xyz_override) > 0:
            las.x = xyz_override[:, 0]
            las.y = xyz_override[:, 1]
            las.z = xyz_override[:, 2]
    if output_path.lower().endswith(".laz") and len(available_backends) > 0:
        las.write(output_path, do_compress=True, laz_backend=available_backends[0])
        return output_path
    else:
        out_path = output_path[:-4] + ".las" if output_path.lower().endswith(".laz") else output_path
        las.write(out_path)
        return out_path


