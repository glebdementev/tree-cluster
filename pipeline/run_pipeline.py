import os
import logging
from glob import glob
from typing import List, Optional

import laspy
import numpy as np

from .config import PipelineConfig, PipelineStage, OutputMode
from .z_filter import filter_below_z
from .precision import quantize_points
from .decimate import voxel_downsample
from .noise_filter import filter_statistical_outliers
from .tiling import (
    header_xy_bounds,
    data_xy_bounds,
    generate_tiles,
    collect_points_in_bbox,
    write_points_las_like,
)

from treeiso.largest_segment import process_las_file_largest
from treeiso.treeiso import write_point_ids_file, write_segment_labels_file
from .merge_tiles import merge_segmented_tiles


logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clean_and_tile_input(input_path: str, output_dir: str, config: PipelineConfig) -> List[str]:
    _ensure_dir(output_dir)
    written: List[str] = []

    if os.path.isfile(input_path) and input_path.lower().endswith((".las", ".laz")):
        logger.info("Cleaning and tiling single file: %s", input_path)
        written.extend(_clean_and_tile_file(input_path, output_dir, config))
        if len(written) == 0:
            raise ValueError(f"No tiles produced from input file: {input_path}")
        logger.info("Produced %d cleaned tile(s) from %s", len(written), input_path)
        return written

    if os.path.isdir(input_path):
        files = glob(os.path.join(input_path, "*.la[sz]"))
        logger.info("Discovered %d LAS/LAZ file(s) in directory: %s", len(files), input_path)
        if len(files) == 0:
            raise ValueError(f"No LAS/LAZ files found in directory: {input_path}")
        for f in files:
            logger.info("Processing file: %s", f)
            written.extend(_clean_and_tile_file(f, output_dir, config))
        if len(written) == 0:
            raise ValueError(f"No tiles produced from inputs in directory: {input_path}")
        logger.info("Produced %d cleaned tile(s) from directory: %s", len(written), input_path)
        return written

    raise FileNotFoundError(f"Input path is neither a LAS/LAZ file nor a directory: {input_path}")


def _clean_and_tile_file(file_path: str, output_dir: str, config: PipelineConfig) -> List[str]:
    tiles_written: List[str] = []
    # First pass: compute bounds (consumes iterator)
    with laspy.open(file_path) as reader:
        header = reader.header
        h_bounds = header_xy_bounds(header)
        d_bounds = data_xy_bounds(reader)
    logger.info(
        "Bounds comparison | header(xmin=%.3f,xmax=%.3f,ymin=%.3f,ymax=%.3f) vs data(xmin=%.3f,xmax=%.3f,ymin=%.3f,ymax=%.3f)",
        h_bounds[0], h_bounds[1], h_bounds[2], h_bounds[3],
        d_bounds[0], d_bounds[1], d_bounds[2], d_bounds[3],
    )

    # Second pass: reopen reader to collect points per tile
    bounds = d_bounds
    tiles = generate_tiles(bounds, config.tile_size_m, config.tile_overlap_m)
    logger.info(
        "Tiling file: %s | bounds(xmin=%.3f, xmax=%.3f, ymin=%.3f, ymax=%.3f) | tile_size=%.3f m | overlap=%.3f m | generated tiles=%d",
        file_path,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        config.tile_size_m,
        config.tile_overlap_m,
        len(tiles),
    )
    base = os.path.splitext(os.path.basename(file_path))[0]
    # Use header from the first pass for output; open a fresh reader per tile to avoid iterator exhaustion
    for i, bbox in enumerate(tiles):
        with laspy.open(file_path) as reader:
            header = reader.header
            pts = collect_points_in_bbox(reader, bbox)
            logger.info(
                "Tile %s[%04d]: bbox(x0=%.3f, x1=%.3f, y0=%.3f, y1=%.3f) | points=%d",
                base,
                i,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                len(pts),
            )
            if len(pts) == 0:
                logger.info("Tile %s[%04d] skipped: no points in bbox", base, i)
                continue

            pts = filter_below_z(pts, config.z_min_m)
            if len(pts) == 0:
                logger.info("Tile %s[%04d] skipped: all points below z_min=%.3f", base, i, config.z_min_m)
                continue

            pts = filter_statistical_outliers(pts, config.sor_neighbors, config.sor_std_ratio)
            if len(pts) == 0:
                logger.info(
                    "Tile %s[%04d] skipped: removed as outliers (neighbors=%d, std_ratio=%.3f)",
                    base,
                    i,
                    config.sor_neighbors,
                    config.sor_std_ratio,
                )
                continue

            pts = quantize_points(pts, config.voxel_size_m)
            pts = voxel_downsample(pts, config.voxel_size_m)
            if len(pts) < config.min_points_per_tile:
                logger.info(
                    "Tile %s[%04d] skipped: %d points < min_points_per_tile=%d after quantize+downsample (voxel=%.3f)",
                    base,
                    i,
                    len(pts),
                    config.min_points_per_tile,
                    config.voxel_size_m,
                )
                continue

            out_name = f"{base}_tile_{i:04d}.laz"
            out_path = os.path.join(output_dir, out_name)
            actual_path = write_points_las_like(header, out_path, pts)
            tiles_written.append(actual_path)
            logger.info(
                "Tile %s[%04d] written: %s | kept points=%d",
                base,
                i,
                actual_path,
                len(pts),
            )

    return tiles_written


def segment_tiles_largest(tiles: List[str], output_dir: str) -> List[str]:
    _ensure_dir(output_dir)
    outputs: List[str] = []
    logger.info("Segmenting %d tile(s) for largest clusters", len(tiles))
    for tile_path in tiles:
        base = os.path.splitext(os.path.basename(tile_path))[0]
        out_base = os.path.join(output_dir, base + "_seg_largest.laz")
        process_las_file_largest(tile_path, out_base)
        outputs.append(out_base if out_base.lower().endswith(".laz") else out_base[:-4] + ".las")
        logger.info("Segmented tile written: %s", outputs[-1])
    return outputs


def segment_tiles_normal(tiles: List[str], output_dir: str) -> List[str]:
    _ensure_dir(output_dir)
    outputs: List[str] = []
    logger.info("Segmenting %d tile(s) (full segmentation)", len(tiles))
    for tile_path in tiles:
        base = os.path.splitext(os.path.basename(tile_path))[0]
        out_base = os.path.join(output_dir, base + "_seg.laz")
        write_segment_labels_file(tile_path, out_base)
        outputs.append(out_base if out_base.lower().endswith(".laz") else out_base[:-4] + ".las")
        logger.info("Segmented tile written: %s", outputs[-1])
    return outputs


def run_pipeline(
    input_path: str,
    clean_tiles_dir: str,
    segmented_output_dir: str,
    stage: PipelineStage,
    output_mode: OutputMode,
    config: PipelineConfig,
    final_output_path: Optional[str] = None,
) -> None:
    tiles: List[str] = []
    if stage in (PipelineStage.clean, PipelineStage.all):
        logger.info("Stage: CLEAN | output_dir=%s", clean_tiles_dir)
        tiles = clean_and_tile_input(input_path, clean_tiles_dir, config)

    if stage in (PipelineStage.segment, PipelineStage.all):
        if output_mode == OutputMode.segmented_largest_per_tile:
            logger.info("Stage: SEGMENT (normal: per-point ids) | output_dir=%s", segmented_output_dir)
            if len(tiles) == 0:
                tiles = glob(os.path.join(clean_tiles_dir, "*.la[sz]"))
            logger.info("Found %d cleaned tile(s) to segment", len(tiles))
            if len(tiles) == 0:
                raise ValueError(f"No cleaned tiles found in: {clean_tiles_dir}")
            segmented_tiles = segment_tiles_normal(tiles, segmented_output_dir)
            if final_output_path is not None and len(final_output_path) > 0:
                merged_out = final_output_path
            else:
                base_name = os.path.splitext(os.path.basename(input_path if os.path.isfile(input_path) else clean_tiles_dir.rstrip(os.sep)))[0]
                merged_out = os.path.join(segmented_output_dir, f"{base_name}_merged.laz")
            merge_radius = max(0.05, config.voxel_size_m * 2.0)
            merged_path = merge_segmented_tiles(segmented_tiles, merged_out, merge_radius)
            logger.info("Merged segmented output: %s", merged_path)


