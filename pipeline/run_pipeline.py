import os
from glob import glob
from typing import List

import laspy
import numpy as np

from .config import PipelineConfig, PipelineStage, OutputMode
from .z_filter import filter_below_z
from .precision import quantize_points
from .decimate import voxel_downsample
from .noise_filter import filter_statistical_outliers
from .tiling import (
    header_xy_bounds,
    generate_tiles,
    collect_points_in_bbox,
    write_points_las_like,
)

from treeiso.treeiso import process_las_file_largest


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clean_and_tile_input(input_path: str, output_dir: str, config: PipelineConfig) -> List[str]:
    _ensure_dir(output_dir)
    written: List[str] = []

    if os.path.isfile(input_path) and input_path.lower().endswith((".las", ".laz")):
        written.extend(_clean_and_tile_file(input_path, output_dir, config))
        if len(written) == 0:
            raise ValueError(f"No tiles produced from input file: {input_path}")
        return written

    if os.path.isdir(input_path):
        files = glob(os.path.join(input_path, "*.la[sz]"))
        if len(files) == 0:
            raise ValueError(f"No LAS/LAZ files found in directory: {input_path}")
        for f in files:
            written.extend(_clean_and_tile_file(f, output_dir, config))
        if len(written) == 0:
            raise ValueError(f"No tiles produced from inputs in directory: {input_path}")
        return written

    raise FileNotFoundError(f"Input path is neither a LAS/LAZ file nor a directory: {input_path}")


def _clean_and_tile_file(file_path: str, output_dir: str, config: PipelineConfig) -> List[str]:
    tiles_written: List[str] = []
    with laspy.open(file_path) as reader:
        header = reader.header
        bounds = header_xy_bounds(header)
        tiles = generate_tiles(bounds, config.tile_size_m, config.tile_overlap_m)
        base = os.path.splitext(os.path.basename(file_path))[0]
        for i, bbox in enumerate(tiles):
            pts = collect_points_in_bbox(reader, bbox)
            if len(pts) == 0:
                continue

            pts = filter_below_z(pts, config.z_min_m)
            if len(pts) == 0:
                continue

            pts = filter_statistical_outliers(pts, config.sor_neighbors, config.sor_std_ratio)
            if len(pts) == 0:
                continue

            pts = quantize_points(pts, config.voxel_size_m)
            pts = voxel_downsample(pts, config.voxel_size_m)
            if len(pts) < config.min_points_per_tile:
                continue

            out_name = f"{base}_tile_{i:04d}.laz"
            out_path = os.path.join(output_dir, out_name)
            actual_path = write_points_las_like(header, out_path, pts)
            tiles_written.append(actual_path)

    return tiles_written


def segment_tiles_largest(tiles: List[str], output_dir: str) -> List[str]:
    _ensure_dir(output_dir)
    outputs: List[str] = []
    for tile_path in tiles:
        base = os.path.splitext(os.path.basename(tile_path))[0]
        out_base = os.path.join(output_dir, base + "_seg_largest.laz")
        process_las_file_largest(tile_path, out_base)
        outputs.append(out_base if out_base.lower().endswith(".laz") else out_base[:-4] + ".las")
    return outputs


def run_pipeline(
    input_path: str,
    clean_tiles_dir: str,
    segmented_output_dir: str,
    stage: PipelineStage,
    output_mode: OutputMode,
    config: PipelineConfig,
) -> None:
    tiles: List[str] = []
    if stage in (PipelineStage.clean, PipelineStage.all):
        tiles = clean_and_tile_input(input_path, clean_tiles_dir, config)

    if stage in (PipelineStage.segment, PipelineStage.all):
        if output_mode == OutputMode.segmented_largest_per_tile:
            if len(tiles) == 0:
                tiles = glob(os.path.join(clean_tiles_dir, "*.la[sz]"))
            if len(tiles) == 0:
                raise ValueError(f"No cleaned tiles found in: {clean_tiles_dir}")
            _ = segment_tiles_largest(tiles, segmented_output_dir)


