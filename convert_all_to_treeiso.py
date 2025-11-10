from __future__ import annotations

from pathlib import Path
from typing import List
import shutil
import logging

from pipeline.config import PipelineConfig, PipelineStage, OutputMode
from pipeline.run_pipeline import run_pipeline


def list_las_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.las"))
    result: List[Path] = []
    for p in files:
        name = p.name
        # Skip already-converted outputs and temporary working directories
        if name.endswith("_treeiso.las"):
            continue
        if any(parent.name.endswith("_tiles_tmp") or parent.name.endswith("_seg_tmp") for parent in p.parents):
            continue
        result.append(p)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pipeline.tiling").setLevel(logging.INFO)

    repo_root = Path(__file__).resolve().parent
    dataset_root = repo_root / "dataset"
    if not dataset_root.exists():
        print(f"Dataset directory not found: {dataset_root}")
        return

    las_files = list_las_files(dataset_root)
    print(f"Found {len(las_files)} .las file(s) under {dataset_root}")

    for fpath in las_files:
        print(f"Processing input: {fpath}")
        clean_tiles_dir = fpath.parent / f"{fpath.stem}_tiles_tmp"
        segmented_output_dir = fpath.parent / f"{fpath.stem}_seg_tmp"
        final_output_path = fpath.with_name(f"{fpath.stem}_treeiso.las")

        if final_output_path.exists():
            print(f"Skipping (exists): {final_output_path}")
            continue

        config = PipelineConfig(
            tile_size_m=30.0,
            tile_overlap_m=2.0,
            z_min_m=0.1,
            sor_neighbors=10,
            sor_std_ratio=1.0,
            voxel_size_m=0.02,
            min_points_per_tile=500,
        )

        run_pipeline(
            input_path=str(fpath),
            clean_tiles_dir=str(clean_tiles_dir),
            segmented_output_dir=str(segmented_output_dir),
            stage=PipelineStage.all,
            output_mode=OutputMode.segmented_largest_per_tile,
            config=config,
            final_output_path=str(final_output_path),
        )

        if clean_tiles_dir.exists():
            shutil.rmtree(clean_tiles_dir)
        if segmented_output_dir.exists():
            shutil.rmtree(segmented_output_dir)
        print(f"Wrote: {final_output_path}")


if __name__ == "__main__":
    main()


