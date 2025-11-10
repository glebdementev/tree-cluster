import logging
from pipeline.config import PipelineConfig, PipelineStage, OutputMode
from pipeline.run_pipeline import run_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pipeline.tiling").setLevel(logging.INFO)
    # Set your absolute paths here
    input_path = "dataset/summer_irkutsk2025/complete_CH435/prob1.las"
    clean_tiles_dir = "dataset/summer_irkutsk2025/complete_CH435/prob1_tiles"
    segmented_output_dir = "dataset/summer_irkutsk2025/complete_CH435/prob1_segmented"
    final_output_path = "dataset/summer_irkutsk2025/complete_CH435/prob1_merged.laz"

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
        input_path=input_path,
        clean_tiles_dir=clean_tiles_dir,
        segmented_output_dir=segmented_output_dir,
        stage=PipelineStage.all,
        output_mode=OutputMode.segmented_largest_per_tile,
        config=config,
        final_output_path=final_output_path,
    )


if __name__ == "__main__":
    main()


