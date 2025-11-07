from pipeline.config import PipelineConfig, PipelineStage, OutputMode
from pipeline.run_pipeline import run_pipeline


def main() -> None:
    # Set your absolute paths here
    input_path = "dataset/sample/117.las"
    clean_tiles_dir = "dataset/sample/117_tiles"
    segmented_output_dir = "dataset/sample/117_segmented"

    config = PipelineConfig(
        tile_size_m=36.0,
        tile_overlap_m=1.0,
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
    )


if __name__ == "__main__":
    main()


