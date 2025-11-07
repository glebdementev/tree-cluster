from enum import Enum
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    clean = "clean"
    segment = "segment"
    all = "all"


class OutputMode(str, Enum):
    cleaned_tiles = "cleaned_tiles"
    segmented_largest_per_tile = "segmented_largest_per_tile"


class PipelineConfig(BaseModel):
    tile_size_m: float = Field(20.0, description="XY tile size in meters")
    tile_overlap_m: float = Field(1.0, description="XY overlap between tiles in meters")
    z_min_m: float = Field(0.1, description="Cut points with z < z_min_m")
    sor_neighbors: int = Field(10, description="NN for Statistical Outlier Removal")
    sor_std_ratio: float = Field(1.0, description="Std ratio for SOR thresholding")
    voxel_size_m: float = Field(0.02, description="Voxel size for precision and decimation (2 cm)")
    min_points_per_tile: int = Field(500, description="Skip tiles with fewer points than this")


