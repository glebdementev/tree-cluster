from typing import Tuple
import numpy as np


def voxel_downsample(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Voxel-grid downsampling. Keep one point per voxel.

    Returns downsampled points ordered by the first occurrence per voxel.
    """
    if len(points_xyz) == 0:
        return points_xyz
    grid = np.floor(points_xyz / voxel_size_m).astype(np.int32)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    return points_xyz[np.sort(unique_indices)]


