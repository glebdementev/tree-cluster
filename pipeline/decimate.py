import numpy as np

def voxel_downsample_indices(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Return indices of the first occurrence per voxel cell.

    The returned indices are sorted to keep original order stable among kept points.
    """
    if len(points_xyz) == 0:
        return np.empty((0,), dtype=int)
    grid = np.floor(points_xyz / voxel_size_m).astype(np.int32)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    return np.sort(unique_indices)


