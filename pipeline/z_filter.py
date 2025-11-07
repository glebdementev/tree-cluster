from typing import Tuple
import numpy as np


def filter_below_z(points_xyz: np.ndarray, z_min_m: float) -> np.ndarray:
    """Return only points with z >= z_min_m.

    points_xyz: (N, 3) array ordered as x, y, z
    """
    z_vals = points_xyz[:, 2]
    mask = z_vals >= z_min_m
    return points_xyz[mask]


