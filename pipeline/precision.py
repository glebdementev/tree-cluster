import numpy as np


def quantize_points(points_xyz: np.ndarray, step_m: float) -> np.ndarray:
    """Snap points to a grid of size step_m.

    This reduces precision to the provided grid step (e.g., 0.02 m).
    """
    scaled = points_xyz / step_m
    rounded = np.round(scaled)
    return rounded * step_m


