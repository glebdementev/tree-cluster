import numpy as np
from scipy.spatial import cKDTree


def filter_statistical_outliers_mask(points_xyz: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    """Return boolean mask of inliers using SOR criterion.

    Keeps points whose mean NN distance is within mean + std_ratio * std.
    """
    if len(points_xyz) == 0:
        return np.zeros((0,), dtype=bool)
    k = neighbors + 1
    k_eff = k if k <= len(points_xyz) else len(points_xyz)
    kdt = cKDTree(points_xyz)
    nn_dists, _ = kdt.query(points_xyz, k=k_eff)
    if k_eff > 1:
        d = nn_dists[:, 1:]
    else:
        d = nn_dists.reshape(-1, 1)
    mean_d = np.mean(d, axis=1)
    mu = float(np.mean(mean_d))
    sigma = float(np.std(mean_d))
    threshold = mu + std_ratio * sigma
    return mean_d <= threshold


