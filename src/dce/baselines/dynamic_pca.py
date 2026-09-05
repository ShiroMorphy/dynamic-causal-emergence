"""
Dynamic Principal Component Analysis (DPCA) Baseline.
"""

from typing import Tuple
import numpy as np
from dce.core.kernels import compute_temporal_weights


def compute_dynamic_pca(
    X: np.ndarray,
    n_components: int = 2,
    bandwidth: float = 24.0,
    kernel_type: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Dynamic PCA trajectories and explained variance ratios over time.
    
    Returns:
        (projections, explained_variance_ratios)
        projections: (T, n_components)
        explained_variance_ratios: (T, n_components)
    """
    n_samples, p_dim = X.shape
    projections = np.zeros((n_samples, n_components), dtype=np.float64)
    exp_var_ratios = np.zeros((n_samples, n_components), dtype=np.float64)
    
    for t in range(n_samples):
        weights = compute_temporal_weights(t, n_samples, bandwidth, kernel_type)
        w_sqrt = np.sqrt(weights)[:, np.newaxis]
        
        mean_t = np.sum(weights[:, np.newaxis] * X, axis=0)
        x_centered = (X - mean_t) * w_sqrt
        
        _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        var_total = np.sum(s ** 2) + 1e-10
        
        projections[t] = (X[t] - mean_t) @ vt[:n_components].T
        exp_var_ratios[t] = (s[:n_components] ** 2) / var_total
        
    return projections, exp_var_ratios
