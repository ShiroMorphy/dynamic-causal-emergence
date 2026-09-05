"""
Fast Linear SVD-based Dynamic Causal Emergence Estimator.

High-throughput analytical estimator suitable for online streaming and ultra-large microstates.
"""

from typing import Dict, List, Optional
import numpy as np

from dce.core.kernels import compute_temporal_weights
from dce.core.effective_info import (
    compute_gaussian_effective_information,
    estimate_local_gaussian_dynamics,
)
from dce.estimators.base import BaseDynamicCE


class LinearSVDDCE(BaseDynamicCE):
    """
    Linear Dynamic Causal Emergence via local Singular Value Decomposition.
    """
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False,
        ridge_alpha: float = 1e-4
    ) -> None:
        super().__init__(
            macro_dims=macro_dims or [1, 2, 4],
            bandwidth=bandwidth,
            kernel_type=kernel_type,
            causal_only=causal_only
        )
        self.ridge_alpha = ridge_alpha

    def fit(self, X: np.ndarray, y: Optional[None] = None) -> "LinearSVDDCE":
        X = np.asarray(X, dtype=np.float64)
        n_samples, p_dim = X.shape
        T_trans = n_samples - 1
        
        x_past = X[:-1]
        x_future = X[1:]
        
        self.micro_ei_ = np.zeros(T_trans, dtype=np.float64)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        
        for t in range(T_trans):
            weights = compute_temporal_weights(t, T_trans, self.bandwidth, self.kernel_type, self.causal_only)
            
            # Local micro transition
            a_micro, sigma_micro = estimate_local_gaussian_dynamics(x_past, x_future, weights, self.ridge_alpha)
            self.micro_ei_[t] = compute_gaussian_effective_information(a_micro, sigma_micro).effective_information
            
            # Dynamic projection via SVD on locally weighted covariance
            w_sqrt = np.sqrt(weights)[:, np.newaxis]
            x_w = x_past * w_sqrt
            _, _, vt = np.linalg.svd(x_w, full_matrices=False)
            
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    continue
                phi_t = vt[:q, :].T
                v_past = x_past @ phi_t
                v_future = x_future @ phi_t
                
                a_macro, sigma_macro = estimate_local_gaussian_dynamics(v_past, v_future, weights, self.ridge_alpha)
                self.macro_ei_[q][t] = compute_gaussian_effective_information(a_macro, sigma_macro).effective_information
                
        dce_mat = np.column_stack([self.macro_ei_[q] - self.micro_ei_ for q in self.macro_dims])
        self.emergence_ = np.max(dce_mat, axis=1)
        best_indices = np.argmax(dce_mat, axis=1)
        self.causal_dimension_ = np.array([self.macro_dims[i] for i in best_indices])
        self.is_fitted_ = True
        return self
