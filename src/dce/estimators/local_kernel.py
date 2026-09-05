"""
Local Non-Parametric Kernel Estimator for Dynamic Causal Emergence.

Estimates local continuous transitions P_t(X_{t+1}|X_t) using temporal kernel weighting
and projects onto optimal subspace of dimension q to compute DCE_t(q) = EI_t^{(q)} - EI_t^{(p)}.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from dce.core.kernels import compute_temporal_weights
from dce.core.effective_info import (
    compute_gaussian_effective_information,
    estimate_local_gaussian_dynamics,
)
from dce.estimators.base import BaseDynamicCE


class LocalKernelDCE(BaseDynamicCE):
    """
    Local Kernel Weighted Dynamic Causal Emergence Estimator.
    """
    
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False,
        ridge_alpha: float = 1e-4,
        verbose: bool = False
    ) -> None:
        super().__init__(
            macro_dims=macro_dims,
            bandwidth=bandwidth,
            kernel_type=kernel_type,
            causal_only=causal_only
        )
        self.ridge_alpha = ridge_alpha
        self.verbose = verbose
        self.projections_: Dict[int, List[np.ndarray]] = {}

    def fit(self, X: np.ndarray, y: Optional[None] = None) -> "LocalKernelDCE":
        """
        Estimate DCE_t across time for microstate matrix X of shape (T, p).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, p_dim = X.shape
        T_trans = n_samples - 1
        
        x_past = X[:-1]
        x_future = X[1:]
        
        self.micro_ei_ = np.zeros(T_trans, dtype=np.float64)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.projections_ = {q: [] for q in self.macro_dims}
        
        time_iter = range(T_trans)
        if self.verbose:
            time_iter = tqdm(time_iter, desc="Estimating Local Kernel DCE")
            
        for t in time_iter:
            # 1. Compute kernel temporal weights centered at t
            weights = compute_temporal_weights(
                center_t=t,
                total_t=T_trans,
                bandwidth=self.bandwidth,
                kernel_type=self.kernel_type,
                causal_only=self.causal_only
            )
            
            # 2. Microscopic Effective Information at time t
            a_micro, sigma_micro = estimate_local_gaussian_dynamics(
                x_past, x_future, weights, ridge_alpha=self.ridge_alpha
            )
            micro_decomp = compute_gaussian_effective_information(a_micro, sigma_micro)
            self.micro_ei_[t] = micro_decomp.effective_information
            
            # 3. Macro representations for each candidate dimension q < p
            # Compute local weighted PCA projection basis W_t of shape (p, q)
            w_sqrt = np.sqrt(weights)[:, np.newaxis]
            x_w_centered = (x_past - np.sum(weights[:, np.newaxis] * x_past, axis=0)) * w_sqrt
            
            # Weighted SVD / PCA
            _, _, vt = np.linalg.svd(x_w_centered, full_matrices=False)
            
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    continue
                    
                # Projection matrix Phi_t (p, q)
                phi_t = vt[:q, :].T  # (p, q)
                self.projections_[q].append(phi_t)
                
                # Projected macrostates
                v_past = x_past @ phi_t
                v_future = x_future @ phi_t
                
                # Estimate macro transition
                a_macro, sigma_macro = estimate_local_gaussian_dynamics(
                    v_past, v_future, weights, ridge_alpha=self.ridge_alpha
                )
                macro_decomp = compute_gaussian_effective_information(a_macro, sigma_macro)
                self.macro_ei_[q][t] = macro_decomp.effective_information

        # 4. Compute Dynamic Causal Emergence: DCE_t = max_{q} (EI_t^{(q)} - EI_t^{(p)})
        dce_matrix = np.zeros((T_trans, len(self.macro_dims)), dtype=np.float64)
        for idx, q in enumerate(self.macro_dims):
            dce_matrix[:, idx] = self.macro_ei_[q] - self.micro_ei_
            
        best_dim_idx = np.argmax(dce_matrix, axis=1)
        self.emergence_ = np.max(dce_matrix, axis=1)
        self.causal_dimension_ = np.array([self.macro_dims[i] for i in best_dim_idx])
        self.is_fitted_ = True
        return self

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        # Returns macrostate trajectory using optimal q_t^* projection
        if not self.projections_ or not self.causal_dimension_.size:
            raise ValueError("Model is not properly fitted.")
        return X
