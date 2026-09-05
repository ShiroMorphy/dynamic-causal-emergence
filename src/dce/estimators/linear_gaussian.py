"""
Local Linear-Gaussian Reference Estimator and Exact Analytical Oracle for Dynamic Causal Emergence.

Implements exact closed-form Effective Information for continuous linear-Gaussian transitions:
    X_{t+1} = A_t * X_t + epsilon_t,  epsilon_t ~ N(0, Sigma_t)
    
Supports both:
1. Analytical Oracle Mode (fit_oracle): Evaluates ground truth theoretical emergence from known (A_t, Sigma_t).
2. Data-Driven Estimation Mode (fit): Estimates local (A_t, Sigma_t) via temporal kernel-weighted regression.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from dce.core.kernels import compute_temporal_weights, audit_kernel_leakage
from dce.core.effective_info import (
    compute_gaussian_effective_information,
    estimate_local_gaussian_dynamics,
    compute_dce,
)
from dce.core.interventions import BaseIntervention, get_intervention
from dce.estimators.base import BaseDynamicCE


class LocalLinearGaussianDCE(BaseDynamicCE):
    """
    Reference Linear-Gaussian Dynamic Causal Emergence Estimator.
    
    Provides mathematically exact Effective Information computation and unit-test oracle.
    """
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False,
        intervention: Union[str, BaseIntervention] = "gaussian",
        ridge_alpha: float = 1e-4,
        confidence_delta: float = 0.05,
        selection_criterion: str = "normalized",
        verbose: bool = False
    ) -> None:
        super().__init__(
            macro_dims=macro_dims or [1, 2, 4],
            bandwidth=bandwidth,
            kernel_type=kernel_type,
            causal_only=causal_only
        )
        self.intervention = intervention
        self.ridge_alpha = ridge_alpha
        self.confidence_delta = confidence_delta
        self.selection_criterion = selection_criterion
        self.verbose = verbose
        
        self.projections_: Dict[int, List[np.ndarray]] = {}
        self.dce_per_dim_: Dict[int, np.ndarray] = {}
        self.normalized_emergence_: Optional[np.ndarray] = None
        self.causal_dimension_confset_: Optional[List[List[int]]] = None
        self.is_oracle_: bool = False


    def fit_oracle(
        self,
        A_sequence: np.ndarray,
        Sigma_sequence: np.ndarray,
        projections: Optional[Dict[int, Union[np.ndarray, List[np.ndarray]]]] = None
    ) -> "LocalLinearGaussianDCE":
        """
        Evaluate theoretical ground truth Dynamic Causal Emergence from exact known process matrices.
        
        Args:
            A_sequence: (T, p, p) or (p, p) true transition matrices
            Sigma_sequence: (T, p, p) or (p, p) true noise covariance matrices
            projections: Optional pre-specified coarse-graining projections W of shape (p, q) or (T, p, q)
        """
        A_seq = np.asarray(A_sequence, dtype=np.float64)
        Sig_seq = np.asarray(Sigma_sequence, dtype=np.float64)
        
        if A_seq.ndim == 2:
            A_seq = A_seq[np.newaxis, ...]
        if Sig_seq.ndim == 2:
            Sig_seq = Sig_seq[np.newaxis, ...]
            
        T_trans = max(len(A_seq), len(Sig_seq))
        if len(A_seq) == 1 and T_trans > 1:
            A_seq = np.repeat(A_seq, T_trans, axis=0)
        if len(Sig_seq) == 1 and T_trans > 1:
            Sig_seq = np.repeat(Sig_seq, T_trans, axis=0)
            
        p_dim = A_seq.shape[1]
        
        self.micro_ei_ = np.zeros(T_trans, dtype=np.float64)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_per_dim_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.projections_ = {q: [] for q in self.macro_dims}
        
        for t in range(T_trans):
            A_t = A_seq[t]
            Sig_t = Sig_seq[t]
            
            # 1. Exact Microscopic EI
            micro_decomp = compute_gaussian_effective_information(
                A_t, Sig_t, intervention=self.intervention, regularization=self.ridge_alpha
            )
            self.micro_ei_[t] = micro_decomp.effective_information
            
            # 2. Exact Macro Projections
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    self.dce_per_dim_[q][t] = 0.0
                    self.projections_[q].append(np.eye(p_dim, q))
                    continue
                    
                # Determine projection matrix W_t (p, q)
                if projections is not None and q in projections:
                    proj = projections[q]
                    W_t = proj[t] if (isinstance(proj, list) or proj.ndim == 3) else proj
                else:
                    # Default canonical projection: principal causal sub-eigenspace of A_t
                    _, _, vt = np.linalg.svd(A_t, full_matrices=False)
                    W_t = vt[:q, :].T
                    
                self.projections_[q].append(W_t)
                
                # Exact projected macro-dynamics under V_t = W_t^T X_t
                # A_{macro} = W_t^T A_t W_t
                # Sigma_{macro} = W_t^T Sigma_t W_t
                A_macro = W_t.T @ A_t @ W_t
                Sig_macro = W_t.T @ Sig_t @ W_t
                
                macro_decomp = compute_gaussian_effective_information(
                    A_macro, Sig_macro, intervention=self.intervention, regularization=self.ridge_alpha
                )
                self.macro_ei_[q][t] = macro_decomp.effective_information
                self.dce_per_dim_[q][t] = compute_dce(
                    self.micro_ei_[t], self.macro_ei_[q][t], p_dim, q, normalized=False
                )
                
        self._resolve_optimal_dimensions(p_dim)
        self.is_oracle_ = True
        self.is_fitted_ = True
        return self

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> "LocalLinearGaussianDCE":
        """
        Estimate time-varying linear-Gaussian DCE from continuous observation matrix X of shape (T, p).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, p_dim = X.shape
        T_trans = n_samples - 1
        
        x_past = X[:-1]
        x_future = X[1:]
        
        self.micro_ei_ = np.zeros(T_trans, dtype=np.float64)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_per_dim_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.projections_ = {q: [] for q in self.macro_dims}
        
        for t in range(T_trans):
            # Compute temporal kernel weights
            weights = compute_temporal_weights(
                center_t=t,
                total_t=T_trans,
                bandwidth=self.bandwidth,
                kernel_type=self.kernel_type,
                causal_only=self.causal_only
            )
            audit_kernel_leakage(weights, center_t=t, causal_only=self.causal_only)
            
            # Filter non-negligible weights for local computation efficiency
            eff_idx = np.where(weights > 1e-7)[0]
            if len(eff_idx) < max(p_dim + 2, 10):
                eff_idx = np.argsort(weights)[-max(p_dim + 2, 10):]
                eff_idx = np.sort(eff_idx)
                
            w_eff = weights[eff_idx]
            x_past_eff = x_past[eff_idx]
            x_future_eff = x_future[eff_idx]
            
            # Estimate local microscopic dynamics
            a_micro, sig_micro = estimate_local_gaussian_dynamics(
                x_past_eff, x_future_eff, w_eff, ridge_alpha=self.ridge_alpha
            )
            micro_decomp = compute_gaussian_effective_information(
                a_micro, sig_micro, intervention=self.intervention, regularization=self.ridge_alpha
            )
            self.micro_ei_[t] = micro_decomp.effective_information
            
            # Dynamic projection via local weighted PCA/SVD
            w_sqrt = np.sqrt(w_eff)[:, np.newaxis]
            x_w_centered = (x_past_eff - np.sum(w_eff[:, np.newaxis] * x_past_eff, axis=0)) * w_sqrt
            _, _, vt = np.linalg.svd(x_w_centered, full_matrices=False)
            
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    self.dce_per_dim_[q][t] = 0.0
                    self.projections_[q].append(np.eye(p_dim, q))
                    continue
                    
                phi_t = vt[:q, :].T  # (p, q)
                self.projections_[q].append(phi_t)
                
                # Projected macro states
                v_past_eff = x_past_eff @ phi_t
                v_future_eff = x_future_eff @ phi_t
                
                a_macro, sig_macro = estimate_local_gaussian_dynamics(
                    v_past_eff, v_future_eff, w_eff, ridge_alpha=self.ridge_alpha
                )
                macro_decomp = compute_gaussian_effective_information(
                    a_macro, sig_macro, intervention=self.intervention, regularization=self.ridge_alpha
                )
                self.macro_ei_[q][t] = macro_decomp.effective_information
                self.dce_per_dim_[q][t] = compute_dce(
                    self.micro_ei_[t], self.macro_ei_[q][t], p_dim, q, normalized=False
                )
                
        self._resolve_optimal_dimensions(p_dim)
        self.is_oracle_ = False
        self.is_fitted_ = True
        return self

    def _resolve_optimal_dimensions(self, p_dim: int) -> None:
        """Select optimal dimension q_t^* and compute confidence sets."""
        T_trans = len(self.micro_ei_)
        self.emergence_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_dimension_ = np.zeros(T_trans, dtype=np.int32)
        self.normalized_emergence_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_dimension_confset_ = []
        
        q_candidates = sorted(self.macro_dims)
        
        for t in range(T_trans):
            if self.selection_criterion == "normalized":
                scores = [
                    compute_dce(self.micro_ei_[t], self.macro_ei_[q][t], p_dim, q, normalized=True)
                    for q in q_candidates
                ]
            else:
                scores = [self.dce_per_dim_[q][t] for q in q_candidates]
                
            best_idx = int(np.argmax(scores))
            best_q = q_candidates[best_idx]
            
            self.causal_dimension_[t] = best_q
            self.emergence_[t] = scores[best_idx]
            self.normalized_emergence_[t] = compute_dce(
                self.micro_ei_[t], self.macro_ei_[best_q][t], p_dim, best_q, normalized=True
            )
            
            # Confidence set: dimensions within confidence_delta gap from max score
            max_score = scores[best_idx]
            threshold = max_score - abs(self.confidence_delta)
            conf_set = [q for q, val in zip(q_candidates, scores) if val >= threshold]
            self.causal_dimension_confset_.append(conf_set)


    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        """Project input using time-varying optimal projections."""
        T_eval = min(len(X), len(self.causal_dimension_))
        outputs = []
        for t in range(T_eval):
            best_q = self.causal_dimension_[t]
            phi_t = self.projections_[best_q][t]
            outputs.append(X[t] @ phi_t)
        return np.array(outputs, dtype=object)
