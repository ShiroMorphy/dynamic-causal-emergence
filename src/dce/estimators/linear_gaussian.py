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
    compute_causal_spectrum,
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
        ridge_alpha: float = 0.01,
        confidence_delta: float = 0.05,
        selection_criterion: str = "raw",
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
        # "raw" (primary strict DCE) or "density" / "normalized" (secondary causal efficiency)
        if selection_criterion in ("normalized", "density"):
            self.selection_criterion = "density"
        else:
            self.selection_criterion = "raw"
        self.verbose = verbose
        
        self.projections_: Dict[int, List[np.ndarray]] = {}
        self.dce_raw_: Dict[int, np.ndarray] = {}
        self.dce_density_: Dict[int, np.ndarray] = {}
        self.dce_per_dim_: Dict[int, np.ndarray] = {}  # for backward compatibility
        self.normalized_emergence_: Optional[np.ndarray] = None
        self.optimal_dce_raw_: Optional[np.ndarray] = None
        self.optimal_dce_density_: Optional[np.ndarray] = None
        self.causal_dimension_raw_: Optional[np.ndarray] = None
        self.causal_dimension_density_: Optional[np.ndarray] = None
        self.causal_dimension_confset_: Optional[List[List[int]]] = None
        self.dcd_pr_: Optional[np.ndarray] = None
        self.dcd_entropy_: Optional[np.ndarray] = None
        self.causal_spectrum_: Optional[np.ndarray] = None
        self.is_oracle_: bool = False


    def fit_oracle(
        self,
        A_sequence: np.ndarray,
        Sigma_sequence: np.ndarray,
        state_cov_sequence: Optional[np.ndarray] = None,
        projections: Optional[Dict[int, Union[np.ndarray, List[np.ndarray]]]] = None
    ) -> "LocalLinearGaussianDCE":
        """
        Evaluate theoretical ground truth Dynamic Causal Emergence from exact known process matrices.
        
        Uses exact projected Markov dynamics:
            B_t = W_t^T A_t Sigma_{X, t} W_t (W_t^T Sigma_{X, t} W_t)^{-1}
            Sigma_{eta, t} = W_t^T (A_t Sigma_{X, t} A_t^T + Sigma_t) W_t - B_t (W_t^T Sigma_{X, t} W_t) B_t^T
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
        self.dcd_pr_ = np.zeros(T_trans, dtype=np.float64)
        self.dcd_entropy_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_spectrum_ = np.zeros((T_trans, p_dim), dtype=np.float64)
        self.q90_ = np.zeros(T_trans, dtype=np.int32)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_raw_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_density_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_per_dim_ = self.dce_raw_
        self.projections_ = {q: [] for q in self.macro_dims}
        
        for t in range(T_trans):
            A_t = A_seq[t]
            Sig_t = Sig_seq[t]
            
            # Resolve microscopic state covariance Sigma_{X, t}
            if state_cov_sequence is not None:
                Sig_X_t = state_cov_sequence[t] if state_cov_sequence.ndim == 3 else state_cov_sequence
            else:
                # Stationary Lyapunov approximation: Sig_X = A Sig_X A^T + Sig_t
                try:
                    from scipy.linalg import solve_discrete_lyapunov
                    Sig_X_t = solve_discrete_lyapunov(A_t, Sig_t)
                except Exception:
                    Sig_X_t = np.eye(p_dim)
                    
            # Symmetrize and regularize Sig_X_t
            Sig_X_t = 0.5 * (Sig_X_t + Sig_X_t.T) + 1e-6 * np.eye(p_dim)
            
            oracle_reg = min(self.ridge_alpha, 1e-6) if self.ridge_alpha > 0 else 0.0
            
            # 1. Exact Microscopic EI and Causal Spectrum
            micro_decomp = compute_gaussian_effective_information(
                A_t, Sig_t, intervention=self.intervention, regularization=oracle_reg
            )
            self.micro_ei_[t] = micro_decomp.effective_information
            spec_rep = compute_causal_spectrum(A_t, Sig_t, regularization=oracle_reg)
            self.dcd_pr_[t] = spec_rep.dcd_pr
            self.dcd_entropy_[t] = spec_rep.dcd_entropy
            self.causal_spectrum_[t] = spec_rep.spectrum
            
            # 2. Exact Projected Macro Dynamics
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    self.dce_raw_[q][t] = 0.0
                    self.dce_density_[q][t] = 0.0
                    self.projections_[q].append(np.eye(p_dim, q))
                    continue
                    
                # Determine projection matrix W_t (p, q)
                if projections is not None and q in projections:
                    proj = projections[q]
                    W_t = proj[t] if (isinstance(proj, list) or proj.ndim == 3) else proj
                else:
                    if spec_rep.projection_v is not None:
                        W_t = spec_rep.projection_v[:, :q]
                    else:
                        _, _, vt = np.linalg.svd(A_t, full_matrices=False)
                        W_t = vt[:q, :].T
                    
                self.projections_[q].append(W_t)
                
                # Projected macro state covariance: Cov(V_t) = W_t^T Sigma_{X, t} W_t
                cov_v_t = W_t.T @ Sig_X_t @ W_t + oracle_reg * np.eye(q)
                cov_v_t_cross = W_t.T @ A_t @ Sig_X_t @ W_t
                
                # Conditional linear projection dynamics: V_{t+1} = B_t V_t + eta_t
                try:
                    B_macro = (np.linalg.solve(cov_v_t, cov_v_t_cross.T)).T
                except np.linalg.LinAlgError:
                    B_macro = W_t.T @ A_t @ W_t
                    
                cov_v_next = W_t.T @ (A_t @ Sig_X_t @ A_t.T + Sig_t) @ W_t
                Sig_macro = cov_v_next - B_macro @ cov_v_t @ B_macro.T
                Sig_macro = 0.5 * (Sig_macro + Sig_macro.T) + max(oracle_reg, 1e-12) * np.eye(q)
                
                macro_decomp = compute_gaussian_effective_information(
                    B_macro, Sig_macro, intervention=self.intervention, regularization=oracle_reg
                )
                self.macro_ei_[q][t] = macro_decomp.effective_information
                self.dce_raw_[q][t] = macro_decomp.effective_information - self.micro_ei_[t]
                self.dce_density_[q][t] = (macro_decomp.effective_information / float(q)) - (self.micro_ei_[t] / float(p_dim))
                
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
        self.dcd_pr_ = np.zeros(T_trans, dtype=np.float64)
        self.dcd_entropy_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_spectrum_ = np.zeros((T_trans, p_dim), dtype=np.float64)
        self.q90_ = np.zeros(T_trans, dtype=np.int32)
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_raw_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_density_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_per_dim_ = self.dce_raw_
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
            sum_w = np.sum(w_eff)
            if sum_w > 1e-12:
                w_norm = w_eff / sum_w
                n_eff = float(1.0 / np.sum(w_norm ** 2))
            else:
                n_eff = float(len(eff_idx))
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
            spec_rep = compute_causal_spectrum(
                a_micro, sig_micro, regularization=self.ridge_alpha, n_eff=n_eff
            )
            self.dcd_pr_[t] = spec_rep.dcd_pr
            self.dcd_entropy_[t] = spec_rep.dcd_entropy
            self.causal_spectrum_[t] = spec_rep.spectrum
            
            # Fisher causal projection: W_t = V_t[:, :q] from SVD of M_t = L_t^{-1} A_t
            V_fisher = spec_rep.projection_v
            
            for q in self.macro_dims:
                if q >= p_dim:
                    self.macro_ei_[q][t] = self.micro_ei_[t]
                    self.dce_per_dim_[q][t] = 0.0
                    self.projections_[q].append(np.eye(p_dim, q))
                    continue
                    
                if V_fisher is not None:
                    phi_t = V_fisher[:, :q]  # (p, q) Fisher causal projection
                else:
                    phi_t = np.eye(p_dim, q)
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
                self.dce_raw_[q][t] = macro_decomp.effective_information - self.micro_ei_[t]
                self.dce_density_[q][t] = (macro_decomp.effective_information / float(q)) - (self.micro_ei_[t] / float(p_dim))
                self.dce_per_dim_[q][t] = self.dce_raw_[q][t]
                
        self._resolve_optimal_dimensions(p_dim)
        self.is_oracle_ = False
        self.is_fitted_ = True
        return self

    def _resolve_optimal_dimensions(self, p_dim: int) -> None:
        """Select optimal dimensions and compute uncertainty confidence sets."""
        T_trans = len(self.micro_ei_)
        self.emergence_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_dimension_ = np.zeros(T_trans, dtype=np.int32)
        self.normalized_emergence_ = np.zeros(T_trans, dtype=np.float64)
        self.optimal_dce_raw_ = np.zeros(T_trans, dtype=np.float64)
        self.optimal_dce_density_ = np.zeros(T_trans, dtype=np.float64)
        self.causal_dimension_raw_ = np.zeros(T_trans, dtype=np.int32)
        self.causal_dimension_density_ = np.zeros(T_trans, dtype=np.int32)
        self.causal_dimension_confset_ = []
        self.q90_ = np.zeros(T_trans, dtype=np.int32)
        
        for t in range(T_trans):
            tot_ei = float(np.sum(self.causal_spectrum_[t]))
            if tot_ei > 1e-12:
                cum_ei = np.cumsum(self.causal_spectrum_[t])
                idx = int(np.searchsorted(cum_ei, 0.90 * tot_ei))
                self.q90_[t] = min(p_dim, idx + 1)
            else:
                self.q90_[t] = 0
                
        q_candidates = sorted(self.macro_dims)
        
        for t in range(T_trans):
            raw_scores = [self.dce_raw_[q][t] for q in q_candidates]
            density_scores = [self.dce_density_[q][t] for q in q_candidates]
            
            best_raw_idx = int(np.argmax(raw_scores))
            best_density_idx = int(np.argmax(density_scores))
            
            self.causal_dimension_raw_[t] = q_candidates[best_raw_idx]
            self.causal_dimension_density_[t] = q_candidates[best_density_idx]
            self.optimal_dce_raw_[t] = raw_scores[best_raw_idx]
            self.optimal_dce_density_[t] = density_scores[best_density_idx]
            
            if self.selection_criterion == "density":
                self.causal_dimension_[t] = self.causal_dimension_density_[t]
                self.emergence_[t] = self.optimal_dce_density_[t]
                scores = density_scores
            else:
                self.causal_dimension_[t] = self.causal_dimension_raw_[t]
                self.emergence_[t] = self.optimal_dce_raw_[t]
                scores = raw_scores
                
            self.normalized_emergence_[t] = self.optimal_dce_density_[t]
            
            # Confidence set: dimensions within confidence_delta gap from max score
            max_score = scores[np.argmax(scores)]
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

