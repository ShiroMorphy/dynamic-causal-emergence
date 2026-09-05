"""
Causal Emergence 2.0 (CE 2.0) Multiscale Apportioning Framework.

Implements Erik Hoel's (2026) exact Causal Apportioning formulation across nested hierarchies:
    Micro (BAs) -> Meso-1 (RTOs) -> Meso-2 (Interconnections) -> Macro (US Grid).
    
Quantifies:
1. Causal Sufficiency (CS)
2. Causal Necessity (CN)
3. Apportioned Effective Information across hierarchical scales.
"""

from typing import Dict, List, Optional
import numpy as np

from dce.core.kernels import compute_temporal_weights
from dce.core.effective_info import (
    compute_gaussian_effective_information,
    estimate_local_gaussian_dynamics,
)


class MultiscaleCE2Apportioner:
    """
    Multiscale Causal Emergence 2.0 Apportioning Estimator.
    """
    def __init__(
        self,
        hierarchy_levels: Optional[List[str]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        ridge_alpha: float = 1e-4,
        eval_step: int = 1
    ) -> None:
        self.hierarchy_levels = hierarchy_levels or ["micro", "rto", "interconnection", "grid"]
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        self.ridge_alpha = ridge_alpha
        self.eval_step = eval_step
        
        # Results
        self.level_ei_: Dict[str, np.ndarray] = {}
        self.apportioned_causality_: Dict[str, np.ndarray] = {}
        self.causal_sufficiency_: Dict[str, np.ndarray] = {}
        self.causal_necessity_: Dict[str, np.ndarray] = {}
        self.total_causality_: Optional[np.ndarray] = None

    def fit(
        self,
        multiscale_states: Dict[str, np.ndarray]
    ) -> "MultiscaleCE2Apportioner":
        """
        Fit CE 2.0 apportioning across hierarchy representations.
        
        Args:
            multiscale_states: Dictionary mapping level names (e.g., 'micro', 'rto', 'interconnection', 'grid')
                               to time-series matrices of shape (T, d_level).
        """
        first_key = list(multiscale_states.keys())[0]
        T_trans = multiscale_states[first_key].shape[0] - 1
        eval_indices = list(range(0, T_trans, self.eval_step))
        if eval_indices[-1] != T_trans - 1:
            eval_indices.append(T_trans - 1)
        
        for level in self.hierarchy_levels:
            self.level_ei_[level] = np.zeros(T_trans, dtype=np.float64)
            self.causal_sufficiency_[level] = np.zeros(T_trans, dtype=np.float64)
            self.causal_necessity_[level] = np.zeros(T_trans, dtype=np.float64)
            
        sampled_ei = {lvl: np.zeros(len(eval_indices), dtype=np.float64) for lvl in self.hierarchy_levels}
        sampled_suff = {lvl: np.zeros(len(eval_indices), dtype=np.float64) for lvl in self.hierarchy_levels}
        sampled_nec = {lvl: np.zeros(len(eval_indices), dtype=np.float64) for lvl in self.hierarchy_levels}
            
        for idx, t in enumerate(eval_indices):
            weights = compute_temporal_weights(t, T_trans, self.bandwidth, self.kernel_type)
            eff_idx = np.where(weights > 1e-7)[0]
            if len(eff_idx) < 15:
                eff_idx = np.argsort(weights)[-15:]
                eff_idx = np.sort(eff_idx)
            w_eff = weights[eff_idx]
            
            for level in self.hierarchy_levels:
                data = multiscale_states[level]
                x_past_eff = data[:-1][eff_idx]
                x_fut_eff = data[1:][eff_idx]
                d_level = x_past_eff.shape[1]
                
                a_mat, sigma_mat = estimate_local_gaussian_dynamics(x_past_eff, x_fut_eff, w_eff, self.ridge_alpha)
                decomp = compute_gaussian_effective_information(a_mat, sigma_mat)
                
                # Dimension-normalized EI for valid cross-scale comparison
                sampled_ei[level][idx] = decomp.effective_information / float(d_level)
                sampled_suff[level][idx] = decomp.determinism / float(d_level)
                sampled_nec[level][idx] = max(0.0, (decomp.determinism - decomp.degeneracy) / float(d_level))
                
        # Interpolate across full timeline
        all_t = np.arange(T_trans)
        for level in self.hierarchy_levels:
            self.level_ei_[level] = np.interp(all_t, eval_indices, sampled_ei[level])
            self.causal_sufficiency_[level] = np.interp(all_t, eval_indices, sampled_suff[level])
            self.causal_necessity_[level] = np.interp(all_t, eval_indices, sampled_nec[level])
                
        # Apportion causality across hierarchy
        self.total_causality_ = np.zeros(T_trans, dtype=np.float64)
        for i, level in enumerate(self.hierarchy_levels):
            if i == 0:
                self.apportioned_causality_[level] = np.maximum(0.0, self.level_ei_[level])
            else:
                prev_level = self.hierarchy_levels[i - 1]
                gain = self.level_ei_[level] - self.level_ei_[prev_level]
                self.apportioned_causality_[level] = np.maximum(0.0, gain)
                
            self.total_causality_ += self.apportioned_causality_[level]
            
        return self
