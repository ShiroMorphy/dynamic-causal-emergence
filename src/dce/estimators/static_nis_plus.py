"""
Windowed Static NIS+ Baseline Estimator (StaticNISPlusWindowed).

Applies static NIS+ independently in sliding temporal windows:
- Cold start at each window (no parameter warm-starting theta_t <- theta_{t-1}).
- Zero temporal regularization (eta_phi = 0, eta_f = 0).
- Fits independent static macroscopic models per window.

Serves as the critical empirical baseline to demonstrate the necessity and tracking advantage
of Dyn-NIS+'s joint temporal regularization, warm-start stability, and Procrustes alignment.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

from dce.core.kernels import compute_temporal_weights, audit_kernel_leakage
from dce.core.effective_info import compute_dce
from dce.estimators.base import BaseDynamicCE
from dce.estimators.dyn_nis import (
    DynNISModule,
    compute_intervention_consistent_ei,
    _robust_logdet,
)


class StaticNISPlusWindowed(BaseDynamicCE):
    """
    Sliding-Window Static NIS+ Baseline Estimator.
    """
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False,
        lambda_ei: float = 1.0,
        gamma_gauge: float = 0.1,
        hidden_dim: int = 32,
        steps_per_window: int = 15,
        eval_step: int = 1,
        lr: float = 2e-3,
        selection_criterion: str = "normalized",
        confidence_delta: float = 0.05,
        device: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        super().__init__(
            macro_dims=macro_dims or [1, 2, 4],
            bandwidth=bandwidth,
            kernel_type=kernel_type,
            causal_only=causal_only
        )
        self.lambda_ei = lambda_ei
        self.gamma_gauge = gamma_gauge
        self.hidden_dim = hidden_dim
        self.steps_per_window = steps_per_window
        self.eval_step = eval_step
        self.lr = lr
        self.selection_criterion = selection_criterion
        self.confidence_delta = confidence_delta
        self.verbose = verbose
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.dce_per_dim_: Dict[int, np.ndarray] = {}
        self.normalized_emergence_: Optional[np.ndarray] = None
        self.causal_dimension_confset_: Optional[List[List[int]]] = None

    def fit(self, X: np.ndarray, y: Optional[None] = None) -> "StaticNISPlusWindowed":
        X_np = np.asarray(X, dtype=np.float32)
        T_total, p_dim = X_np.shape
        T_trans = T_total - 1
        
        X_tensor = torch.from_numpy(X_np).to(self.device)
        x_past_all = X_tensor[:-1]
        x_future_all = X_tensor[1:]
        
        eval_indices = np.arange(0, T_trans, self.eval_step)
        if eval_indices[-1] != T_trans - 1:
            eval_indices = np.append(eval_indices, T_trans - 1)
            
        # Microscopic EI evaluation (baseline micro dynamics linear transition)
        from dce.core.effective_info import estimate_local_gaussian_dynamics, compute_gaussian_effective_information
        
        micro_ei_sampled = np.zeros(len(eval_indices), dtype=np.float64)
        for idx, t in enumerate(eval_indices):
            w = compute_temporal_weights(int(t), T_trans, self.bandwidth, self.kernel_type, self.causal_only)
            audit_kernel_leakage(w, center_t=int(t), causal_only=self.causal_only)
            
            a_micro, sig_micro = estimate_local_gaussian_dynamics(
                X_np[:-1], X_np[1:], w, ridge_alpha=1e-4
            )
            micro_decomp = compute_gaussian_effective_information(
                a_micro, sig_micro, intervention="gaussian", regularization=1e-4
            )
            micro_ei_sampled[idx] = micro_decomp.effective_information
            
        if len(eval_indices) > 1 and self.eval_step > 1:

            interp_func_micro = interp1d(eval_indices, micro_ei_sampled, kind="linear", fill_value="extrapolate")
            self.micro_ei_ = interp_func_micro(np.arange(T_trans))
        else:
            self.micro_ei_ = micro_ei_sampled
            
        self.macro_ei_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        self.dce_per_dim_ = {q: np.zeros(T_trans, dtype=np.float64) for q in self.macro_dims}
        
        for q in self.macro_dims:
            macro_ei_sampled_q = np.zeros(len(eval_indices), dtype=np.float64)
            
            for idx, t in enumerate(eval_indices):
                w = compute_temporal_weights(int(t), T_trans, self.bandwidth, self.kernel_type, self.causal_only)
                audit_kernel_leakage(w, center_t=int(t), causal_only=self.causal_only)
                w_t = torch.from_numpy(w.astype(np.float32)).to(self.device)
                
                active_mask = w_t > 1e-5
                if torch.sum(active_mask) < 8:
                    active_mask = torch.ones_like(w_t, dtype=torch.bool)
                    
                x_p = x_past_all[active_mask]
                x_f = x_future_all[active_mask]
                w_sub = w_t[active_mask]
                
                # Independent COLD START at every window: fresh initialization
                model = DynNISModule(p_dim, q, self.hidden_dim).to(self.device)
                optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
                
                model.train()
                for _ in range(self.steps_per_window):
                    optimizer.zero_grad()
                    v_p, v_pred, x_recon = model(x_p)
                    v_target = model.encoder(x_f)
                    
                    loss_latent = torch.sum(w_sub * torch.mean((v_target - v_pred) ** 2, dim=-1))
                    loss_recon = torch.sum(w_sub * torch.mean((x_f - x_recon) ** 2, dim=-1))
                    loss_pred = loss_latent + 0.3 * loss_recon
                    
                    ei_loss = compute_intervention_consistent_ei(model.dynamics, v_p, v_target, w_sub)
                    
                    v_cent = v_p - torch.mean(v_p, dim=0, keepdim=True)
                    cov_v = torch.mm(v_cent.t(), v_cent) / float(v_p.shape[0] - 1 + 1e-6)
                    loss_gauge = torch.norm(cov_v - torch.eye(q, device=self.device), p="fro") ** 2
                    
                    # No temporal continuity constraints (cold independent window)
                    total_loss = loss_pred - self.lambda_ei * ei_loss + self.gamma_gauge * loss_gauge
                    total_loss.backward()
                    optimizer.step()
                    
                model.eval()
                with torch.no_grad():
                    v_p, v_pred, _ = model(x_p)
                    v_target = model.encoder(x_f)
                    macro_ei_sampled_q[idx] = float(
                        compute_intervention_consistent_ei(model.dynamics, v_p, v_target, w_sub).item()
                    )
                    
            if len(eval_indices) > 1 and self.eval_step > 1:
                interp_func_macro = interp1d(eval_indices, macro_ei_sampled_q, kind="linear", fill_value="extrapolate")
                self.macro_ei_[q] = interp_func_macro(np.arange(T_trans))
            else:
                self.macro_ei_[q] = macro_ei_sampled_q
                
            self.dce_per_dim_[q] = self.macro_ei_[q] - self.micro_ei_
            
        self._resolve_optimal_dimensions(p_dim)
        self.is_fitted_ = True
        return self

    def _resolve_optimal_dimensions(self, p_dim: int) -> None:
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
            
            max_score = scores[best_idx]
            threshold = max_score - abs(self.confidence_delta)
            conf_set = [q for q, val in zip(q_candidates, scores) if val >= threshold]
            self.causal_dimension_confset_.append(conf_set)
