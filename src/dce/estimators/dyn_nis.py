"""
Dynamic Neural Information Slicing (Dyn-NIS+) Estimator (Q1 Architecture).

Implements:
1. Continuous neural coarse-graining encoder phi_{t,q}: R^p -> R^q.
2. Latent forward dynamics predictor f_t: R^q -> R^q.
3. Auxiliary state reconstruction decoder g_t: R^q -> R^p.
4. Intervention-consistent differentiable Effective Information via Monte Carlo interventional simulation do(V ~ N(0, I_q)).
5. Function-space temporal smoothness via orthogonal Procrustes representation alignment.
6. Gauge whitening penalty to eliminate representation scale gaming.
7. Held-out evaluation across temporal transitions to avoid winner's curse in causal dimension selection q_t^*.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from tqdm import tqdm

from dce.core.kernels import compute_temporal_weights, audit_kernel_leakage
from dce.core.effective_info import compute_dce
from dce.estimators.base import BaseDynamicCE


class DynamicEncoder(nn.Module):
    """Continuous neural coarse-graining projection phi: R^p -> R^q."""
    def __init__(self, p_dim: int, q_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, q_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDynamics(nn.Module):
    """Latent macroscopic transition predictor f: R^q -> R^q."""
    def __init__(self, q_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, q_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v + self.net(v)


class DynamicDecoder(nn.Module):
    """Auxiliary reconstruction decoder g: R^q -> R^p."""
    def __init__(self, q_dim: int, p_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, p_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v)


class DynNISModule(nn.Module):
    """Coupled Dyn-NIS+ architecture."""
    def __init__(self, p_dim: int, q_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.encoder = DynamicEncoder(p_dim, q_dim, hidden_dim)
        self.dynamics = LatentDynamics(q_dim, max(16, hidden_dim // 2))
        self.decoder = DynamicDecoder(q_dim, p_dim, hidden_dim)

    def forward(self, x_past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_past = self.encoder(x_past)
        v_pred = self.dynamics(v_past)
        x_recon = self.decoder(v_pred)
        return v_past, v_pred, x_recon


def _robust_logdet(cov_mat: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """Compute differentiable log-determinant via slogdet with positive-definite ridge."""
    d = cov_mat.shape[0]
    device = cov_mat.device
    sym = 0.5 * (cov_mat + cov_mat.t()) + reg * torch.eye(d, device=device, dtype=cov_mat.dtype)
    
    # Check on CPU if on MPS due to slogdet stability in older MPS runtimes
    if device.type == "mps":
        sym_cpu = sym.cpu()
        sign, logdet = torch.linalg.slogdet(sym_cpu)
        fallback = torch.tensor(d * np.log(reg), dtype=sym_cpu.dtype)
        logdet_safe = torch.where(sign > 0, logdet, fallback)
        return logdet_safe.to(device)
    else:
        sign, logdet = torch.linalg.slogdet(sym)
        fallback = torch.tensor(d * np.log(reg), device=device, dtype=sym.dtype)
        return torch.where(sign > 0, logdet, fallback)


def compute_intervention_consistent_ei(
    dynamics_fn: nn.Module,
    v_past: torch.Tensor,
    v_target: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int = 128,
    reg: float = 1e-4
) -> torch.Tensor:
    """
    Compute differentiable, intervention-consistent Effective Information:
        EI = 0.5 * [ ln det(Sigma_{out}) - ln det(Sigma_res) ]
        
    Simulates do(V ~ N(0, I_q)), passing through dynamics_fn to estimate interventional dispersion.
    """
    q_dim = v_past.shape[1]
    device = v_past.device
    w = weights / (torch.sum(weights) + 1e-8)
    
    # 1. Residual noise covariance Sigma_res from transition error
    residuals = v_target - dynamics_fn(v_past)
    res_mean = torch.sum(w.unsqueeze(1) * residuals, dim=0, keepdim=True)
    res_cent = residuals - res_mean
    cov_res = torch.mm((res_cent * w.unsqueeze(1)).t(), res_cent) + reg * torch.eye(q_dim, device=device)
    
    # 2. Interventional dispersion via Monte Carlo samples do(Z ~ N(0, I_q))
    z_interv = torch.randn(n_samples, q_dim, device=device)
    z_pred = dynamics_fn(z_interv)
    z_cent = z_pred - torch.mean(z_pred, dim=0, keepdim=True)
    cov_interv_dyn = torch.mm(z_cent.t(), z_cent) / float(n_samples - 1)
    
    # Total interventional output covariance: Sigma_{out} = Cov(f(Z)) + Sigma_res
    cov_out = cov_interv_dyn + cov_res
    
    logdet_out = _robust_logdet(cov_out, reg=reg)
    logdet_res = _robust_logdet(cov_res, reg=reg)
    
    ei = 0.5 * (logdet_out - logdet_res)
    return ei


def compute_procrustes_alignment(
    v_curr: torch.Tensor,
    v_prev: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute orthogonal Procrustes distance between consecutive macro-representations:
        R_phi = sum_s w_s || V_t[s] - V_{t-1}[s] * R^T ||^2
    where R = U V^T from SVD(V_t^T W V_{t-1}).
    Invariance to coordinate rotations or latent neuron permutations.
    """
    w = weights / (torch.sum(weights) + 1e-8)
    v_curr_w = v_curr * torch.sqrt(w).unsqueeze(1)
    v_prev_w = v_prev * torch.sqrt(w).unsqueeze(1)
    
    # Cross-covariance matrix M = V_t^T W V_{t-1}
    cross_cov = torch.mm(v_curr_w.t(), v_prev_w)
    
    with torch.no_grad():
        try:
            cov_cpu = cross_cov.cpu()
            U, _, Vh = torch.linalg.svd(cov_cpu)
            R = torch.mm(U, Vh).to(v_curr.device)
        except Exception:
            R = torch.eye(v_curr.shape[1], device=v_curr.device)
            
    v_prev_aligned = torch.mm(v_prev, R.t())
    diff = v_curr - v_prev_aligned
    loss_procrustes = torch.sum(w.unsqueeze(1) * (diff ** 2))
    return loss_procrustes


class DynamicNIS(BaseDynamicCE):
    """
    Dynamic Neural Information Slicing (Dyn-NIS+) Estimator.
    
    Jointly optimizes time-varying macroscopic representations phi_t, latent forward dynamics f_t,
    and causal scale q_t^* with representation-space temporal continuity and gauge regularization.
    """
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False,
        lambda_ei: float = 1.0,
        eta_phi: float = 0.5,
        eta_f: float = 0.2,
        gamma_gauge: float = 0.1,
        hidden_dim: int = 48,
        steps_per_window: int = 10,
        eval_step: int = 1,
        lr: float = 1e-3,
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
        self.eta_phi = eta_phi
        self.eta_f = eta_f
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
            
        self.models_: Dict[int, List[DynNISModule]] = {}
        self.dce_per_dim_: Dict[int, np.ndarray] = {}
        self.normalized_emergence_: Optional[np.ndarray] = None
        self.causal_dimension_confset_: Optional[List[List[int]]] = None

    def fit(self, X: np.ndarray, y: Optional[None] = None) -> "DynamicNIS":
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
        self.models_ = {q: [] for q in self.macro_dims}
        
        for q in self.macro_dims:
            model = DynNISModule(p_dim, q, self.hidden_dim).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            prev_model = None
            macro_ei_sampled_q = np.zeros(len(eval_indices), dtype=np.float64)
            
            anchor_iter = enumerate(eval_indices)
            if self.verbose:
                anchor_iter = tqdm(anchor_iter, total=len(eval_indices), desc=f"Training Dyn-NIS+ (q={q})")
                
            for idx, t in anchor_iter:
                w = compute_temporal_weights(int(t), T_trans, self.bandwidth, self.kernel_type, self.causal_only)
                audit_kernel_leakage(w, center_t=int(t), causal_only=self.causal_only)
                w_t = torch.from_numpy(w.astype(np.float32)).to(self.device)
                
                # Active support mask for computational efficiency
                active_mask = w_t > 1e-5
                if torch.sum(active_mask) < 8:
                    active_mask = torch.ones_like(w_t, dtype=torch.bool)
                    
                x_p = x_past_all[active_mask]
                x_f = x_future_all[active_mask]
                w_sub = w_t[active_mask]
                
                # Local train / held-out validation split to eliminate winner's curse
                n_active = len(x_p)
                val_mask = torch.zeros(n_active, dtype=torch.bool, device=self.device)
                val_indices = torch.arange(0, n_active, 4, device=self.device)  # 25% held-out
                val_mask[val_indices] = True
                train_mask = ~val_mask
                
                # Local optimization
                model.train()
                for _ in range(self.steps_per_window):
                    optimizer.zero_grad()
                    v_p, v_pred, x_recon = model(x_p[train_mask])
                    v_target = model.encoder(x_f[train_mask])
                    w_tr = w_sub[train_mask]
                    
                    # 1. Prediction & Reconstruction Loss
                    loss_latent = torch.sum(w_tr * torch.mean((v_target - v_pred) ** 2, dim=-1))
                    loss_recon = torch.sum(w_tr * torch.mean((x_f[train_mask] - x_recon) ** 2, dim=-1))
                    loss_pred = loss_latent + 0.3 * loss_recon
                    
                    # 2. Intervention-consistent EI
                    ei_loss = compute_intervention_consistent_ei(model.dynamics, v_p, v_target, w_tr)
                    
                    # 3. Gauge Whitening Regularizer: ||Cov_w(V) - I_q||_F^2
                    v_cent = v_p - torch.mean(v_p, dim=0, keepdim=True)
                    cov_v = torch.mm(v_cent.t(), v_cent) / float(v_p.shape[0] - 1 + 1e-6)
                    loss_gauge = torch.norm(cov_v - torch.eye(q, device=self.device), p="fro") ** 2
                    
                    # 4. Procrustes Representation Alignment & Dynamics Smoothness
                    loss_phi = torch.tensor(0.0, device=self.device)
                    loss_f = torch.tensor(0.0, device=self.device)
                    if prev_model is not None:
                        with torch.no_grad():
                            v_prev = prev_model.encoder(x_p[train_mask])
                        loss_phi = compute_procrustes_alignment(v_p, v_prev, w_tr)
                        loss_f = torch.mean((model.dynamics(v_p) - prev_model.dynamics(v_p)) ** 2)
                        
                    # Combined objective
                    total_loss = (
                        loss_pred
                        - self.lambda_ei * ei_loss
                        + self.eta_phi * loss_phi
                        + self.eta_f * loss_f
                        + self.gamma_gauge * loss_gauge
                    )
                    total_loss.backward()
                    optimizer.step()
                    
                # Evaluate EI on held-out transitions
                model.eval()
                with torch.no_grad():
                    v_p_val = model.encoder(x_p[val_mask])
                    v_f_val = model.encoder(x_f[val_mask])
                    w_val = w_sub[val_mask]
                    macro_ei_sampled_q[idx] = float(
                        compute_intervention_consistent_ei(model.dynamics, v_p_val, v_f_val, w_val).item()
                    )
                    
                # Warm-start checkpointing
                prev_model = copy.deepcopy(model)
                self.models_[q].append(copy.deepcopy(model))
                
            # Interpolate to all timestamps if eval_step > 1
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
        """Select optimal dimension q_t^* and compute uncertainty confidence sets."""
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

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        """Return macroscopic representations using the time-varying optimal encoders."""
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        T_eval = min(len(X), len(self.causal_dimension_))
        outputs = []
        
        with torch.no_grad():
            for t in range(T_eval):
                best_q = self.causal_dimension_[t]
                # Map timestamp to model checkpoint
                model_idx = min(t // self.eval_step, len(self.models_[best_q]) - 1)
                model = self.models_[best_q][model_idx]
                v_t = model.encoder(X_tensor[t:t+1]).cpu().numpy()[0]
                outputs.append(v_t)
        return np.array(outputs, dtype=object)
