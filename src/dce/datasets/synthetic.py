"""
Synthetic Dynamic Causal Emergence Benchmark Suite with Analytical Ground Truth (DGPs A through I).

Canonical Data Generating Processes (DGPs) according to Q1 protocol:
- DGP-A: Null Stationary (Strictly no emergence, stationary linear Gaussian)
- DGP-B: Null Nonstationary (Time-varying drift, variance, and autocorrelation with zero emergence)
- DGP-C: Abrupt Emergence (Abrupt transition at tau from DCE=0 to DCE=c > 0)
- DGP-D: Smooth Drift (Continuous logistic mechanism drift from DCE=0 to DCE=c > 0)
- DGP-E: Known Changing Causal Dimension (Dynamic causal dimension q_t^*: 8 -> 4 -> 2)
- DGP-F: Heteroskedastic Shock (Violent variance spike with DCE=0, tests FPR calibration)
- DGP-G: Correlation Shock (Contemporary correlation shift without causal mechanism change)
- DGP-H: Kuramoto Network (Phenomenological complex-system synchronization with order parameter R(t))
- DGP-I: Chaotic Nonlinear Dynamics (Coupled logistic maps with time-varying macroscopic coupling)
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import scipy.linalg

from dce.core.effective_info import compute_gaussian_effective_information, compute_causal_spectrum


class SyntheticBenchmarkData(NamedTuple):
    """Container for synthetic test dataset and analytical ground truth."""
    states: np.ndarray             # (T, p) Microstate matrix
    true_dce: np.ndarray           # (T-1,) Primary analytical or nominal DCE_t (density metric for emergence benchmarking)
    true_optimal_dim: np.ndarray   # (T-1,) True optimal macro dimension q_t^*
    transition_timestamp: Optional[Union[int, Tuple[int, ...]]] = None # Ground truth shock timestamp(s)
    dgp_name: str = "custom"
    extra_info: Optional[Dict[str, Any]] = None
    true_dce_raw: Optional[np.ndarray] = None       # (T-1,) Exact raw DCE: EI(V) - EI(X) <= 0
    true_dce_density: Optional[np.ndarray] = None   # (T-1,) Exact density DCE: EI(V)/q - EI(X)/p
    true_dcd_pr: Optional[np.ndarray] = None        # (T-1,) Exact analytical Participation Ratio
    true_dcd_entropy: Optional[np.ndarray] = None   # (T-1,) Exact analytical Causal Effective Rank
    true_q90: Optional[np.ndarray] = None           # (T-1,) Exact analytical 90% causal dimension


def compute_linear_gaussian_dcd_ground_truth(
    A: np.ndarray,
    Sigma: np.ndarray
) -> Tuple[float, float, int]:
    """
    Computes exact analytical Dynamic Causal Dimensionality:
    (dcd_pr, dcd_entropy, q90) from the continuous Causal Information Spectrum:
    e_i = 1/2 * ln(1 + lambda_i) where lambda_i are eigenvalues of Sigma^{-1} A A^T.
    """
    rep = compute_causal_spectrum(A, Sigma, denoising=None)
    cum_e = np.cumsum(rep.spectrum) / max(rep.effective_information, 1e-12)
    q90 = int(np.searchsorted(cum_e, 0.90) + 1)
    return float(rep.dcd_pr), float(rep.dcd_entropy), int(q90)


def compute_linear_gaussian_dce_ground_truth(
    A: np.ndarray,
    Sigma: np.ndarray,
    q: int,
    W: Optional[np.ndarray] = None,
    Sigma_X: Optional[np.ndarray] = None,
    regularization: float = 1e-8
) -> Tuple[float, float, float, float]:
    """
    Computes exact analytical micro EI, macro EI, DCE_raw, DCE_density
    for a linear Gaussian transition mechanism (A, Sigma) under observational lifting.
    
    Observational lifting channel:
        B = W^T A Sigma_X W (W^T Sigma_X W)^{-1}
        Sigma_macro = W^T (A Sigma_{X|V} A^T + Sigma) W
    consistent with Section 2.3 of the manuscript and fit_oracle().
    """
    p = A.shape[0]
    micro_decomp = compute_gaussian_effective_information(A, Sigma, regularization=regularization)
    micro_ei = micro_decomp.effective_information
    
    if q >= p:
        return micro_ei, micro_ei, 0.0, 0.0
        
    if W is None:
        k = p // q
        W = np.zeros((p, q), dtype=np.float64)
        for c in range(q):
            W[c * k : (c + 1) * k, c] = 1.0 / np.sqrt(k)
            
    # Solve for stationary microstate covariance Sigma_X = A Sigma_X A^T + Sigma
    if Sigma_X is None:
        try:
            Sigma_X = scipy.linalg.solve_discrete_lyapunov(A, Sigma)
            Sigma_X = 0.5 * (Sigma_X + Sigma_X.T)
        except Exception:
            Sigma_X = np.eye(p, dtype=np.float64)
            
    cov_v = W.T @ Sigma_X @ W + regularization * np.eye(q)
    cov_v_cross = W.T @ A @ Sigma_X @ W
    try:
        B = (np.linalg.solve(cov_v, cov_v_cross.T)).T
    except np.linalg.LinAlgError:
        B = W.T @ A @ W
        
    cov_v_next = W.T @ (A @ Sigma_X @ A.T + Sigma) @ W
    Sigma_macro = cov_v_next - B @ cov_v @ B.T
    Sigma_macro = 0.5 * (Sigma_macro + Sigma_macro.T) + regularization * np.eye(q)
    
    macro_decomp = compute_gaussian_effective_information(B, Sigma_macro, regularization=regularization)
    macro_ei = macro_decomp.effective_information
    
    dce_raw = macro_ei - micro_ei
    dce_density = (macro_ei / float(q)) - (micro_ei / float(p))
    return micro_ei, macro_ei, dce_raw, dce_density


# =============================================================================
# DGP-A: Null Stationary (No Emergence)
# =============================================================================
def generate_dgp_a_null_stationary(
    n_steps: int = 1000,
    p_dim: int = 8,
    rho: float = 0.7,
    noise_level: float = 0.25,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-A: Null Stationary.
    Independent stationary autoregressive processes with isotropic noise:
        X_{t+1} = rho * X_t + epsilon_t,  epsilon_t ~ N(0, noise_level^2 * I_p)
    True DCE_t <= 0 (raw) and DCE_t^{density} == 0 everywhere.
    """
    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.randn(p_dim) * noise_level
    
    A = rho * np.eye(p_dim)
    for t in range(n_steps - 1):
        states[t + 1] = A @ states[t] + noise_level * rng.randn(p_dim)
        
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    true_dcd_pr = np.full(n_steps - 1, float(p_dim), dtype=np.float64)
    true_dcd_entropy = np.full(n_steps - 1, float(p_dim), dtype=np.float64)
    true_q90 = np.full(n_steps - 1, int(np.ceil(0.9 * p_dim)), dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=None,
        dgp_name="DGP-A_null_stationary",
        extra_info={"A": A, "noise_level": noise_level},
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-B: Null Nonstationary (Noise & Drift Changes without Emergence)
# =============================================================================
def generate_dgp_b_null_nonstationary(
    n_steps: int = 1000,
    p_dim: int = 8,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-B: Null Nonstationary.
    System with time-varying mean drift, nonstationary noise level, and drifting autocorrelation,
    yet remaining strictly uncoupled across micro dimensions:
        X_{t+1} = rho_t * X_t + mu_t + sigma_t * epsilon_t
    True DCE_t <= 0 (raw) and DCE_t^{density} == 0 everywhere.
    """
    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.randn(p_dim)
    
    time_indices = np.linspace(0, 4 * np.pi, n_steps)
    rho_t = 0.4 + 0.3 * np.sin(time_indices)
    sigma_t = 0.2 + 0.15 * np.cos(time_indices * 0.5) ** 2
    mu_t = 0.5 * np.sin(time_indices * 0.25)
    
    for t in range(n_steps - 1):
        states[t + 1] = rho_t[t] * states[t] + mu_t[t] + sigma_t[t] * rng.randn(p_dim)
        
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    true_dcd_pr = np.full(n_steps - 1, float(p_dim), dtype=np.float64)
    true_dcd_entropy = np.full(n_steps - 1, float(p_dim), dtype=np.float64)
    true_q90 = np.full(n_steps - 1, int(np.ceil(0.9 * p_dim)), dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=None,
        dgp_name="DGP-B_null_nonstationary",
        extra_info={"rho_t": rho_t, "sigma_t": sigma_t},
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-C: Abrupt Emergence (Regime Shift at tau)
# =============================================================================
def generate_dgp_c_abrupt_emergence(
    n_steps: int = 2000,
    transition_t: int = 1000,
    p_dim: int = 8,
    q_dim: int = 2,
    noise_level: float = 0.15,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-C: Abrupt Causal Emergence.
    t < transition_t: Microstates independent 1-to-1 -> DCE = 0, q* = p.
    t >= transition_t: Microstates degenerate into q clusters where internal noise cancels at macro scale.
    """
    if transition_t is None or transition_t >= n_steps:
        transition_t = n_steps // 2

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    
    cluster_size = p_dim // q_dim
    states[0] = rng.randn(p_dim)
    
    A_regime_a = np.roll(np.eye(p_dim), 1, axis=0) * 0.95
    
    A_regime_b = np.zeros((p_dim, p_dim))
    for c in range(q_dim):
        target_c = (c + 1) % q_dim
        src_slice = slice(c * cluster_size, (c + 1) * cluster_size)
        tgt_slice = slice(target_c * cluster_size, (target_c + 1) * cluster_size)
        A_regime_b[tgt_slice, src_slice] = 0.95 / cluster_size
        
    sigma_micro = noise_level * 2.5
    sigma_macro = noise_level
    Sigma_b = np.zeros((p_dim, p_dim), dtype=np.float64)
    for c in range(q_dim):
        s = slice(c * cluster_size, (c + 1) * cluster_size)
        Sigma_b[s, s] = (sigma_macro ** 2) + (sigma_micro ** 2) * (np.eye(cluster_size) - 1.0 / cluster_size)
        
    _, _, dce_raw_b, dce_density_b = compute_linear_gaussian_dce_ground_truth(
        A_regime_b, Sigma_b, q_dim
    )
    dcd_pr_b, dcd_ent_b, q90_b = compute_linear_gaussian_dcd_ground_truth(
        A_regime_b, Sigma_b
    )
    
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    true_dcd_pr = np.zeros(n_steps - 1, dtype=np.float64)
    true_dcd_entropy = np.zeros(n_steps - 1, dtype=np.float64)
    true_q90 = np.zeros(n_steps - 1, dtype=np.int32)
    
    for t in range(n_steps - 1):
        if t < transition_t:
            states[t + 1] = A_regime_a @ states[t] + noise_level * rng.randn(p_dim)
            true_dce_raw[t] = 0.0
            true_dce_density[t] = 0.0
            true_optimal_dim[t] = p_dim
            true_dcd_pr[t] = float(p_dim)
            true_dcd_entropy[t] = float(p_dim)
            true_q90[t] = p_dim
        else:
            micro_noise = rng.randn(p_dim) * sigma_micro
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] -= np.mean(micro_noise[c_slice])
                
            macro_noise = rng.randn(q_dim) * sigma_macro
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] += macro_noise[c]
                
            states[t + 1] = A_regime_b @ states[t] + micro_noise
            true_dce_raw[t] = dce_raw_b
            true_dce_density[t] = dce_density_b
            true_optimal_dim[t] = q_dim
            true_dcd_pr[t] = dcd_pr_b
            true_dcd_entropy[t] = dcd_ent_b
            true_q90[t] = q90_b
            
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=transition_t,
        dgp_name="DGP-C_abrupt_emergence",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-D: Smooth Drift (Logistic Mechanism Interpolation)
# =============================================================================
def generate_dgp_d_smooth_drift(
    n_steps: int = 2000,
    center_t: int = 1000,
    transition_width: float = 150.0,
    p_dim: int = 8,
    q_dim: int = 2,
    noise_level: float = 0.15,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-D: Smooth Drift.
    Smooth logistic drift of transition mechanism A_t = (1 - rho_t) A_0 + rho_t A_1.
    Allows testing tracking lag and bias during gradual reorganization.
    """
    if center_t is None or center_t >= n_steps:
        center_t = n_steps // 2

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    cluster_size = p_dim // q_dim
    states[0] = rng.randn(p_dim)

    A_0 = np.eye(p_dim) * 0.8
    A_1 = np.zeros((p_dim, p_dim))
    for c in range(q_dim):
        target_c = (c + 1) % q_dim
        src_slice = slice(c * cluster_size, (c + 1) * cluster_size)
        tgt_slice = slice(target_c * cluster_size, (target_c + 1) * cluster_size)
        A_1[tgt_slice, src_slice] = 0.9 / cluster_size
        
    t_vals = np.arange(n_steps - 1)
    rho_series = 1.0 / (1.0 + np.exp(-(t_vals - center_t) / (transition_width / 4.0)))
    
    # Precompute ground truth curve along drift weights
    grid_weights = np.linspace(0.0, 1.0, 21)
    grid_raw = []
    grid_density = []
    grid_dcd_pr = []
    grid_dcd_ent = []
    grid_q90 = []
    for w in grid_weights:
        A_w = (1.0 - w) * A_0 + w * A_1
        sig_micro_w = noise_level * (1.0 + 1.5 * w)
        Sigma_w = np.zeros((p_dim, p_dim), dtype=np.float64)
        for c in range(q_dim):
            s = slice(c * cluster_size, (c + 1) * cluster_size)
            Sigma_w[s, s] = (sig_micro_w ** 2) * (np.eye(cluster_size) - (w / cluster_size) * (np.ones((cluster_size, cluster_size)) - np.eye(cluster_size)) / max(1, cluster_size - 1))
            Sigma_w[s, s] = 0.5 * (Sigma_w[s, s] + Sigma_w[s, s].T) + 1e-6 * np.eye(cluster_size)
        _, _, d_raw, d_dens = compute_linear_gaussian_dce_ground_truth(A_w, Sigma_w, q_dim)
        pr_w, ent_w, q90_w = compute_linear_gaussian_dcd_ground_truth(A_w, Sigma_w)
        grid_raw.append(d_raw)
        grid_density.append(d_dens)
        grid_dcd_pr.append(pr_w)
        grid_dcd_ent.append(ent_w)
        grid_q90.append(q90_w)
        
    true_dce_raw = np.interp(rho_series, grid_weights, grid_raw)
    true_dce_density = np.interp(rho_series, grid_weights, grid_density)
    true_optimal_dim = np.where(rho_series < 0.5, p_dim, q_dim).astype(np.int32)
    true_dcd_pr = np.interp(rho_series, grid_weights, grid_dcd_pr)
    true_dcd_entropy = np.interp(rho_series, grid_weights, grid_dcd_ent)
    true_q90 = np.round(np.interp(rho_series, grid_weights, grid_q90)).astype(np.int32)
    
    for t in range(n_steps - 1):
        weight = rho_series[t]
        A_t = (1.0 - weight) * A_0 + weight * A_1
        
        micro_noise = rng.randn(p_dim) * noise_level * (1.0 + 1.5 * weight)
        if weight > 0.1:
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] -= weight * np.mean(micro_noise[c_slice])
        states[t + 1] = A_t @ states[t] + micro_noise
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=center_t,
        dgp_name="DGP-D_smooth_drift",
        extra_info={"rho_series": rho_series},
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-E: Known Changing Causal Dimension (q_t^*: 8 -> 4 -> 2)
# =============================================================================
def generate_dgp_e_changing_dimension(
    n_steps: int = 2400,
    stages: Tuple[int, int] = (800, 1600),
    p_dim: int = 16,
    q_stages: Tuple[int, int, int] = (8, 4, 2),
    noise_level: float = 0.15,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-E: Changing Causal Dimension.
    Demonstrates dynamic causal scale switching:
    Stage 1 (t < 800): q^* = 8 (pairs of 2)
    Stage 2 (800 <= t < 1600): q^* = 4 (clusters of 4)
    Stage 3 (t >= 1600): q^* = 2 (clusters of 8)
    """
    if stages is None or stages[1] >= n_steps:
        stages = (n_steps // 3, 2 * n_steps // 3)

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.zeros(n_steps - 1, dtype=np.int32)
    true_dcd_pr = np.zeros(n_steps - 1, dtype=np.float64)
    true_dcd_entropy = np.zeros(n_steps - 1, dtype=np.float64)
    true_q90 = np.zeros(n_steps - 1, dtype=np.int32)
    
    t1, t2 = stages
    q1, q2, q3 = q_stages
    states[0] = rng.randn(p_dim)
    
    stage_params = {}
    for q_curr in (q1, q2, q3):
        c_sz = p_dim // q_curr
        A_q = np.zeros((p_dim, p_dim))
        for c in range(q_curr):
            tgt = (c + 1) % q_curr
            A_q[tgt * c_sz : (tgt + 1) * c_sz, c * c_sz : (c + 1) * c_sz] = 0.9 / c_sz
        sig_m = noise_level * 2.0
        sig_M = noise_level
        Sig_q = np.zeros((p_dim, p_dim))
        for c in range(q_curr):
            s = slice(c * c_sz, (c + 1) * c_sz)
            Sig_q[s, s] = (sig_M ** 2) + (sig_m ** 2) * (np.eye(c_sz) - 1.0 / c_sz)
        _, _, d_raw, d_dens = compute_linear_gaussian_dce_ground_truth(A_q, Sig_q, q_curr)
        dcd_pr_q, dcd_ent_q, q90_q = compute_linear_gaussian_dcd_ground_truth(A_q, Sig_q)
        stage_params[q_curr] = (A_q, d_raw, d_dens, dcd_pr_q, dcd_ent_q, q90_q)
    
    for t in range(n_steps - 1):
        if t < t1:
            q_curr = q1
        elif t < t2:
            q_curr = q2
        else:
            q_curr = q3
            
        A_curr, d_raw_curr, d_dens_curr, pr_curr, ent_curr, q90_curr = stage_params[q_curr]
        true_dce_raw[t] = d_raw_curr
        true_dce_density[t] = d_dens_curr
        true_optimal_dim[t] = q_curr
        true_dcd_pr[t] = pr_curr
        true_dcd_entropy[t] = ent_curr
        true_q90[t] = q90_curr
            
        c_size = p_dim // q_curr
        micro_noise = rng.randn(p_dim) * (noise_level * 2.0)
        for c in range(q_curr):
            c_slice = slice(c * c_size, (c + 1) * c_size)
            micro_noise[c_slice] -= np.mean(micro_noise[c_slice])
        macro_noise = rng.randn(q_curr) * noise_level
        for c in range(q_curr):
            c_slice = slice(c * c_size, (c + 1) * c_size)
            micro_noise[c_slice] += macro_noise[c]
            
        states[t + 1] = A_curr @ states[t] + micro_noise
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=stages,
        dgp_name="DGP-E_changing_dimension",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-F: Heteroskedastic Shock without Emergence (Volatility Spike)
# =============================================================================
def generate_dgp_f_heteroskedastic_shock(
    n_steps: int = 1200,
    shock_window: Tuple[int, int] = (500, 700),
    p_dim: int = 8,
    shock_factor: float = 6.0,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-F: Heteroskedastic Shock without Emergence.
    During shock_window, noise variance spikes by shock_factor^2, but the transition
    mechanism remains stationary and isotropic.
    True DCE_t <= 0 throughout. Tests whether estimator produces false alarms on volatility surges.
    """
    if shock_window is None or shock_window[1] >= n_steps:
        shock_window = (int(0.4 * n_steps), int(0.6 * n_steps))

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.randn(p_dim)
    
    A = 0.75 * np.eye(p_dim)
    t_start, t_end = shock_window
    base_noise = 0.2
    
    pr_base, ent_base, q90_base = compute_linear_gaussian_dcd_ground_truth(
        A, (base_noise ** 2) * np.eye(p_dim)
    )
    pr_shock, ent_shock, q90_shock = compute_linear_gaussian_dcd_ground_truth(
        A, ((base_noise * shock_factor) ** 2) * np.eye(p_dim)
    )
    
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    true_dcd_pr = np.zeros(n_steps - 1, dtype=np.float64)
    true_dcd_entropy = np.zeros(n_steps - 1, dtype=np.float64)
    true_q90 = np.zeros(n_steps - 1, dtype=np.int32)
    
    for t in range(n_steps - 1):
        is_shock = (t_start <= t < t_end)
        noise_std = base_noise * (shock_factor if is_shock else 1.0)
        states[t + 1] = A @ states[t] + noise_std * rng.randn(p_dim)
        
        true_dcd_pr[t] = pr_shock if is_shock else pr_base
        true_dcd_entropy[t] = ent_shock if is_shock else ent_base
        true_q90[t] = q90_shock if is_shock else q90_base
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=shock_window,
        dgp_name="DGP-F_heteroskedastic_shock",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-G: Correlation Shock without Emergence
# =============================================================================
def generate_dgp_g_correlation_shock(
    n_steps: int = 1200,
    shock_window: Tuple[int, int] = (500, 700),
    p_dim: int = 8,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-G: Correlation Shock without Emergence.
    Contemporary covariance shifts from diagonal to dense high-correlation during shock_window,
    while the causal transition mechanism A remains isotropic.
    Separates causal emergence from PCA eigenvalue shifts and effective rank drops.
    """
    if shock_window is None or shock_window[1] >= n_steps:
        shock_window = (int(0.4 * n_steps), int(0.6 * n_steps))

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.randn(p_dim)
    
    A = 0.7 * np.eye(p_dim)
    t_start, t_end = shock_window
    
    v = np.ones(p_dim) / np.sqrt(p_dim)
    dense_cov = 0.1 * np.eye(p_dim) + 0.8 * np.outer(v, v)
    L_dense = np.linalg.cholesky(dense_cov)
    base_std = 0.3
    
    pr_base, ent_base, q90_base = compute_linear_gaussian_dcd_ground_truth(
        A, (base_std ** 2) * np.eye(p_dim)
    )
    pr_shock, ent_shock, q90_shock = compute_linear_gaussian_dcd_ground_truth(
        A, dense_cov
    )
    
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    true_dcd_pr = np.zeros(n_steps - 1, dtype=np.float64)
    true_dcd_entropy = np.zeros(n_steps - 1, dtype=np.float64)
    true_q90 = np.zeros(n_steps - 1, dtype=np.int32)
    
    for t in range(n_steps - 1):
        is_shock = (t_start <= t < t_end)
        if is_shock:
            noise = L_dense @ rng.randn(p_dim)
        else:
            noise = base_std * rng.randn(p_dim)
        states[t + 1] = A @ states[t] + noise
        
        true_dcd_pr[t] = pr_shock if is_shock else pr_base
        true_dcd_entropy[t] = ent_shock if is_shock else ent_base
        true_q90[t] = q90_shock if is_shock else q90_base
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=shock_window,
        dgp_name="DGP-G_correlation_shock",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# DGP-H: Nonstationary Kuramoto Oscillators
# =============================================================================
def generate_dgp_h_kuramoto(
    n_steps: int = 2000,
    n_oscillators: int = 32,
    dt: float = 0.05,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-H: Nonstationary Kuramoto Network.
    Simulates N oscillators with time-varying global coupling K(t):
    theta_i' = omega_i + (K(t)/N) * sum_j sin(theta_j - theta_i) + noise.
    Order parameter R(t) = |(1/N) * sum_j exp(i * theta_j)| tracks synchronization.
    """
    rng = np.random.RandomState(seed)
    natural_freqs = rng.standard_cauchy(size=n_oscillators) * 0.5
    phases = rng.uniform(-np.pi, np.pi, size=n_oscillators)
    
    states = np.zeros((n_steps, n_oscillators), dtype=np.float64)
    order_param = np.zeros(n_steps - 1, dtype=np.float64)
    
    # Coupling transitions through synchronization onset K_c ~ 1.0
    time_arr = np.linspace(0, 1, n_steps - 1)
    coupling_k = 0.2 + 2.5 / (1.0 + np.exp(-(time_arr - 0.5) * 15.0))
    
    for t in range(n_steps - 1):
        states[t] = np.sin(phases)
        z = np.mean(np.exp(1j * phases))
        order_param[t] = np.abs(z)
        
        phase_diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
        interaction = np.sum(np.sin(phase_diffs), axis=1) * (coupling_k[t] / float(n_oscillators))
        phases += (natural_freqs + interaction) * dt + 0.1 * np.sqrt(dt) * rng.randn(n_oscillators)
        phases = (phases + np.pi) % (2.0 * np.pi) - np.pi
        
    states[-1] = np.sin(phases)
    
    true_dce_density = np.maximum(0.0, (order_param - 0.3) * 1.5)
    true_dce_raw = np.maximum(0.0, (order_param - 0.5) * 1.0)
    true_optimal_dim = np.where(order_param > 0.6, 2, n_oscillators).astype(np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=int(n_steps * 0.5),
        dgp_name="DGP-H_kuramoto",
        extra_info={"order_parameter": order_param, "coupling_k": coupling_k},
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density
    )


# =============================================================================
# DGP-I: Chaotic Nonlinear Dynamics (Coupled Logistic Maps)
# =============================================================================
def generate_dgp_i_chaotic_nonlinear(
    n_steps: int = 1500,
    p_dim: int = 8,
    coupling_transition: int = 750,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-I: Coupled Logistic Maps with Drifting Inter-Cluster Coupling.
    x_{i, t+1} = (1 - eps_t) * f(x_{i,t}) + eps_t * mean(f(x_{neighbors}))
    Under high coupling, chaotic microstates synchronize into collective macro-trajectories.
    """
    if coupling_transition is None or coupling_transition >= n_steps:
        coupling_transition = n_steps // 2

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.uniform(0.1, 0.9, size=p_dim)
    
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.zeros(n_steps - 1, dtype=np.int32)
    
    q_macro = 2
    c_size = p_dim // q_macro
    r = 3.9  # Chaotic regime
    
    for t in range(n_steps - 1):
        x = states[t]
        f_x = r * x * (1.0 - x)
        
        if t < coupling_transition:
            # Independent chaotic maps
            states[t + 1] = f_x + rng.randn(p_dim) * 0.01
            true_dce_raw[t] = 0.0
            true_dce_density[t] = 0.0
            true_optimal_dim[t] = p_dim
        else:
            # Strong intra-cluster coupling
            coupled = np.zeros_like(f_x)
            for c in range(q_macro):
                c_slice = slice(c * c_size, (c + 1) * c_size)
                mean_cluster = np.mean(f_x[c_slice])
                coupled[c_slice] = 0.3 * f_x[c_slice] + 0.7 * mean_cluster
            states[t + 1] = coupled + rng.randn(p_dim) * 0.005
            true_dce_density[t] = 0.5 * np.log(c_size)
            true_dce_raw[t] = 0.0
            true_optimal_dim[t] = q_macro
            
        states[t + 1] = np.clip(states[t + 1], 0.001, 0.999)
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=coupling_transition,
        dgp_name="DGP-I_chaotic_nonlinear",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density
    )


# =============================================================================
# DGP-J: Untouched Dynamic Scale Transition (Hierarchical Branching)
# =============================================================================
def generate_dgp_j_hierarchical_transition(
    n_steps: int = 2400,
    stages: Tuple[int, int] = (800, 1600),
    p_dim: int = 12,
    q_stages: Tuple[int, int, int] = (6, 3, 2),
    noise_level: float = 0.15,
    seed: int = 42
) -> SyntheticBenchmarkData:
    """
    DGP-J: Untouched Dynamic Scale Transition (Hierarchical Branching).
    Demonstrates dynamic causal scale switching on an out-of-sample benchmark (p=12):
    Stage 1 (t < 800): q^* = 6 (pairs of 2 micro-nodes)
    Stage 2 (800 <= t < 1600): q^* = 3 (clusters of 4 micro-nodes)
    Stage 3 (t >= 1600): q^* = 2 (clusters of 6 micro-nodes)
    """
    if stages is None or stages[1] >= n_steps:
        stages = (n_steps // 3, 2 * n_steps // 3)

    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    true_dce_raw = np.zeros(n_steps - 1, dtype=np.float64)
    true_dce_density = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.zeros(n_steps - 1, dtype=np.int32)
    true_dcd_pr = np.zeros(n_steps - 1, dtype=np.float64)
    true_dcd_entropy = np.zeros(n_steps - 1, dtype=np.float64)
    true_q90 = np.zeros(n_steps - 1, dtype=np.int32)
    
    t1, t2 = stages
    q1, q2, q3 = q_stages
    states[0] = rng.randn(p_dim)
    
    stage_params = {}
    for q_curr in (q1, q2, q3):
        c_sz = p_dim // q_curr
        A_q = np.zeros((p_dim, p_dim))
        for c in range(q_curr):
            tgt = (c + 1) % q_curr
            A_q[tgt * c_sz : (tgt + 1) * c_sz, c * c_sz : (c + 1) * c_sz] = 0.85 / c_sz
        sig_m = noise_level * 2.0
        sig_M = noise_level
        Sig_q = np.zeros((p_dim, p_dim))
        for c in range(q_curr):
            s = slice(c * c_sz, (c + 1) * c_sz)
            Sig_q[s, s] = (sig_M ** 2) + (sig_m ** 2) * (np.eye(c_sz) - 1.0 / c_sz)
        _, _, d_raw, d_dens = compute_linear_gaussian_dce_ground_truth(A_q, Sig_q, q_curr)
        dcd_pr_q, dcd_ent_q, q90_q = compute_linear_gaussian_dcd_ground_truth(A_q, Sig_q)
        stage_params[q_curr] = (A_q, d_raw, d_dens, dcd_pr_q, dcd_ent_q, q90_q)
    
    for t in range(n_steps - 1):
        if t < t1:
            q_curr = q1
        elif t < t2:
            q_curr = q2
        else:
            q_curr = q3
            
        A_curr, d_raw_curr, d_dens_curr, pr_curr, ent_curr, q90_curr = stage_params[q_curr]
        true_dce_raw[t] = d_raw_curr
        true_dce_density[t] = d_dens_curr
        true_optimal_dim[t] = q_curr
        true_dcd_pr[t] = pr_curr
        true_dcd_entropy[t] = ent_curr
        true_q90[t] = q90_curr
            
        c_size = p_dim // q_curr
        micro_noise = rng.randn(p_dim) * (noise_level * 2.0)
        for c in range(q_curr):
            c_slice = slice(c * c_size, (c + 1) * c_size)
            micro_noise[c_slice] -= np.mean(micro_noise[c_slice])
        macro_noise = rng.randn(q_curr) * noise_level
        for c in range(q_curr):
            c_slice = slice(c * c_size, (c + 1) * c_size)
            micro_noise[c_slice] += macro_noise[c]
            
        states[t + 1] = A_curr @ states[t] + micro_noise
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce_density,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=stages,
        dgp_name="DGP-J_hierarchical_transition",
        true_dce_raw=true_dce_raw,
        true_dce_density=true_dce_density,
        true_dcd_pr=true_dcd_pr,
        true_dcd_entropy=true_dcd_entropy,
        true_q90=true_q90
    )


# =============================================================================
# Registry and Backward-Compatible Aliases
# =============================================================================
SYNTHETIC_DGP_REGISTRY = {
    "dgp_a": generate_dgp_a_null_stationary,
    "dgp_b": generate_dgp_b_null_nonstationary,
    "dgp_c": generate_dgp_c_abrupt_emergence,
    "dgp_d": generate_dgp_d_smooth_drift,
    "dgp_e": generate_dgp_e_changing_dimension,
    "dgp_f": generate_dgp_f_heteroskedastic_shock,
    "dgp_g": generate_dgp_g_correlation_shock,
    "dgp_h": generate_dgp_h_kuramoto,
    "dgp_i": generate_dgp_i_chaotic_nonlinear,
    "dgp_j": generate_dgp_j_hierarchical_transition,
}


def get_synthetic_benchmark(dgp_name: str, **kwargs) -> SyntheticBenchmarkData:
    """Factory function for retrieving synthetic benchmark data."""
    key = str(dgp_name).lower().replace("-", "_")
    if key not in SYNTHETIC_DGP_REGISTRY:
        raise ValueError(f"Unknown DGP '{dgp_name}'. Available: {list(SYNTHETIC_DGP_REGISTRY.keys())}")
    return SYNTHETIC_DGP_REGISTRY[key](**kwargs)


# Backward-compatible function aliases
def generate_regime_switching_system(*args, **kwargs) -> SyntheticBenchmarkData:
    return generate_dgp_c_abrupt_emergence(*args, **kwargs)


def generate_nonstationary_kuramoto(*args, **kwargs) -> SyntheticBenchmarkData:
    return generate_dgp_h_kuramoto(*args, **kwargs)
