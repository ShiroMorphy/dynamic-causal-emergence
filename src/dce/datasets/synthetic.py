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


class SyntheticBenchmarkData(NamedTuple):
    """Container for synthetic test dataset and analytical ground truth."""
    states: np.ndarray             # (T, p) Microstate matrix
    true_dce: np.ndarray           # (T-1,) Analytical or nominal DCE_t
    true_optimal_dim: np.ndarray   # (T-1,) True optimal macro dimension q_t^*
    transition_timestamp: Optional[Union[int, Tuple[int, ...]]] = None # Ground truth shock timestamp(s)
    dgp_name: str = "custom"
    extra_info: Optional[Dict[str, Any]] = None


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
    True DCE_t <= 0 (raw) and DCE_t^{norm} == 0 everywhere.
    """
    rng = np.random.RandomState(seed)
    states = np.zeros((n_steps, p_dim), dtype=np.float64)
    states[0] = rng.randn(p_dim) * noise_level
    
    A = rho * np.eye(p_dim)
    for t in range(n_steps - 1):
        states[t + 1] = A @ states[t] + noise_level * rng.randn(p_dim)
        
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=None,
        dgp_name="DGP-A_null_stationary",
        extra_info={"A": A, "noise_level": noise_level}
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
    True DCE_t <= 0 everywhere. Essential to ensure DCE does not falsely detect emergence from volatility.
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
        
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=None,
        dgp_name="DGP-B_null_nonstationary",
        extra_info={"rho_t": rho_t, "sigma_t": sigma_t}
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
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.zeros(n_steps - 1, dtype=np.int32)
    
    cluster_size = p_dim // q_dim
    states[0] = rng.randn(p_dim)
    
    A_regime_a = np.roll(np.eye(p_dim), 1, axis=0) * 0.95
    
    A_regime_b = np.zeros((p_dim, p_dim))
    for c in range(q_dim):
        target_c = (c + 1) % q_dim
        src_slice = slice(c * cluster_size, (c + 1) * cluster_size)
        tgt_slice = slice(target_c * cluster_size, (target_c + 1) * cluster_size)
        A_regime_b[tgt_slice, src_slice] = 0.95 / cluster_size
        
    expected_dce_b = float(np.log(cluster_size) * 0.5)
    
    for t in range(n_steps - 1):
        if t < transition_t:
            states[t + 1] = A_regime_a @ states[t] + noise_level * rng.randn(p_dim)
            true_dce[t] = 0.0
            true_optimal_dim[t] = p_dim
        else:
            micro_noise = rng.randn(p_dim) * (noise_level * 2.5)
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] -= np.mean(micro_noise[c_slice])
                
            macro_noise = rng.randn(q_dim) * noise_level
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] += macro_noise[c]
                
            states[t + 1] = A_regime_b @ states[t] + micro_noise
            true_dce[t] = expected_dce_b
            true_optimal_dim[t] = q_dim
            
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=transition_t,
        dgp_name="DGP-C_abrupt_emergence"
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
    
    expected_dce_max = float(np.log(cluster_size) * 0.5)
    true_dce = rho_series * expected_dce_max
    true_optimal_dim = np.where(rho_series < 0.5, p_dim, q_dim).astype(np.int32)
    
    for t in range(n_steps - 1):
        weight = rho_series[t]
        A_t = (1.0 - weight) * A_0 + weight * A_1
        
        # Noise mixture
        micro_noise = rng.randn(p_dim) * noise_level * (1.0 + 1.5 * weight)
        if weight > 0.1:
            for c in range(q_dim):
                c_slice = slice(c * cluster_size, (c + 1) * cluster_size)
                micro_noise[c_slice] -= weight * np.mean(micro_noise[c_slice])
        states[t + 1] = A_t @ states[t] + micro_noise
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=center_t,
        dgp_name="DGP-D_smooth_drift",
        extra_info={"rho_series": rho_series}
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
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.zeros(n_steps - 1, dtype=np.int32)
    
    t1, t2 = stages
    q1, q2, q3 = q_stages
    states[0] = rng.randn(p_dim)
    
    for t in range(n_steps - 1):
        if t < t1:
            q_curr = q1
            true_dce[t] = 0.5 * np.log(p_dim / q1)
            true_optimal_dim[t] = q1
        elif t < t2:
            q_curr = q2
            true_dce[t] = 0.5 * np.log(p_dim / q2)
            true_optimal_dim[t] = q2
        else:
            q_curr = q3
            true_dce[t] = 0.5 * np.log(p_dim / q3)
            true_optimal_dim[t] = q3
            
        c_size = p_dim // q_curr
        A_curr = np.zeros((p_dim, p_dim))
        for c in range(q_curr):
            target_c = (c + 1) % q_curr
            src_slice = slice(c * c_size, (c + 1) * c_size)
            tgt_slice = slice(target_c * c_size, (target_c + 1) * c_size)
            A_curr[tgt_slice, src_slice] = 0.9 / c_size
            
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
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=stages,
        dgp_name="DGP-E_changing_dimension"
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
    
    for t in range(n_steps - 1):
        noise_std = base_noise * (shock_factor if (t_start <= t < t_end) else 1.0)
        states[t + 1] = A @ states[t] + noise_std * rng.randn(p_dim)
        
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=shock_window,
        dgp_name="DGP-F_heteroskedastic_shock"
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
    
    # Dense correlation matrix during shock
    v = np.ones(p_dim) / np.sqrt(p_dim)
    dense_cov = 0.1 * np.eye(p_dim) + 0.8 * np.outer(v, v)
    L_dense = np.linalg.cholesky(dense_cov)
    
    base_std = 0.3
    
    for t in range(n_steps - 1):
        if t_start <= t < t_end:
            noise = L_dense @ rng.randn(p_dim)
        else:
            noise = base_std * rng.randn(p_dim)
        states[t + 1] = A @ states[t] + noise
        
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
    true_optimal_dim = np.full(n_steps - 1, p_dim, dtype=np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=shock_window,
        dgp_name="DGP-G_correlation_shock"
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
    
    true_dce = np.maximum(0.0, (order_param - 0.3) * 1.5)
    true_optimal_dim = np.where(order_param > 0.6, 2, n_oscillators).astype(np.int32)
    
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=int(n_steps * 0.5),
        dgp_name="DGP-H_kuramoto",
        extra_info={"order_parameter": order_param, "coupling_k": coupling_k}
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
    
    true_dce = np.zeros(n_steps - 1, dtype=np.float64)
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
            true_dce[t] = 0.0
            true_optimal_dim[t] = p_dim
        else:
            # Strong intra-cluster coupling
            coupled = np.zeros_like(f_x)
            for c in range(q_macro):
                c_slice = slice(c * c_size, (c + 1) * c_size)
                mean_cluster = np.mean(f_x[c_slice])
                coupled[c_slice] = 0.3 * f_x[c_slice] + 0.7 * mean_cluster
            states[t + 1] = coupled + rng.randn(p_dim) * 0.005
            true_dce[t] = 0.5 * np.log(c_size)
            true_optimal_dim[t] = q_macro
            
        states[t + 1] = np.clip(states[t + 1], 0.001, 0.999)
        
    return SyntheticBenchmarkData(
        states=states,
        true_dce=true_dce,
        true_optimal_dim=true_optimal_dim,
        transition_timestamp=coupling_transition,
        dgp_name="DGP-I_chaotic_nonlinear"
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
