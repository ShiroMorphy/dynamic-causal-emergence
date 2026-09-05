"""
Independent Analytical Oracles and Strict Contract Tests for Dynamic Causal Emergence (DCE).

No hardcoded constants. All test expectations are calculated analytically from stochastic equations.
"""

import numpy as np
import pytest

from dce.core.effective_info import (
    compute_gaussian_effective_information,
    compute_dce_raw,
    compute_dce_density,
    compute_dce
)
from dce.core.kernels import compute_causal_weights, compute_retrospective_weights, audit_kernel_leakage
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE


def test_oracle_isotropic_null_analytical_identity():
    """
    For an uncoupled isotropic AR(1) system:
        X_{t+1} = c * X_t + epsilon_t,  epsilon_t ~ N(0, s^2 * I_p)
    Theoretical values:
        EI(X) = (p / 2) * ln((c^2 + s^2) / s^2)
        EI(V^{(q)}) = (q / 2) * ln((c^2 + s^2) / s^2)
        DCE(q) = (q - p) * 0.5 * ln((c^2 + s^2) / s^2) < 0  (Strictly negative!)
        DCE^{density}(q) = 0.0  (Strictly zero!)
    """
    p = 6
    c = 0.85
    s = 0.35
    snr_factor = 0.5 * np.log((c**2 + s**2) / (s**2))
    
    A = c * np.eye(p)
    Sigma = (s**2) * np.eye(p)
    
    macro_dims = [1, 2, 3]
    oracle = LocalLinearGaussianDCE(macro_dims=macro_dims, ridge_alpha=0.0, selection_criterion="raw")
    oracle.fit_oracle(A, Sigma)
    
    expected_micro_ei = p * snr_factor
    assert np.isclose(oracle.micro_ei_[0], expected_micro_ei, rtol=1e-5)
    
    for q in macro_dims:
        expected_macro_ei = q * snr_factor
        expected_raw_dce = (q - p) * snr_factor
        
        assert np.isclose(oracle.macro_ei_[q][0], expected_macro_ei, rtol=1e-5)
        # Raw DCE must be strictly negative
        assert oracle.dce_raw_[q][0] < -0.1
        assert np.isclose(oracle.dce_raw_[q][0], expected_raw_dce, rtol=1e-5)
        # Density DCE must be exactly zero
        assert np.isclose(oracle.dce_density_[q][0], 0.0, atol=1e-5)
        
    # Absolute emergence is strictly negative
    assert oracle.optimal_dce_raw_[0] < 0.0


def test_oracle_redundant_canceling_noise_genuine_emergence():
    """
    Genuine Causal Emergence (Liu et al. 2024 / Klein & Hoel 2020):
    Macro state s_{t+1} = a * s_t + eta_t, eta ~ N(0, sigma_eta^2)
    Micro units:
        x_1 = s + xi
        x_2 = s - xi
    where xi ~ N(0, sigma_xi^2) with sigma_xi >> sigma_eta.
    Coarse-graining v = (x_1 + x_2) / sqrt(2) = sqrt(2) * s completely cancels internal noise!
    """
    a = 0.9
    sigma_eta = 0.1
    sigma_xi = 1.0  # Large internal micro noise
    
    # In macro coordinate v = (x_1 + x_2)/sqrt(2), internal noise xi cancels out
    w_macro = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
    w_micro = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    
    A = a * (w_macro @ w_macro.T)
    Sigma = (sigma_eta**2) * (w_macro @ w_macro.T) + (sigma_xi**2) * (w_micro @ w_micro.T)
    
    oracle = LocalLinearGaussianDCE(macro_dims=[1], ridge_alpha=0.0, selection_criterion="raw")
    oracle.fit_oracle(A, Sigma, projections={1: w_macro})
    
    # Macro scale filters out large micro noise sigma_xi
    # Therefore macro EI should equal or exceed micro EI
    micro_ei = oracle.micro_ei_[0]
    macro_ei = oracle.macro_ei_[1][0]
    
    # Verify exact macro SNR: (a * 1 / sigma_eta)^2
    exact_macro_ei = 0.5 * np.log(1.0 + (a / sigma_eta)**2)
    assert np.isclose(macro_ei, exact_macro_ei, rtol=1e-4)
    
    # Micro EI is degraded by sigma_xi
    assert macro_ei >= micro_ei - 1e-4
    # Density gain is strictly positive
    density_gain = macro_ei - micro_ei / 2.0
    assert density_gain > 0.5


def test_oracle_diagonal_system_exact_closed_form():
    """Verify exact multi-channel independent AR(1) sum formula."""
    a_diag = np.array([0.95, 0.80, 0.60, 0.30])
    s_diag = np.array([0.05, 0.10, 0.20, 0.50])
    
    A = np.diag(a_diag)
    Sigma = np.diag(s_diag)
    
    decomp = compute_gaussian_effective_information(A, Sigma, regularization=0.0)
    
    expected_ei = 0.5 * np.sum(np.log((a_diag**2 + s_diag) / s_diag))
    assert np.isclose(decomp.effective_information, expected_ei, rtol=1e-12)


def test_oracle_orthogonal_rotation_invariance():
    """Under any orthogonal coordinate rotation Q in O(p), EI must be invariant to machine precision."""
    p = 5
    rng = np.random.RandomState(123)
    A = rng.randn(p, p) * 0.4
    Sigma = np.diag(rng.uniform(0.1, 0.6, size=p))
    
    # Random Haar-distributed orthogonal matrix
    H = rng.randn(p, p)
    Q, _ = np.linalg.qr(H)
    
    A_rot = Q @ A @ Q.T
    Sigma_rot = Q @ Sigma @ Q.T
    
    ei_orig = compute_gaussian_effective_information(A, Sigma).effective_information
    ei_rot = compute_gaussian_effective_information(A_rot, Sigma_rot).effective_information
    
    assert np.isclose(ei_orig, ei_rot, atol=1e-10)


def test_dual_metric_divergence_contract():
    """
    DEF-01 Contract Test:
    Demonstrate that DCE^{raw} and DCE^{density} can have OPPOSITE signs,
    proving they are fundamentally different physical estimands.
    """
    micro_ei = 8.0   # p = 4 channels -> 2.0 nats/dim
    macro_ei = 3.0   # q = 1 channel  -> 3.0 nats/dim
    p_dim = 4
    q_dim = 1
    
    raw_dce = compute_dce_raw(micro_ei, macro_ei)
    density_dce = compute_dce_density(micro_ei, macro_ei, p_dim, q_dim)
    
    # Raw is strictly negative (-5.0 nats)
    assert raw_dce == -5.0
    assert raw_dce < 0.0
    
    # Density is strictly positive (+1.0 nats/dim)
    assert density_dce == +1.0
    assert density_dce > 0.0
    
    # They diverge in sign:
    assert (raw_dce > 0) != (density_dce > 0)


def test_causal_kernel_strict_zero_leakage():
    """Contract test: Causal kernel must have exactly 0.0 weight on any future observation."""
    T = 120
    bandwidth = 18.0
    
    for t in [0, 5, 25, 60, 119]:
        w_causal = compute_causal_weights(center_t=t, total_t=T, bandwidth=bandwidth)
        assert audit_kernel_leakage(w_causal, center_t=t, causal_only=True)
        if t < T - 1:
            assert np.max(w_causal[t+1:]) == 0.0
            assert np.sum(w_causal[t+1:]) == 0.0
            
    # Retrospective kernel must have non-zero future weights for interior points
    w_retro = compute_retrospective_weights(center_t=50, total_t=T, bandwidth=bandwidth)
    assert np.sum(w_retro[51:]) > 0.1
