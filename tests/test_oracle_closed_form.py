"""
Unit-test Oracle and Closed-Form Analytical Tests for Linear-Gaussian DCE.

Verifies:
1. Closed-form scalar and multivariate EI formulas against exact theory.
2. Orthogonal invariance under coordinate rotations.
3. Strict non-emergence (DCE <= 0) on isotropic Markov null systems.
4. Positive causal emergence on redundant noisy clusters (Liu et al. 2024).
5. Strict temporal kernel anti-leakage audit for causal mode.
"""

import numpy as np
import pytest

from dce.core.interventions import (
    GaussianMaxEntropyIntervention,
    UniformCompactIntervention,
)
from dce.core.effective_info import (
    compute_gaussian_effective_information,
    compute_dce,
)
from dce.core.kernels import compute_temporal_weights, audit_kernel_leakage
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE


def test_scalar_analytical_ei():
    """
    Test scalar X_{t+1} = a * X_t + eps against exact analytical formula:
    EI = 0.5 * ln(1 + (a * sigma_do / sigma_eps)^2)
    """
    a = 1.8
    sigma_do = 1.5
    sigma_eps = 0.4
    
    A = np.array([[a]])
    Sigma = np.array([[sigma_eps ** 2]])
    interv = GaussianMaxEntropyIntervention(dim=1, sigma_do=sigma_do)
    
    decomp = compute_gaussian_effective_information(A, Sigma, intervention=interv)
    
    snr = (a * sigma_do / sigma_eps) ** 2
    exact_ei = 0.5 * np.log(1.0 + snr)
    
    assert np.isclose(decomp.effective_information, exact_ei, rtol=1e-5)
    assert decomp.determinism > decomp.degeneracy


def test_orthogonal_invariance():
    """
    Under an orthogonal change of coordinates X' = Q X,
    the continuous Effective Information must be invariant.
    """
    p = 4
    rng = np.random.RandomState(42)
    A = rng.randn(p, p) * 0.5
    Sigma = np.diag(rng.uniform(0.1, 0.5, size=p))
    
    # Random orthogonal matrix Q
    H = rng.randn(p, p)
    Q, _ = np.linalg.qr(H)
    
    A_rot = Q @ A @ Q.T
    Sigma_rot = Q @ Sigma @ Q.T
    
    decomp_orig = compute_gaussian_effective_information(A, Sigma)
    decomp_rot = compute_gaussian_effective_information(A_rot, Sigma_rot)
    
    assert np.isclose(decomp_orig.effective_information, decomp_rot.effective_information, atol=1e-5)


def test_isotropic_null_system_has_no_emergence():
    """
    For an isotropic system A = c * I_p, Sigma = s^2 * I_p:
    Every macro projection of dimension q < p must have:
    - Raw DCE(q) = (q - p) * ei_per_dim < 0
    - Normalized DCE^{norm}(q) == 0
    """
    p = 6
    macro_dims = [1, 2, 3]
    c = 0.8
    s = 0.2
    
    A = c * np.eye(p)
    Sigma = (s ** 2) * np.eye(p)
    
    oracle = LocalLinearGaussianDCE(macro_dims=macro_dims)
    oracle.fit_oracle(A, Sigma)
    
    # Raw DCE must be negative for all q < p
    for q in macro_dims:
        assert np.all(oracle.dce_per_dim_[q] < 0.0)
        
    # Selected emergence is non-positive
    assert np.all(oracle.emergence_ <= 0.0)


def test_redundant_microstates_produce_positive_dce():
    """
    Two coupled units with identical macro signal but high anti-correlated internal noise.
    Projecting onto the sum coordinate cancels internal noise, proving positive causal emergence (Liu et al. 2024).
    """
    p = 2
    q = 1
    # Micro dynamics: both units follow average state
    A = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    
    # Noise covariance: common macro noise variance 0.04, anti-correlated internal noise variance 0.64
    v_macro = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
    v_micro = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    
    Sigma = 0.04 * (v_macro @ v_macro.T) + 0.64 * (v_micro @ v_micro.T)
    
    oracle = LocalLinearGaussianDCE(macro_dims=[1])
    oracle.fit_oracle(A, Sigma, projections={1: v_macro})
    
    micro_ei = oracle.micro_ei_[0]
    macro_ei = oracle.macro_ei_[1][0]
    raw_dce = oracle.dce_per_dim_[1][0]
    norm_dce = oracle.normalized_emergence_[0]
    
    # 1D macro captures full causal power of the 2D micro system: macro_ei approx micro_ei
    assert np.isclose(macro_ei, micro_ei, rtol=5e-3)
    # Normalized causal emergence per degree of freedom is strictly positive:
    # EI(V)/1 > EI(X)/2 by factor of 2x
    assert norm_dce > 0.5
    assert np.isclose(norm_dce, macro_ei - micro_ei / 2.0, rtol=5e-3)



def test_temporal_kernel_leakage_audit():
    """Verify that causal_only mode strictly respects past-only causality."""
    T = 100
    for t in [0, 20, 50, 99]:
        w_causal = compute_temporal_weights(center_t=t, total_t=T, bandwidth=15.0, causal_only=True)
        # Must pass audit
        assert audit_kernel_leakage(w_causal, center_t=t, causal_only=True)
        # Explicit test
        if t < T - 1:
            assert np.all(w_causal[t + 1:] == 0.0)
            
    # Retrospective kernel has future weights
    w_retro = compute_temporal_weights(center_t=50, total_t=T, bandwidth=15.0, causal_only=False)
    assert np.sum(w_retro[51:]) > 0.0


def test_data_driven_local_linear_gaussian():
    """Test data-driven LocalLinearGaussianDCE fitting in both retrospective and causal modes."""
    rng = np.random.RandomState(42)
    T = 150
    p = 4
    X = np.zeros((T, p))
    A = 0.7 * np.eye(p)
    for t in range(T - 1):
        X[t + 1] = A @ X[t] + 0.1 * rng.randn(p)
        
    model_retro = LocalLinearGaussianDCE(macro_dims=[1, 2], bandwidth=20.0, causal_only=False)
    model_retro.fit(X)
    assert model_retro.emergence_.shape == (T - 1,)
    assert model_retro.causal_dimension_.shape == (T - 1,)
    assert np.all(np.isfinite(model_retro.emergence_))
    assert len(model_retro.causal_dimension_confset_) == T - 1
    
    model_causal = LocalLinearGaussianDCE(macro_dims=[1, 2], bandwidth=20.0, causal_only=True)
    model_causal.fit(X)
    assert model_causal.emergence_.shape == (T - 1,)
    assert np.all(np.isfinite(model_causal.emergence_))

