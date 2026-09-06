"""
Tests for Synthetic Benchmark Suite (DGPs A through I).
"""

import numpy as np
import pytest

from dce.datasets.synthetic import (
    SYNTHETIC_DGP_REGISTRY,
    get_synthetic_benchmark,
    generate_dgp_a_null_stationary,
    generate_dgp_b_null_nonstationary,
    generate_dgp_c_abrupt_emergence,
    generate_dgp_d_smooth_drift,
    generate_dgp_e_changing_dimension,
    generate_dgp_f_heteroskedastic_shock,
    generate_dgp_g_correlation_shock,
    generate_dgp_h_kuramoto,
    generate_dgp_i_chaotic_nonlinear,
)


@pytest.mark.parametrize("dgp_key", list(SYNTHETIC_DGP_REGISTRY.keys()))
def test_all_dgps_generate_valid_data(dgp_key):
    """Verify that every canonical DGP generates valid, finite, correctly-shaped data."""
    data = get_synthetic_benchmark(dgp_key, n_steps=200, seed=123)
    
    assert data.states.ndim == 2
    T, p = data.states.shape
    assert T == 200
    assert p >= 2
    
    assert data.true_dce.shape == (T - 1,)
    assert data.true_optimal_dim.shape == (T - 1,)
    
    assert np.all(np.isfinite(data.states))
    assert np.all(np.isfinite(data.true_dce))
    assert np.all(np.isfinite(data.true_optimal_dim))


def test_dgp_a_and_b_null_properties():
    """Null DGPs must have true DCE identically zero or non-positive."""
    dgp_a = generate_dgp_a_null_stationary(n_steps=300)
    dgp_b = generate_dgp_b_null_nonstationary(n_steps=300)
    
    assert np.all(dgp_a.true_dce <= 0.0)
    assert np.all(dgp_b.true_dce <= 0.0)


def test_dgp_c_abrupt_emergence_transition():
    """DGP-C must exhibit clean jump from zero to positive emergence at transition_t."""
    trans_t = 250
    data = generate_dgp_c_abrupt_emergence(n_steps=500, transition_t=trans_t)
    
    assert np.all(data.true_dce[:trans_t] == 0.0)
    assert np.all(data.true_dce[trans_t:] > 0.0)
    assert np.all(data.true_optimal_dim[:trans_t] == 8)
    assert np.all(data.true_optimal_dim[trans_t:] == 2)


def test_dgp_e_changing_dimensions():
    """DGP-E must step through dimensions 8 -> 4 -> 2."""
    data = generate_dgp_e_changing_dimension(n_steps=600, stages=(200, 400), p_dim=16, q_stages=(8, 4, 2))
    
    assert np.all(data.true_optimal_dim[:200] == 8)
    assert np.all(data.true_optimal_dim[200:400] == 4)
    assert np.all(data.true_optimal_dim[400:] == 2)


def test_dgp_f_heteroskedastic_shock():
    """DGP-F must have higher sample variance during shock window."""
    data = generate_dgp_f_heteroskedastic_shock(n_steps=500, shock_window=(200, 300), shock_factor=5.0)
    
    var_normal = np.var(data.states[:200])
    var_shock = np.var(data.states[200:300])
    assert var_shock > 3.0 * var_normal


def test_dgp_g_correlation_shock():
    """DGP-G off-diagonal correlation must increase during shock window."""
    data = generate_dgp_g_correlation_shock(n_steps=500, shock_window=(200, 300))
    
    corr_normal = np.corrcoef(data.states[:200].T)
    corr_shock = np.corrcoef(data.states[200:300].T)
    
    off_diag_normal = np.abs(corr_normal[np.triu_indices_from(corr_normal, k=1)])
    off_diag_shock = np.abs(corr_shock[np.triu_indices_from(corr_shock, k=1)])
    
    assert np.mean(off_diag_shock) > np.mean(off_diag_normal) + 0.2


def test_dgp_h_kuramoto_order_parameter():
    """DGP-H synchronization order parameter must increase with coupling."""
    data = generate_dgp_h_kuramoto(n_steps=400, n_oscillators=16)
    order_p = data.extra_info["order_parameter"]
    assert np.mean(order_p[-50:]) > np.mean(order_p[:50])


def test_dgp_j_hierarchical_transition():
    """DGP-J must transition through dimensions 6 -> 3 -> 2."""
    from dce.datasets.synthetic import generate_dgp_j_hierarchical_transition
    data = generate_dgp_j_hierarchical_transition(n_steps=600, stages=(200, 400), p_dim=12)
    assert np.all(data.true_optimal_dim[:200] == 6)
    assert np.all(data.true_optimal_dim[200:400] == 3)
    assert np.all(data.true_optimal_dim[400:] == 2)
    assert data.true_dcd_pr is not None
    assert np.all(np.isfinite(data.true_dcd_pr))


def test_dgp_k_holdout_transition():
    """DGP-K must transition through dimensions 5 -> 2 -> 1 on 10D system."""
    from dce.datasets.synthetic import generate_dgp_k_holdout_transition
    data = generate_dgp_k_holdout_transition(n_steps=600, stages=(200, 400), p_dim=10)
    assert np.all(data.true_optimal_dim[:200] == 5)
    assert np.all(data.true_optimal_dim[200:400] == 2)
    assert np.all(data.true_optimal_dim[400:] == 1)
    assert data.true_dcd_pr is not None
    assert np.all(np.isfinite(data.true_dcd_pr))
    assert data.true_q90 is not None
