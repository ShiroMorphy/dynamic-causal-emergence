"""Adversarial scientific tests produced by the independent audit.

These tests are intentionally outside ``tests/`` because the repository's
pytest configuration only collects that directory.  Run them explicitly with

    pytest -q audit_tests/test_adversarial_scientific.py

Every failing test identifies a scientific contract that the current
implementation does not satisfy.  They are not marked xfail: turning a failure
green must require a scientific correction, not a waiver.
"""

import numpy as np
import pandas as pd
import pytest

from dce.core.effective_info import (
    compute_gaussian_effective_information,
    estimate_local_affine_dynamics,
    estimate_local_gaussian_dynamics,
)
from dce.datasets.eia930.balance import compute_balance_residuals
from dce.stats.surrogates import generate_multivariate_iaaft_surrogate


def _cross_spectral_relative_error(x: np.ndarray, y: np.ndarray) -> float:
    fx = np.fft.rfft(x, axis=0)
    fy = np.fft.rfft(y, axis=0)
    sx = np.einsum("fi,fj->fij", fx, fx.conj())
    sy = np.einsum("fi,fj->fij", fy, fy.conj())
    return float(np.linalg.norm(sy - sx) / np.linalg.norm(sx))


def test_zero_channel_has_zero_effective_information_even_when_singular():
    """A=0 makes intervention and output independent, hence EI is exactly zero."""
    result = compute_gaussian_effective_information(
        np.zeros((3, 3)), np.zeros((3, 3)), regularization=1e-6
    )
    assert result.effective_information == pytest.approx(0.0, abs=1e-12)


def test_nonfinite_model_parameters_are_rejected_not_silently_replaced():
    """NaN/Inf in a fitted scientific model must invalidate the estimate."""
    with pytest.raises(ValueError):
        compute_gaussian_effective_information(
            np.array([[np.nan]]), np.array([[1.0]])
        )


def test_uniform_intervention_does_not_use_gaussian_entropy_formula():
    """Equal covariance does not make uniform-input mutual information Gaussian."""
    with pytest.raises(NotImplementedError):
        compute_gaussian_effective_information(
            np.array([[1.0]]),
            np.array([[0.25]]),
            intervention="uniform",
            domain_bound=np.sqrt(3.0),
        )


def test_local_dynamics_recovers_affine_transition():
    """Local affine dynamics must recover transition matrix and intercept under affine drift."""
    x = np.linspace(-1.0, 1.0, 101)[:, None]
    y = 2.0 * x + 5.0
    weights = np.ones(len(x)) / len(x)
    a_hat, c_hat, _ = estimate_local_affine_dynamics(x, y, weights, ridge_alpha=1e-12)
    prediction = x @ a_hat.T + c_hat
    assert np.max(np.abs(prediction - y)) < 1e-8


def test_multivariate_iaaft_preserves_cross_spectrum_to_declared_tolerance():
    """The claimed multivariate null requires the full cross-spectrum."""
    rng = np.random.RandomState(19)
    t = np.arange(2048)
    common = np.sin(2 * np.pi * t / 31) + 0.5 * np.sin(2 * np.pi * t / 73)
    x = np.column_stack(
        [
            common + 0.05 * rng.randn(len(t)),
            np.roll(common, 4) + 0.05 * rng.randn(len(t)),
            0.7 * common + 0.3 * np.roll(common, 11) + 0.05 * rng.randn(len(t)),
        ]
    )
    surrogate = generate_multivariate_iaaft_surrogate(x, max_iter=100, seed=7)
    assert _cross_spectral_relative_error(x, surrogate) < 0.05


def test_multivariate_iaaft_returns_exact_empirical_marginals():
    """The returned iterate, not an intermediate array, must preserve marginals."""
    rng = np.random.RandomState(23)
    x = rng.standard_t(df=3, size=(1024, 4))
    surrogate = generate_multivariate_iaaft_surrogate(x, max_iter=100, seed=11)
    np.testing.assert_allclose(
        np.sort(surrogate, axis=0), np.sort(x, axis=0), rtol=0.0, atol=1e-12
    )


def test_eia930_total_interchange_uses_export_positive_sign_convention():
    """For EIA-930, +10 MW interchange is net export, not net import."""
    frame = pd.DataFrame(
        {"generation": [100.0], "interchange": [10.0], "demand": [90.0]}
    )
    result = compute_balance_residuals(frame)
    assert result.loc[0, "accounting_residual"] == pytest.approx(0.0)


def test_dgpe_finite_sample_recovery():
    """DGP-E finite-sample DCD^PR and q90 recovery under spectral Marchenko-Pastur thresholding."""
    from dce.datasets.synthetic import generate_dgp_e_changing_dimension
    from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
    
    data = generate_dgp_e_changing_dimension(n_steps=2400, seed=42)
    est = LocalLinearGaussianDCE(bandwidth=36.0, macro_dims=[1, 2, 4, 8, 16], ridge_alpha=0.01)
    est.fit(data.states)
    
    rmse = float(np.sqrt(np.mean((est.dcd_pr_ - data.true_dcd_pr) ** 2)))
    bias = float(np.mean(est.dcd_pr_ - data.true_dcd_pr))
    acc_q90 = float(np.mean(est.q90_ == data.true_q90))
    
    assert rmse < 1.0, f"RMSE(DCD^PR) must be < 1.0, got {rmse:.4f}"
    assert abs(bias) < 0.35, f"|Bias(DCD^PR)| must be < 0.35, got {bias:.4f}"
    assert acc_q90 >= 0.50, f"Accuracy(q90) must be >= 50%, got {acc_q90*100:.1f}%"


def test_fisher_causal_projection_orthonormality_and_maximal_trace():
    """Fisher causal projection W_t must be orthonormal and maximize Fisher causal trace."""
    from dce.core.effective_info import compute_causal_spectrum
    rng = np.random.RandomState(42)
    p, q = 6, 2
    A = rng.randn(p, p)
    Sig_raw = rng.randn(p, p)
    Sigma = Sig_raw @ Sig_raw.T + np.eye(p)
    
    report = compute_causal_spectrum(A, Sigma, regularization=0.0)
    W = report.projection_v[:, :q]
    
    # 1. Orthonormality W^T W = I_q
    np.testing.assert_allclose(W.T @ W, np.eye(q), atol=1e-12)
    
    # 2. Trace maximality: Tr(W^T A^T Sigma^{-1} A W) = sum_{i=1}^q lambda_i
    F = A.T @ np.linalg.inv(Sigma) @ A
    actual_trace = float(np.trace(W.T @ F @ W))
    expected_trace = float(np.sum(report.eigenvalues[:q]))
    assert actual_trace == pytest.approx(expected_trace, rel=1e-10)


def test_causal_data_pipeline_no_lookahead_bfill():
    """In causal rolling mode, missing values at t=0 must not be backfilled with future data."""
    from dce.datasets.eia930.microstate import build_power_grid_microstate
    timestamps = pd.date_range("2021-01-01", periods=10, freq="h")
    # BA1 has missing values at t=0 and t=1
    data = []
    for t in timestamps:
        data.append({"timestamp": t, "ba_code": "BA1", "demand": np.nan if t <= timestamps[1] else 100.0, "generation": 100.0, "wind": 10.0, "solar": 5.0, "interchange": 0.0, "interconnection": "ERCOT"})
        data.append({"timestamp": t, "ba_code": "BA2", "demand": 50.0, "generation": 50.0, "wind": 5.0, "solar": 2.0, "interchange": 0.0, "interconnection": "ERCOT"})
    df = pd.DataFrame(data)
    
    grid_data = build_power_grid_microstate(df, interconnection="ERCOT", scaling="causal_rolling")
    # BA1 demand feature is column 0
    assert grid_data.microstate_matrix[0, 0] == 0.0
    assert grid_data.microstate_matrix[1, 0] == 0.0


def test_h1_surrogate_mean_critical_value():
    """Hypothesis1Result must compute and store the 5th percentile critical value of surrogate means."""
    from dce.stats.hypothesis import compute_surrogate_significance_from_ensemble
    rng = np.random.RandomState(42)
    T, B = 500, 100
    empirical = rng.randn(T) + 5.0
    surrogates = rng.randn(B, T) + 7.0
    
    result = compute_surrogate_significance_from_ensemble(empirical, surrogates, test_direction="less")
    assert result.critical_value_mean is not None
    expected_crit = float(np.percentile(np.mean(surrogates, axis=1), 5.0))
    assert result.critical_value_mean == pytest.approx(expected_crit, rel=1e-10)


def test_null_dgp_no_overcontraction():
    """Null DGPs A, F, G must maintain DCD^PR >= 7.0 (no over-contraction under Gavish-Donoho thresholding)."""
    from dce.datasets.synthetic import (
        generate_dgp_a_null_stationary,
        generate_dgp_f_heteroskedastic_shock,
        generate_dgp_g_correlation_shock
    )
    from dce.estimators.linear_gaussian import LocalLinearGaussianDCE

    for gen in (generate_dgp_a_null_stationary, generate_dgp_f_heteroskedastic_shock, generate_dgp_g_correlation_shock):
        data = gen(n_steps=1000, seed=42)
        est = LocalLinearGaussianDCE(bandwidth=50.0, ridge_alpha=0.01)
        est.fit(data.states)
        mean_pr = float(np.mean(est.dcd_pr_))
        assert mean_pr >= 7.0, f"Null DGP DCD^PR collapsed to {mean_pr:.3f}, expected >= 7.0"


def test_dgp_j_untouched_dynamic_scale_recovery():
    """Untouched benchmark DGP-J (6 -> 3 -> 2) must be recovered with RMSE < 0.8 and accurate q90 tracking."""
    from dce.datasets.synthetic import generate_dgp_j_hierarchical_transition
    from dce.estimators.linear_gaussian import LocalLinearGaussianDCE

    data = generate_dgp_j_hierarchical_transition(n_steps=1800, seed=42)
    est = LocalLinearGaussianDCE(bandwidth=50.0, ridge_alpha=0.01)
    est.fit(data.states)

    rmse = float(np.sqrt(np.mean((est.dcd_pr_ - data.true_dcd_pr) ** 2)))
    acc_q90 = float(np.mean(est.q90_ == data.true_q90))
    assert rmse < 0.80, f"DGP-J RMSE must be < 0.80, got {rmse:.4f}"
    assert acc_q90 >= 0.60, f"DGP-J q90 accuracy must be >= 60%, got {acc_q90*100:.1f}%"


def test_observational_lifting_ground_truth_consistency():
    """compute_linear_gaussian_dce_ground_truth must strictly match fit_oracle() under identical projection."""
    from dce.datasets.synthetic import compute_linear_gaussian_dce_ground_truth
    from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
    import scipy.linalg

    rng = np.random.RandomState(42)
    p, q = 6, 2
    # Generate stable transition matrix
    A = rng.randn(p, p) * 0.3
    Sig = rng.randn(p, p)
    Sigma = 0.2 * (Sig @ Sig.T) + 0.1 * np.eye(p)
    Sigma_X = scipy.linalg.solve_discrete_lyapunov(A, Sigma)

    W = np.zeros((p, q))
    W[0:3, 0] = 1.0 / np.sqrt(3)
    W[3:6, 1] = 1.0 / np.sqrt(3)

    micro_ei, macro_ei, dce_raw, dce_dens = compute_linear_gaussian_dce_ground_truth(
        A, Sigma, q, W=W, Sigma_X=Sigma_X
    )

    oracle = LocalLinearGaussianDCE(macro_dims=[q], ridge_alpha=1e-8)
    oracle.fit_oracle(
        A_sequence=A[np.newaxis, :, :],
        Sigma_sequence=Sigma[np.newaxis, :, :],
        state_cov_sequence=Sigma_X[np.newaxis, :, :],
        projections={q: [W]}
    )

    assert micro_ei == pytest.approx(float(oracle.micro_ei_[0]), rel=1e-6)
    assert macro_ei == pytest.approx(float(oracle.macro_ei_[q][0]), rel=1e-6)
    assert dce_raw == pytest.approx(float(oracle.dce_raw_[q][0]), rel=1e-6)
    assert dce_dens == pytest.approx(float(oracle.dce_density_[q][0]), rel=1e-6)

