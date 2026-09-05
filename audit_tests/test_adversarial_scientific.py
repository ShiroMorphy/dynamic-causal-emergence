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
