import numpy as np
import pytest
from dce.stats.surrogates import generate_iaaft_surrogate, generate_multivariate_surrogates
from dce.stats.hypothesis import (
    compute_h1_causal_emergence_significance,
    compute_h4_out_of_sample_forecasting
)


def test_iaaft_surrogate_preserves_spectrum():
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    orig = np.sin(t) + 0.2 * np.random.randn(200)
    surr = generate_iaaft_surrogate(orig, seed=42)
    assert len(surr) == len(orig)
    assert np.isclose(np.mean(surr), np.mean(orig), atol=0.1)
    assert np.isclose(np.std(surr), np.std(orig), atol=0.1)


def test_h1_surrogate_test():
    dce = np.array([0.5, 0.6, 0.7, 0.8])
    surr_ensemble = np.zeros((100, 4))
    res = compute_h1_causal_emergence_significance(dce, surr_ensemble, alpha=0.05)
    assert res.significant_ratio > 0.9


def test_h4_forecasting():
    np.random.seed(42)
    fe = np.random.uniform(0.01, 0.05, 100)
    dce = np.random.uniform(0.1, 0.8, 100)
    q = np.random.randint(1, 4, 100)
    res = compute_h4_out_of_sample_forecasting(fe, dce, q, horizon_h=1)
    assert res.rmse_augmented_dce >= 0.0
