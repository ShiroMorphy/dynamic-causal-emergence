import numpy as np
import pytest
from dce.core.entropy import (
    gaussian_differential_entropy,
    knn_differential_entropy,
    conditional_gaussian_entropy
)
from dce.core.kernels import compute_temporal_weights, select_bandwidth_cv
from dce.core.effective_info import compute_gaussian_effective_information, estimate_local_gaussian_dynamics


def test_gaussian_entropy_1d():
    var = 2.0
    cov = np.array([[var]])
    h_analytical = 0.5 * np.log(2.0 * np.pi * np.e * var)
    h_calc = gaussian_differential_entropy(cov)
    assert np.isclose(h_calc, h_analytical, atol=1e-5)


def test_knn_entropy_gaussian():
    np.random.seed(42)
    # 1D standard Gaussian samples
    samples = np.random.randn(2000, 1)
    h_analytical = 0.5 * np.log(2.0 * np.pi * np.e)
    h_knn = knn_differential_entropy(samples, k=5)
    assert np.isclose(h_knn, h_analytical, atol=0.1)


def test_temporal_weights_normalization():
    weights = compute_temporal_weights(center_t=50, total_t=100, bandwidth=10.0, kernel_type="gaussian")
    assert len(weights) == 100
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert np.argmax(weights) == 50


def test_effective_information_deterministic():
    A = np.eye(2)
    Sigma = 0.01 * np.eye(2)
    decomp = compute_gaussian_effective_information(A, Sigma)
    assert decomp.effective_information > 0.0
    assert decomp.determinism > decomp.degeneracy
