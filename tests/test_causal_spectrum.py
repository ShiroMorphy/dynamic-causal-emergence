"""
Gate B2: Comprehensive Verification of Causal Information Spectrum and Dynamic Causal Dimensionality (DCD).

Tests:
1. Exact equivalence of sum(e_i) to closed-form Gaussian Effective Information.
2. Exact rotational invariance under orthogonal coordinate changes Q in O(p).
3. Exact integer recovery of effective rank for k-mode degenerate systems (k in {1, 2, 4, 8}).
4. Numerical stability guard under A = 0 and vanishing effective information (zero division avoidance).
5. Monotonic non-increasing order and non-negativity of spectrum (e_1 >= e_2 >= ... >= 0).
6. Non-finite parameter rejection.
"""

import numpy as np
import pytest
from scipy.stats import ortho_group

from dce.core.effective_info import (
    compute_causal_spectrum,
    compute_gaussian_effective_information,
)


class TestCausalSpectrumAndDCD:
    """Test suite for Gate B2: Causal Spectrum & Dynamic Causal Dimensionality."""

    def test_causal_spectrum_matches_gaussian_effective_information(self):
        """Sum of causal information spectrum sum(e_i) must equal logdet EI exactly."""
        rng = np.random.RandomState(42)
        p = 5
        A = rng.randn(p, p) * 0.8
        Sigma_raw = rng.randn(p, p)
        Sigma = Sigma_raw @ Sigma_raw.T + 0.5 * np.eye(p)
        
        report = compute_causal_spectrum(A, Sigma, regularization=0.0)
        decomp = compute_gaussian_effective_information(A, Sigma, regularization=0.0)
        
        assert report.effective_information == pytest.approx(decomp.effective_information, rel=1e-12)
        assert np.sum(report.spectrum) == pytest.approx(decomp.effective_information, rel=1e-12)

    def test_rotational_invariance(self):
        """Under orthogonal coordinate transformation Q in O(p), spectrum and DCD are strictly invariant."""
        rng = np.random.RandomState(101)
        p = 6
        A = rng.randn(p, p)
        Sigma_raw = rng.randn(p, p)
        Sigma = Sigma_raw @ Sigma_raw.T + 0.3 * np.eye(p)
        
        report_orig = compute_causal_spectrum(A, Sigma, regularization=0.0)
        
        # Draw 5 random orthogonal matrices
        for seed in [1, 2, 3, 4, 5]:
            Q = ortho_group.rvs(p, random_state=seed)
            A_rot = Q @ A @ Q.T
            Sigma_rot = Q @ Sigma @ Q.T
            
            report_rot = compute_causal_spectrum(A_rot, Sigma_rot, regularization=0.0)
            
            # 1. Total EI invariant
            assert report_rot.effective_information == pytest.approx(report_orig.effective_information, rel=1e-10)
            # 2. Causal spectrum modes invariant
            np.testing.assert_allclose(report_rot.spectrum, report_orig.spectrum, rtol=1e-9, atol=1e-11)
            # 3. DCD Participation Ratio invariant
            assert report_rot.dcd_pr == pytest.approx(report_orig.dcd_pr, rel=1e-10)
            # 4. DCD Effective Rank (Entropy) invariant
            assert report_rot.dcd_entropy == pytest.approx(report_orig.dcd_entropy, rel=1e-10)

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_k_rank_recovery(self, k):
        """For a system with k identical active modes, DCD^PR and DCD^entropy must recover k exactly."""
        p = 16
        # Construct A with exactly k singular values equal to alpha, and p-k equal to 0
        alpha = 2.0
        A = np.zeros((p, p))
        for i in range(k):
            A[i, i] = alpha
            
        Sigma = np.eye(p)
        
        report = compute_causal_spectrum(A, Sigma, regularization=0.0)
        
        assert report.dcd_pr == pytest.approx(float(k), rel=1e-10)
        assert report.dcd_entropy == pytest.approx(float(k), rel=1e-10)
        assert np.count_nonzero(report.spectrum > 1e-10) == k

    def test_zero_dynamics_guard(self):
        """When A = 0, EI is 0, and DCD PR and entropy are 0.0 with no NaN/Inf or warnings."""
        p = 4
        A = np.zeros((p, p))
        Sigma = 2.0 * np.eye(p)
        
        report = compute_causal_spectrum(A, Sigma)
        assert report.effective_information == 0.0
        assert report.dcd_pr == 0.0
        assert report.dcd_entropy == 0.0
        assert np.all(report.spectrum == 0.0)
        assert np.all(report.normalized_weights == 0.0)

    def test_spectrum_monotonicity_and_positivity(self):
        """Spectrum modes e_i must be non-negative and sorted descending."""
        rng = np.random.RandomState(99)
        p = 8
        A = rng.randn(p, p)
        Sigma = rng.randn(p, p)
        Sigma = Sigma @ Sigma.T + np.eye(p)
        
        report = compute_causal_spectrum(A, Sigma)
        
        assert np.all(report.spectrum >= 0.0)
        assert np.all(report.eigenvalues >= 0.0)
        diffs = np.diff(report.spectrum)
        assert np.all(diffs <= 1e-12), "Spectrum must be sorted in descending order"

    def test_nonfinite_inputs_raise_value_error(self):
        """NaN or Inf in inputs must raise ValueError."""
        with pytest.raises(ValueError):
            compute_causal_spectrum(np.array([[np.nan]]), np.array([[1.0]]))
            
        with pytest.raises(ValueError):
            compute_causal_spectrum(np.array([[1.0]]), np.array([[np.inf]]))
