"""
Gate A2: Exact Oracle Test for Liu, Yuan & Zhang (2024) vs Gaussian Channel DCE.

Paper Reference:
Liu, Yuan & Zhang, "A Mathematical Framework for Causal Emergence in Continuous Dynamical Systems",
Entropy 2024, 26(8), 618; arXiv:2405.09207.

Key Mathematical Distinctions:
1. Liu et al. (2024):
   - Continuous intervention over a compact hypercube: do(X) ~ Uniform([-L, L]^n)
   - Continuous Effective Information:
     EI_{Liu}(A, Sigma; L) = ln [ |det(A)| * L^n / ((2*pi*e)^{n/2} * det(Sigma)^{1/2}) ]
                           = ln |det(A)| + n * ln(L) - (n/2)*ln(2*pi*e) - 0.5 * ln det(Sigma)
   - Dimension-normalized EI: J_{Liu} = EI_{Liu} / n
   - Properties:
     * Requires det(A) != 0 (diverges to -infinity if A is singular or rank-deficient)
     * Value depends monotonically on hypercube half-width L: d(EI)/dL = n/L > 0
     * Can become negative if L or |det(A)| is small relative to noise Sigma

2. Gaussian Channel Framework (Our DCD/DCE formulation):
   - Reference intervention: Maximum entropy distribution with covariance constraint Sigma_{do} = I_p:
     do(X) ~ N(0, I_p)
   - Closed-form mutual information:
     EI_{Gauss}(A, Sigma) = 0.5 * ln det(I_p + Sigma^{-1} A A^T) = 0.5 * sum_{i=1}^p ln(1 + lambda_i)
     where lambda_i >= 0 are the generalized eigenvalues of (A A^T, Sigma).
   - Properties:
     * Scale-free: no arbitrary domain bound parameter L
     * Unconditionally non-negative: EI_{Gauss} >= 0, with EI_{Gauss} = 0 iff A = 0
     * Well-defined even when A is singular or rank-deficient (det(A) = 0)
     * Invariant under orthogonal coordinate changes Q in O(p)
"""

import numpy as np
import pytest

from dce.core.effective_info import compute_gaussian_effective_information


def compute_liu2024_effective_information(
    A: np.ndarray,
    Sigma: np.ndarray,
    L: float = 1.0
) -> float:
    """
    Compute continuous Effective Information strictly according to Liu et al. (2024, Eq. 8).
    
    EI_{Liu}(A, Sigma; L) = ln |det(A)| + n*ln(L) - (n/2)*ln(2*pi*e) - 0.5*ln det(Sigma)
    """
    A = np.asarray(A, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = A.shape[0]
    
    det_A = np.linalg.det(A)
    if np.abs(det_A) <= 1e-15:
        raise ValueError("Liu et al. (2024) formula requires non-singular A (det(A) != 0).")
        
    sign_sigma, logdet_sigma = np.linalg.slogdet(Sigma)
    if sign_sigma <= 0:
        raise ValueError("Noise covariance Sigma must be positive definite.")
        
    ei_liu = (
        np.log(np.abs(det_A))
        + n * np.log(L)
        - 0.5 * n * np.log(2.0 * np.pi * np.e)
        - 0.5 * logdet_sigma
    )
    return float(ei_liu)


def compute_liu2024_normalized_ei(
    A: np.ndarray,
    Sigma: np.ndarray,
    L: float = 1.0
) -> float:
    """Compute dimension-normalized continuous EI: J = EI / n (Liu et al. 2024)."""
    n = A.shape[0]
    return compute_liu2024_effective_information(A, Sigma, L) / float(n)


class TestLiu2024OracleDistinction:
    """Rigorous verification of Liu et al. (2024) vs Gaussian Channel DCD."""

    def test_liu2024_literal_formula_reproduction(self):
        """Verify the exact algebraic identity of the Liu et al. formula."""
        n = 2
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        Sigma = np.array([[0.5, 0.0], [0.0, 0.5]])
        L = 2.0
        
        # Manual analytical calculation:
        # |det(A)| = 6
        # L^n = 4
        # (2*pi*e)^n = (2*pi*e)^2
        # det(Sigma) = 0.25 -> sqrt(det(Sigma)) = 0.5
        # Ratio = 6 * 4 / ((2*pi*e) * 0.5) = 48 / (2*pi*e)
        expected_ei = np.log(6.0 * 4.0 / (2.0 * np.pi * np.e * 0.5))
        
        computed_ei = compute_liu2024_effective_information(A, Sigma, L=L)
        assert computed_ei == pytest.approx(expected_ei, rel=1e-12)
        
        # Normalized J = EI / 2
        computed_j = compute_liu2024_normalized_ei(A, Sigma, L=L)
        assert computed_j == pytest.approx(expected_ei / 2.0, rel=1e-12)

    def test_liu2024_domain_bound_sensitivity(self):
        """Demonstrate that Liu et al.'s EI depends strictly on arbitrary bound L."""
        A = np.eye(2)
        Sigma = 0.1 * np.eye(2)
        
        ei_L1 = compute_liu2024_effective_information(A, Sigma, L=1.0)
        ei_L10 = compute_liu2024_effective_information(A, Sigma, L=10.0)
        
        # Theoretical delta: n * ln(10 / 1) = 2 * ln(10)
        expected_delta = 2.0 * np.log(10.0)
        assert (ei_L10 - ei_L1) == pytest.approx(expected_delta, rel=1e-12)

    def test_liu2024_can_be_negative_for_small_L(self):
        """Unlike Gaussian EI (which is >= 0), Liu's continuous EI can be negative."""
        A = np.eye(2)
        Sigma = np.eye(2)
        L = 0.1  # small bounded support
        
        ei_liu = compute_liu2024_effective_information(A, Sigma, L=L)
        assert ei_liu < 0.0, "Liu EI should be negative when support volume is smaller than noise entropy."

    def test_liu2024_fails_on_singular_A(self):
        """Liu et al.'s formula diverges when A has reduced rank (det A = 0)."""
        A_singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        Sigma = np.eye(2)
        
        with pytest.raises(ValueError, match="requires non-singular A"):
            compute_liu2024_effective_information(A_singular, Sigma, L=1.0)

    def test_gaussian_channel_well_defined_on_singular_A(self):
        """Our Gaussian channel EI is well-defined, finite, and strictly non-negative on singular A."""
        A_singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        Sigma = np.eye(2)
        
        decomp = compute_gaussian_effective_information(A_singular, Sigma, regularization=0.0)
        # Expected: 0.5 * ln(1 + 1.0) + 0.5 * ln(1 + 0.0) = 0.5 * ln(2)
        assert decomp.effective_information == pytest.approx(0.5 * np.log(2.0), rel=1e-12)
        assert decomp.effective_information >= 0.0

    def test_conceptual_contrast_summary(self):
        """
        Verify that our Gaussian channel and Liu's uniform EI are distinct functionals,
        confirming that dimension-normalization in Liu (J = EI/n) is conceptually analogous
        in motivation (normalizing by dimension) but mathematically distinct in formulation.
        """
        A = np.array([[1.5, 0.3], [-0.2, 0.8]])
        Sigma = np.array([[0.4, 0.1], [0.1, 0.3]])
        
        ei_gauss = compute_gaussian_effective_information(A, Sigma).effective_information
        ei_liu = compute_liu2024_effective_information(A, Sigma, L=1.0)
        
        # They should differ numerically because they measure different quantities:
        # Mutual information under N(0, I) vs differential entropy volume under U([-L, L]^n)
        assert abs(ei_gauss - ei_liu) > 0.05
