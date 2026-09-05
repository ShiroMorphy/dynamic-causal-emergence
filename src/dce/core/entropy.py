"""
Continuous Differential Entropy Estimators.

Implements:
1. Analytical Gaussian differential entropy (multivariate).
2. Non-parametric Kozachenko-Leonenko k-NN entropy estimator.
3. Conditional differential entropy and mutual information.
"""

from typing import Optional
import numpy as np
from scipy.special import digamma, gamma
from scipy.spatial import cKDTree


def gaussian_differential_entropy(
    cov_matrix: np.ndarray,
    regularization: float = 1e-6
) -> float:
    """
    Compute multivariate Gaussian differential entropy.
    
    H(X) = 0.5 * ln((2 * pi * e)^d * det(Sigma))
         = 0.5 * (d * ln(2 * pi * e) + ln(det(Sigma)))
    
    Args:
        cov_matrix: (d, d) Covariance matrix.
        regularization: Ridge regularization added to diagonal for numerical stability.
        
    Returns:
        Differential entropy in nats.
    """
    d = cov_matrix.shape[0]
    reg_cov = cov_matrix + regularization * np.eye(d)
    sign, logdet = np.linalg.slogdet(reg_cov)
    if sign <= 0:
        # Fallback to eigenvalue thresholding if slightly negative due to precision
        eigenvalues = np.linalg.eigvalsh(reg_cov)
        eigenvalues = np.clip(eigenvalues, a_min=regularization, a_max=None)
        logdet = float(np.sum(np.log(eigenvalues)))
        
    return float(0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet))


def knn_differential_entropy(
    x: np.ndarray,
    k: int = 5,
    p_norm: float = 2.0
) -> float:
    """
    Kozachenko-Leonenko k-NN differential entropy estimator.
    
    H(X) = -psi(k) + psi(N) + ln(c_d) + (d / N) * sum_{i=1}^N ln(2 * eps_i)
    where eps_i is the distance to the k-th nearest neighbor.
    
    Args:
        x: (N, d) Sample data matrix.
        k: Number of nearest neighbors (default: 5).
        p_norm: Lp norm for distance calculation (default: 2.0 - Euclidean).
        
    Returns:
        Estimated differential entropy in nats.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n_samples, d = x.shape
    
    if n_samples <= k:
        raise ValueError(f"Number of samples ({n_samples}) must be greater than k ({k}).")
        
    tree = cKDTree(x)
    # Query k+1 neighbors because the point itself is neighbor #1 at distance 0
    distances, _ = tree.query(x, k=k + 1, p=p_norm)
    eps = distances[:, -1]
    
    # Avoid log(0) for duplicate points
    eps = np.clip(eps, a_min=1e-12, a_max=None)
    
    # Volume of d-dimensional unit ball under L2 norm
    if p_norm == 2.0:
        unit_ball_volume = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
    else:
        # General Lp ball volume: (2 * Gamma(1 + 1/p))^d / Gamma(1 + d/p)
        unit_ball_volume = ((2.0 * gamma(1.0 + 1.0 / p_norm)) ** d) / gamma(1.0 + d / p_norm)
        
    h_est = (
        -digamma(k)
        + digamma(n_samples)
        + np.log(unit_ball_volume)
        + (d / n_samples) * np.sum(np.log(eps))
    )
    return float(h_est)


def conditional_gaussian_entropy(
    joint_cov: np.ndarray,
    target_dim: int,
    source_dim: int
) -> float:
    """
    Compute conditional differential entropy H(Y | X) under joint Gaussianity.
    
    H(Y | X) = H(Y, X) - H(X)
    
    Args:
        joint_cov: (target_dim + source_dim, target_dim + source_dim) Joint covariance.
        target_dim: Dimension of target variable Y.
        source_dim: Dimension of conditioning variable X.
        
    Returns:
        Conditional differential entropy in nats.
    """
    h_joint = gaussian_differential_entropy(joint_cov)
    cov_x = joint_cov[target_dim:, target_dim:]
    h_x = gaussian_differential_entropy(cov_x)
    return h_joint - h_x
