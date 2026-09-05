"""
Complexity and Information-Theoretic Baselines.

Implements:
1. Dynamic Permutation Entropy (Bandt & Pompe 2002).
2. Dynamic Effective Rank (Roy & Vetterli 2007).
3. Dynamic O-Information (Higher-Order Synergy vs Redundancy, Rosas et al. 2019).
"""

from typing import Tuple
import numpy as np
from scipy.special import factorial
from dce.core.kernels import compute_temporal_weights


def permutation_entropy_1d(series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute permutation entropy of a 1D time series."""
    n = len(series)
    if n < order:
        return 0.0
    n_vectors = n - (order - 1) * delay
    if n_vectors <= 0:
        return 0.0
        
    patterns = np.zeros((n_vectors, order))
    for i in range(order):
        patterns[:, i] = series[i * delay : i * delay + n_vectors]
        
    # Get ordinal patterns
    ranks = np.argsort(patterns, axis=1)
    # Hash ordinal pattern into unique integer
    weights = np.cumprod([1] + list(range(1, order)))[::-1]
    pattern_hashes = np.sum(ranks * weights, axis=1)
    
    _, counts = np.unique(pattern_hashes, return_counts=True)
    probs = counts / float(np.sum(counts))
    
    # Shannon entropy normalized by ln(order!)
    pe = -np.sum(probs * np.log(probs + 1e-12))
    pe_norm = pe / np.log(factorial(order))
    return float(pe_norm)


def compute_dynamic_effective_rank(
    X: np.ndarray,
    bandwidth: float = 24.0,
    kernel_type: str = "gaussian"
) -> np.ndarray:
    """
    Compute Roy & Vetterli Effective Rank over time.
    
    erank(Sigma) = exp( - sum_i p_i ln(p_i) ), where p_i = lambda_i / sum_j lambda_j.
    """
    n_samples, p_dim = X.shape
    erank_traj = np.zeros(n_samples, dtype=np.float64)
    
    for t in range(n_samples):
        weights = compute_temporal_weights(t, n_samples, bandwidth, kernel_type)
        mean_t = np.sum(weights[:, np.newaxis] * X, axis=0)
        x_c = (X - mean_t) * np.sqrt(weights)[:, np.newaxis]
        
        _, s, _ = np.linalg.svd(x_c, full_matrices=False)
        singular_sum = np.sum(s) + 1e-12
        p_sing = s / singular_sum
        p_sing = p_sing[p_sing > 1e-10]
        
        entropy_s = -np.sum(p_sing * np.log(p_sing))
        erank_traj[t] = float(np.exp(entropy_s))
        
    return erank_traj


def compute_dynamic_o_information(
    X: np.ndarray,
    bandwidth: float = 24.0,
    kernel_type: str = "gaussian"
) -> np.ndarray:
    """
    Compute Dynamic O-Information (Omega-Information) of multivariate state X.
    
    Omega(X) = (p - 2) H(X) + sum_{i=1}^p [ H(X_i) - H(X_{-i}) ]
    Positive Omega -> Redundancy dominated
    Negative Omega -> Synergy dominated
    """
    n_samples, p = X.shape
    o_info_traj = np.zeros(n_samples, dtype=np.float64)
    
    for t in range(n_samples):
        weights = compute_temporal_weights(t, n_samples, bandwidth, kernel_type)
        mean_t = np.sum(weights[:, np.newaxis] * X, axis=0)
        x_c = (X - mean_t) * np.sqrt(weights)[:, np.newaxis]
        cov = (x_c.T @ x_c) / (np.sum(weights) + 1e-8) + 1e-4 * np.eye(p)
        
        # Joint entropy H(X)
        _, logdet_joint = np.linalg.slogdet(cov)
        h_joint = 0.5 * (p * np.log(2.0 * np.pi * np.e) + logdet_joint)
        
        sum_marginal_diffs = 0.0
        for i in range(p):
            # Marginal H(X_i)
            var_i = cov[i, i]
            h_i = 0.5 * np.log(2.0 * np.pi * np.e * var_i + 1e-8)
            
            # Leave-one-out H(X_{-i})
            mask = np.ones(p, dtype=bool)
            mask[i] = False
            cov_loo = cov[mask][:, mask]
            _, logdet_loo = np.linalg.slogdet(cov_loo)
            h_loo = 0.5 * ((p - 1) * np.log(2.0 * np.pi * np.e) + logdet_loo)
            
            sum_marginal_diffs += (h_i - h_loo)
            
        o_info = (p - 2.0) * h_joint + sum_marginal_diffs
        o_info_traj[t] = float(o_info)
        
    return o_info_traj
