"""
Dynamic Network Topology and Graph Metrics.

Implements:
1. Dynamic Correlation Graph Modularity (Newman-Girvan community structure).
2. Dynamic Graph Laplacian Spectral Gap.
"""

import numpy as np
import networkx as nx
from dce.core.kernels import compute_temporal_weights


def compute_dynamic_network_metrics(
    X: np.ndarray,
    threshold: float = 0.3,
    bandwidth: float = 24.0,
    kernel_type: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Graph Modularity and Laplacian Spectral Gap over time.
    
    Returns:
        (modularity_traj, spectral_gap_traj)
    """
    n_samples, p = X.shape
    modularity_traj = np.zeros(n_samples, dtype=np.float64)
    spectral_gap_traj = np.zeros(n_samples, dtype=np.float64)
    
    for t in range(n_samples):
        weights = compute_temporal_weights(t, n_samples, bandwidth, kernel_type)
        mean_t = np.sum(weights[:, np.newaxis] * X, axis=0)
        x_c = (X - mean_t) * np.sqrt(weights)[:, np.newaxis]
        cov = (x_c.T @ x_c) / (np.sum(weights) + 1e-8)
        
        # Correlation matrix
        stds = np.sqrt(np.diag(cov)) + 1e-8
        corr = cov / np.outer(stds, stds)
        
        # Adjacency matrix: thresholded absolute correlation
        adj = np.abs(corr)
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        
        # Build NetworkX graph
        G = nx.from_numpy_array(adj)
        
        # Modularity
        if G.number_of_edges() > 0:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
        else:
            modularity = 0.0
        modularity_traj[t] = float(modularity)
        
        # Normalized Laplacian Spectral Gap: lambda_2 (algebraic connectivity)
        degrees = np.sum(adj, axis=1)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-6)))
        laplacian_norm = np.eye(p) - deg_inv_sqrt @ adj @ deg_inv_sqrt
        eigenvals = np.sort(np.linalg.eigvalsh(laplacian_norm))
        
        spectral_gap = eigenvals[1] if p > 1 else 0.0
        spectral_gap_traj[t] = float(spectral_gap)
        
    return modularity_traj, spectral_gap_traj
