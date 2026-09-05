"""
Temporal Weighting Kernels and Bandwidth Selection for Nonstationary Dynamics.

Provides symmetric and causal (one-sided) kernels for local state-space estimation.
"""

from typing import Callable, Optional, Union
import numpy as np


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel: K(u) = (1 / sqrt(2*pi)) * exp(-0.5 * u^2)."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (u ** 2))


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov parabolic kernel: K(u) = 0.75 * (1 - u^2) for |u| <= 1, 0 otherwise."""
    weights = 0.75 * (1.0 - u ** 2)
    weights[np.abs(u) > 1.0] = 0.0
    return weights


def tricube_kernel(u: np.ndarray) -> np.ndarray:
    """Tricube kernel: K(u) = (70/81) * (1 - |u|^3)^3 for |u| <= 1, 0 otherwise."""
    abs_u = np.abs(u)
    weights = (70.0 / 81.0) * ((1.0 - abs_u ** 3) ** 3)
    weights[abs_u > 1.0] = 0.0
    return weights


def exponential_kernel(u: np.ndarray) -> np.ndarray:
    """Exponential (Laplace) kernel: K(u) = 0.5 * exp(-|u|)."""
    return 0.5 * np.exp(-np.abs(u))


KERNEL_REGISTRY = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "tricube": tricube_kernel,
    "exponential": exponential_kernel,
}


def compute_temporal_weights(
    center_t: int,
    total_t: int,
    bandwidth: float,
    kernel_type: str = "gaussian",
    causal_only: bool = False
) -> np.ndarray:
    """
    Compute normalized temporal weights centered around index center_t.
    
    w_{t, s} = K((s - t) / h) / sum_{tau} K((tau - t) / h)
    
    Args:
        center_t: Target time index (0-indexed).
        total_t: Total number of timestamps T.
        bandwidth: Kernel bandwidth h > 0.
        kernel_type: Name of kernel ('gaussian', 'epanechnikov', 'tricube', 'exponential').
        causal_only: If True, only use past and present observations (s <= center_t).
        
    Returns:
        (total_t,) Normalized probability vector of weights.
    """
    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be strictly positive, got {bandwidth}.")
        
    if kernel_type not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel type '{kernel_type}'. Options: {list(KERNEL_REGISTRY.keys())}")
        
    indices = np.arange(total_t, dtype=np.float64)
    time_diff = (indices - float(center_t)) / float(bandwidth)
    
    kernel_func = KERNEL_REGISTRY[kernel_type]
    raw_weights = kernel_func(time_diff)
    
    if causal_only:
        raw_weights[indices > center_t] = 0.0
        
    sum_w = np.sum(raw_weights)
    if sum_w <= 1e-12:
        # Fallback to delta distribution at center_t if bandwidth is too small
        weights = np.zeros(total_t, dtype=np.float64)
        weights[center_t] = 1.0
        return weights
        
    return raw_weights / sum_w


def compute_retrospective_weights(
    center_t: int,
    total_t: int,
    bandwidth: float,
    kernel_type: str = "gaussian"
) -> np.ndarray:
    """
    Symmetric retrospective temporal kernel weights:
        w_{t, s}^{retro} = K((s - t) / h) / sum_{tau} K((tau - t) / h)
    """
    return compute_temporal_weights(
        center_t=center_t,
        total_t=total_t,
        bandwidth=bandwidth,
        kernel_type=kernel_type,
        causal_only=False
    )


def compute_causal_weights(
    center_t: int,
    total_t: int,
    bandwidth: float,
    kernel_type: str = "gaussian"
) -> np.ndarray:
    """
    Strictly one-sided causal online temporal kernel weights:
        w_{t, s}^{causal} = K((t - s) / h) * 1(s <= t) / sum_{tau <= t} K((t - tau) / h)
    """
    return compute_temporal_weights(
        center_t=center_t,
        total_t=total_t,
        bandwidth=bandwidth,
        kernel_type=kernel_type,
        causal_only=True
    )


def audit_kernel_leakage(weights: np.ndarray, center_t: int, causal_only: bool = True) -> bool:
    """
    Verify that temporal weights do not leak future information.
    For causal mode, all weights strictly past center_t must be 0.0.
    """
    if not causal_only:
        return True
    if center_t >= len(weights) - 1:
        return True
    future_weights = weights[center_t + 1:]
    is_clean = bool(np.all(future_weights == 0.0))
    if not is_clean:
        max_leakage = float(np.max(future_weights))
        raise AssertionError(f"Future data leakage detected! Max future weight = {max_leakage}")
    return True
def select_bandwidth_cv(
    x_series: np.ndarray,
    candidate_bandwidths: list[float],
    kernel_type: str = "gaussian",
    step_size: int = 10
) -> float:
    """
    Select optimal bandwidth h via Rolling-Origin Time-Series Cross Validation.
    
    Minimizes 1-step-ahead forward prediction error.
    """
    n_samples, _ = x_series.shape
    best_h = candidate_bandwidths[0]
    min_error = float("inf")
    
    eval_points = np.arange(int(n_samples * 0.2), int(n_samples * 0.9), step_size)
    
    for h in candidate_bandwidths:
        errors = []
        for t in eval_points:
            w = compute_temporal_weights(t, n_samples, bandwidth=h, kernel_type=kernel_type, causal_only=True)
            # Weighted average state
            x_pred = np.sum(w[:t, np.newaxis] * x_series[:t], axis=0) / (np.sum(w[:t]) + 1e-10)
            errors.append(np.mean((x_series[t] - x_pred) ** 2))
        avg_err = float(np.mean(errors))
        if avg_err < min_error:
            min_error = avg_err
            best_h = h
            
    return best_h
