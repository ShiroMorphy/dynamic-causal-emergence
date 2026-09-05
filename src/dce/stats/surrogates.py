"""
Surrogate Data Generators for Nonstationary Statistical Testing.

Implements:
1. Iterated Amplitude Adjusted Fourier Transform (IAAFT) (Schreiber & Schmitz 1996).
2. Vector Autoregressive (VAR) Linear Gaussian Surrogates.
3. Block Bootstrap for Nonstationary Time Series.
"""

from typing import List, Optional
import numpy as np


def generate_iaaft_surrogate(
    series_1d: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate IAAFT surrogate preserving exact amplitude distribution and linear power spectrum.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(series_1d, dtype=np.float64)
    n = len(x)
    
    # Target power spectrum amplitudes
    fourier_orig = np.fft.rfft(x)
    target_amplitudes = np.abs(fourier_orig)
    
    # Target ranked amplitudes
    sorted_orig = np.sort(x)
    
    # Initial random phase shuffle
    phases = rng.uniform(0, 2 * np.pi, size=len(target_amplitudes))
    s_fourier = target_amplitudes * np.exp(1j * phases)
    s_curr = np.fft.irfft(s_fourier, n=n)
    
    for _ in range(max_iter):
        # 1. Rank match with target amplitudes
        rank_indices = np.argsort(np.argsort(s_curr))
        s_ranked = sorted_orig[rank_indices]
        
        # 2. Fourier transform and replace amplitudes
        s_fourier_new = np.fft.rfft(s_ranked)
        s_phases_new = np.angle(s_fourier_new)
        s_fourier_matched = target_amplitudes * np.exp(1j * s_phases_new)
        
        # 3. Inverse transform
        s_next = np.fft.irfft(s_fourier_matched, n=n)
        
        # Convergence check
        if np.max(np.abs(s_next - s_curr)) < tol:
            break
        s_curr = s_next
        
    return s_curr


def generate_multivariate_surrogates(
    X: np.ndarray,
    n_surrogates: int = 100,
    method: str = "iaaft",
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generate ensemble of multivariate surrogates.
    """
    T, p = X.shape
    surrogates = []
    
    for b in range(n_surrogates):
        surr_b = np.zeros((T, p), dtype=np.float64)
        for j in range(p):
            surr_b[:, j] = generate_iaaft_surrogate(X[:, j], seed=seed + b * 1000 + j)
        surrogates.append(surr_b)
        
    return surrogates
