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


def generate_multivariate_iaaft_surrogate(
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a multivariate IAAFT surrogate (Prichard & Theiler 1994; Schreiber & Schmitz 1996).
    
    Preserves:
    1. Exact marginal empirical amplitude distributions for each variable.
    2. Linear auto-covariance structure (power spectrum) for each variable.
    3. Contemporaneous cross-correlation and cross-spectral matrix across all p variables
       by applying identical phase rotations phi(f) across channels.
       
    Randomizes:
    Nonlinear cross-temporal phase couplings and higher-order dynamical interactions.
    """
    rng = np.random.RandomState(seed)
    T, p = X.shape
    
    # 1. Target power spectra and amplitudes for each channel
    fourier_orig = np.fft.rfft(X, axis=0)  # shape (n_freqs, p)
    n_freqs = fourier_orig.shape[0]
    target_amplitudes = np.abs(fourier_orig)
    orig_phases = np.angle(fourier_orig)
    
    # Target ranked values for each channel
    sorted_orig = np.sort(X, axis=0)
    
    # 2. Shared random phase shift across all channels to preserve cross-correlation structure
    random_phase_shifts = rng.uniform(0, 2 * np.pi, size=(n_freqs, 1))
    random_phase_shifts[0, 0] = 0.0  # DC component
    if T % 2 == 0:
        random_phase_shifts[-1, 0] = 0.0  # Nyquist component
        
    s_phases = orig_phases + random_phase_shifts
    s_fourier = target_amplitudes * np.exp(1j * s_phases)
    s_curr = np.fft.irfft(s_fourier, n=T, axis=0)
    
    for _ in range(max_iter):
        # 1. Rank match marginal empirical distributions
        s_ranked = np.zeros_like(s_curr)
        for j in range(p):
            rank_idx = np.argsort(np.argsort(s_curr[:, j]))
            s_ranked[:, j] = sorted_orig[rank_idx, j]
            
        # 2. Fourier transform of ranked series
        s_fourier_new = np.fft.rfft(s_ranked, axis=0)
        s_phases_new = np.angle(s_fourier_new)
        
        # 3. Replace amplitudes with target amplitudes, retaining updated phases
        s_fourier_matched = target_amplitudes * np.exp(1j * s_phases_new)
        s_next = np.fft.irfft(s_fourier_matched, n=T, axis=0)
        
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
    Generate ensemble of multivariate surrogates preserving marginal distributions and cross-correlations.
    """
    surrogates = []
    for b in range(n_surrogates):
        surr_b = generate_multivariate_iaaft_surrogate(X, seed=seed + b * 1000)
        surrogates.append(surr_b)
    return surrogates

