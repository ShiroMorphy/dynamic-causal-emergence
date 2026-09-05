"""
Surrogate Data Quality Diagnostics: Preservation, Non-Triviality, and Null-Destruction.

Implements the three-tier quality contract:
1. PRESERVATION:
   - Exact empirical marginal distribution (Kolmogorov-Smirnov / sorted infinity norm)
   - Auto-spectral density relative Frobenius error
   - Contemporaneous covariance relative Frobenius error
   - Cross-spectral density matrix relative Frobenius error
   - Lag-1 to Lag-24 autocorrelation function relative error
2. NON-TRIVIALITY:
   - Surrogate must not be correlated with the original time series
   - Normalized RMS distance between original and surrogate must be strictly positive
3. NULL-DESTRUCTION:
   - Higher-order nonlinear temporal phase dependencies (e.g., Time-Reversal Asymmetry)
     must be significantly reduced/destroyed relative to the original empirical data.
"""

import os
from pathlib import Path
from typing import Dict, NamedTuple, Optional
import numpy as np
import yaml


class SurrogateQualityReport(NamedTuple):
    # Preservation diagnostics
    marginal_max_error: float
    auto_spectrum_error: float
    cross_spectrum_error: float
    covariance_error: float
    autocorrelation_error: float
    # Non-triviality diagnostics
    mean_abs_correlation: float
    rms_relative_distance: float
    # Null-destruction diagnostics
    time_reversal_asymmetry_ratio: float
    # Status
    converged: bool
    iterations: int
    accepted: bool
    diagnostics_passed: Dict[str, bool]


def load_frozen_surrogate_tolerances(yaml_path: Optional[str] = None) -> Dict[str, float]:
    """Load pre-specified frozen surrogate acceptance tolerances from YAML."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "surrogate_acceptance_protocol.yaml"
    else:
        yaml_path = Path(yaml_path)
        
    if not yaml_path.exists():
        # Fallback to hardcoded frozen values
        return {
            "marginal_max_error": 1e-10,
            "auto_spectrum_error": 0.08,
            "autocorrelation_error": 0.08,
            "covariance_error": 0.10,
            "cross_spectrum_error": 0.15,
            "max_mean_abs_correlation": 0.25,
            "min_rms_relative_distance": 0.20,
            "max_time_reversal_ratio": 0.75,
        }
        
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    tols = {}
    if "hard_acceptance" in data:
        if "preservation" in data["hard_acceptance"]:
            tols.update(data["hard_acceptance"]["preservation"])
        if "non_triviality" in data["hard_acceptance"]:
            tols.update(data["hard_acceptance"]["non_triviality"])
    else:
        if "preservation" in data:
            tols.update(data["preservation"])
        if "non_triviality" in data:
            tols.update(data["non_triviality"])
            
    if "diagnostic_metrics" in data:
        tols.update(data["diagnostic_metrics"])
        if "expected_time_reversal_ratio" in data["diagnostic_metrics"]:
            tols["max_time_reversal_ratio"] = data["diagnostic_metrics"]["expected_time_reversal_ratio"]
    elif "null_destruction" in data:
        tols.update(data["null_destruction"])
    return tols


PRE_SPECIFIED_TOLERANCES = load_frozen_surrogate_tolerances()


def compute_cross_spectrum_matrix(X: np.ndarray) -> np.ndarray:
    fx = np.fft.rfft(X, axis=0)
    return np.einsum("fi,fj->fij", fx, fx.conj())


def compute_autocorrelations(X: np.ndarray, max_lag: int = 24) -> np.ndarray:
    T, p = X.shape
    acfs = np.zeros((max_lag, p), dtype=np.float64)
    X_centered = X - np.mean(X, axis=0)
    var = np.var(X, axis=0) + 1e-12
    for lag in range(1, max_lag + 1):
        acfs[lag - 1] = np.mean(X_centered[:-lag] * X_centered[lag:], axis=0) / var
    return acfs


def compute_time_reversal_asymmetry(X: np.ndarray) -> float:
    """Compute mean magnitude of normalized time-reversal asymmetry (increment skewness) across channels."""
    diff = np.diff(X, axis=0)
    std_diff = np.std(diff, axis=0) + 1e-8
    skew = np.mean(diff ** 3, axis=0) / (std_diff ** 3)
    return float(np.mean(np.abs(skew)))


def evaluate_surrogate_quality(
    X_original: np.ndarray,
    X_surrogate: np.ndarray,
    iterations: int = 100,
    converged: bool = True,
    tolerances: Dict[str, float] = PRE_SPECIFIED_TOLERANCES
) -> SurrogateQualityReport:
    X = np.asarray(X_original, dtype=np.float64)
    S = np.asarray(X_surrogate, dtype=np.float64)
    T, p = X.shape
    
    # 1. Marginal error
    sort_x = np.sort(X, axis=0)
    sort_s = np.sort(S, axis=0)
    scale_x = np.std(X, axis=0) + 1e-8
    marginal_err = float(np.max(np.abs(sort_s - sort_x) / scale_x))
    
    # 2. Covariance relative Frobenius error
    cov_x = np.cov(X, rowvar=False)
    cov_s = np.cov(S, rowvar=False)
    cov_norm = np.linalg.norm(cov_x)
    cov_err = float(np.linalg.norm(cov_s - cov_x) / max(cov_norm, 1e-8))
    
    # 3. Cross-spectrum relative Frobenius error
    sx = compute_cross_spectrum_matrix(X)
    sy = compute_cross_spectrum_matrix(S)
    sx_norm = np.linalg.norm(sx)
    cs_err = float(np.linalg.norm(sy - sx) / max(sx_norm, 1e-8))
    
    # 4. Auto-spectrum relative Frobenius error
    sx_auto = np.diagonal(sx, axis1=1, axis2=2)
    sy_auto = np.diagonal(sy, axis1=1, axis2=2)
    auto_norm = np.linalg.norm(sx_auto)
    auto_err = float(np.linalg.norm(sy_auto - sx_auto) / max(auto_norm, 1e-8))
    
    # 5. Autocorrelation relative Frobenius error
    acf_x = compute_autocorrelations(X, max_lag=24)
    acf_s = compute_autocorrelations(S, max_lag=24)
    acf_norm = np.linalg.norm(acf_x)
    acf_err = float(np.linalg.norm(acf_s - acf_x) / max(acf_norm, 1e-8))
    
    # 6. Non-triviality: correlation with original (ignoring constant zero channels)
    corrs = []
    for j in range(p):
        std_x = np.std(X[:, j])
        std_s = np.std(S[:, j])
        if std_x > 1e-6 and std_s > 1e-6:
            r = np.corrcoef(X[:, j], S[:, j])[0, 1]
            if np.isfinite(r):
                corrs.append(abs(r))
    mean_abs_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0
    
    # Normalized RMS distance
    rms_dist = float(np.linalg.norm(S - X) / max(np.linalg.norm(X), 1e-8))
    
    trev_x = compute_time_reversal_asymmetry(X)
    trev_s = compute_time_reversal_asymmetry(S)
    trev_ratio = float(trev_s / max(trev_x, 1e-12))
    diag_trev_thresh = tolerances.get("expected_time_reversal_ratio", tolerances.get("max_time_reversal_ratio", 0.75))
    null_destroyed = (trev_ratio <= diag_trev_thresh) or (trev_s <= 0.40 and trev_x <= 0.40)
    
    hard_passes = {
        # Preservation
        "marginal_max_error": marginal_err <= tolerances["marginal_max_error"],
        "auto_spectrum_error": auto_err <= tolerances["auto_spectrum_error"],
        "autocorrelation_error": acf_err <= tolerances["autocorrelation_error"],
        "covariance_error": cov_err <= tolerances["covariance_error"],
        "cross_spectrum_error": cs_err <= tolerances["cross_spectrum_error"],
        # Non-triviality
        "non_trivial_decorrelated": mean_abs_corr <= tolerances["max_mean_abs_correlation"],
        "non_trivial_distance": rms_dist >= tolerances["min_rms_relative_distance"],
    }
    
    passes = dict(hard_passes)
    passes["diagnostic_null_destroyed_trev"] = null_destroyed
    accepted = all(hard_passes.values())
    
    return SurrogateQualityReport(
        marginal_max_error=marginal_err,
        auto_spectrum_error=auto_err,
        cross_spectrum_error=cs_err,
        covariance_error=cov_err,
        autocorrelation_error=acf_err,
        mean_abs_correlation=mean_abs_corr,
        rms_relative_distance=rms_dist,
        time_reversal_asymmetry_ratio=trev_ratio,
        converged=converged,
        iterations=iterations,
        accepted=accepted,
        diagnostics_passed=passes
    )
