"""
Effective Information (EI) and Causal Effectiveness for Continuous Multivariate Systems.

Formulates:
EI = I_{nu}(do(X_t); X_{t+1}) = H_{nu}(X_{t+1}) - E_{X_t ~ nu}[H(P_t(X_{t+1} | do(X_t)))]
Determinism (Effectiveness) = H_{nu}(X_{t+1})
Degeneracy / Noise = H(epsilon_t) = E[H(P_t(X_{t+1} | X_t))]

For linear-Gaussian dynamics:
    X_{t+1} = A * X_t + epsilon,   epsilon ~ N(0, Sigma)
    do(X_t) ~ nu with Cov(do(X_t)) = Sigma_{do}
    
    Output covariance: Sigma_{out} = A * Sigma_{do} * A^T + Sigma
    EI = 0.5 * [ ln det(Sigma_{out}) - ln det(Sigma) ]
"""

from typing import NamedTuple, Optional, Tuple, Union
import numpy as np

import scipy.linalg
from dce.core.entropy import gaussian_differential_entropy
from dce.core.interventions import (
    BaseIntervention,
    GaussianMaxEntropyIntervention,
    UniformCompactIntervention,
    get_intervention,
)


class CausalDecomposition(NamedTuple):
    """Container for Information-Theoretic Causal Emergence metrics."""
    effective_information: float
    determinism: float
    degeneracy: float
    effectiveness: float


class CausalSpectrumReport(NamedTuple):
    """Container for Causal Information Spectrum and Dynamic Causal Dimensionality."""
    spectrum: np.ndarray           # e_i = 0.5 * ln(1 + lambda_i) sorted descending
    eigenvalues: np.ndarray        # lambda_i of Sigma^{-1} A A^T sorted descending
    effective_information: float   # sum e_i
    dcd_pr: float                  # Causal Participation Ratio (sum e_i)^2 / sum e_i^2
    dcd_entropy: float             # Causal Effective Rank exp(-sum pi_i ln pi_i)
    normalized_weights: np.ndarray # pi_i = e_i / EI
    projection_v: Optional[np.ndarray] = None # Fisher causal right-singular vectors (columns of V in M = U S V^T)


def _stable_logdet(matrix: np.ndarray, reg: float = 1e-8, max_condition: float = 1e8) -> float:
    """
    Compute log determinant using Cholesky with condition number control and fallback to slogdet.
    Guarantees symmetry, positive-definiteness, and bounded conditioning: kappa(Sigma) <= max_condition.
    """
    d = matrix.shape[0]
    sym = 0.5 * (matrix + matrix.T)
    w_eigs = np.linalg.eigvalsh(sym)
    min_eig = float(w_eigs[0])
    max_eig = float(w_eigs[-1])
    
    req_min = max(reg, 1e-12)
    if max_eig > 0.0 and (max_eig / max_condition) > req_min:
        req_min = max_eig / max_condition
        
    if min_eig < req_min:
        sym = sym + (req_min - min_eig) * np.eye(d)
        
    try:
        L = np.linalg.cholesky(sym)
        return float(2.0 * np.sum(np.log(np.diag(L))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(sym)
        if sign <= 0:
            sym = sym + 1e-4 * np.eye(d)
            _, logdet = np.linalg.slogdet(sym)
        return float(logdet)


def compute_gaussian_effective_information(
    transition_matrix_A: np.ndarray,
    noise_covariance_Sigma: np.ndarray,
    intervention: Optional[Union[BaseIntervention, str]] = None,
    domain_bound: Optional[float] = None,
    regularization: float = 1e-6
) -> CausalDecomposition:
    """
    Compute closed-form Effective Information for linear-Gaussian transition:
        X_{t+1} = A * X_t + epsilon,   epsilon ~ N(0, Sigma)
        
    Reference intervention: Maximum-entropy Gaussian distribution under covariance constraint Sigma_{do} = I_p:
        do(X_t) ~ N(0, I_p)
        
    Args:
        transition_matrix_A: (d, d) local transition matrix A
        noise_covariance_Sigma: (d, d) transition noise covariance Sigma
        intervention: BaseIntervention instance or name ('gaussian', 'uniform', 'whitened').
        domain_bound: (Deprecated) Hypercube half-width for uniform intervention.
        regularization: Numerical ridge for covariance conditioning.
        
    Returns:
        CausalDecomposition with effective_information, determinism, degeneracy, effectiveness.
    """
    A_arr = np.asarray(transition_matrix_A, dtype=np.float64)
    Sigma_arr = np.asarray(noise_covariance_Sigma, dtype=np.float64)
    
    # Contract: non-finite parameters must be rejected, not silently replaced
    if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(Sigma_arr)):
        raise ValueError("Transition matrix A and noise covariance Sigma must contain finite numbers.")
    
    d = A_arr.shape[0]
    if d == 0:
        raise ValueError("Cannot compute EI for zero-dimensional state.")
        
    # Contract: Uniform intervention must not use Gaussian entropy formula
    if intervention == "uniform" or isinstance(intervention, UniformCompactIntervention) or domain_bound is not None:
        raise NotImplementedError("Uniform continuous intervention is not supported under the Gaussian differential entropy formula.")
        
    Sigma_sym = 0.5 * (Sigma_arr + Sigma_arr.T) + regularization * np.eye(d)
    logdet_noise = _stable_logdet(Sigma_sym, reg=regularization)
    h_noise = 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet_noise)
    
    # Contract: A=0 makes intervention and output independent, hence EI is exactly zero
    if np.allclose(A_arr, 0.0):
        return CausalDecomposition(
            effective_information=0.0,
            determinism=float(h_noise),
            degeneracy=float(h_noise),
            effectiveness=0.0
        )
        
    # Resolve intervention: default to standard Gaussian max-entropy Sigma_{do} = I_d
    if intervention is None:
        interv_obj = GaussianMaxEntropyIntervention(dim=d, sigma_do=1.0)
    elif isinstance(intervention, str):
        interv_obj = get_intervention(intervention, dim=d)
    else:
        interv_obj = intervention
        
    sigma_do = interv_obj.covariance_matrix
    
    # Output covariance under intervention: Sigma_{out} = A * Sigma_{do} * A^T + Sigma
    sigma_interventional = A_arr @ sigma_do @ A_arr.T + Sigma_sym
    sigma_interventional = 0.5 * (sigma_interventional + sigma_interventional.T)
    
    # Determinism: H(X_{t+1} | do(X_t)) = 0.5 * ln((2*pi*e)^d * det(Sigma_{out}))
    logdet_interventional = _stable_logdet(sigma_interventional, reg=regularization)
    h_interventional = 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet_interventional)
    
    # Effective Information: EI = H(X_{t+1} | do(X_t)) - H(epsilon) = 0.5 * (logdet_out - logdet_noise)
    ei = max(0.0, 0.5 * (logdet_interventional - logdet_noise))
    
    # Effectiveness normalization relative to intervention theoretical maximum entropy
    max_h = interv_obj.max_entropy
    effectiveness = float(ei / max_h) if max_h > 0 else 0.0
    
    return CausalDecomposition(
        effective_information=float(ei),
        determinism=float(h_interventional),
        degeneracy=float(h_noise),
        effectiveness=float(effectiveness)
    )


def gavish_donoho_lambda_star(beta: float) -> float:
    """
    Optimal hard threshold coefficient for singular values under known noise variance
    (Gavish & Donoho, IEEE Trans. Inf. Theory 2014, eq. 11).
    """
    b = max(1e-6, min(1.0, float(beta)))
    return float(np.sqrt(2.0 * (b + 1.0) + (8.0 * b) / (b + 1.0 + np.sqrt(b**2 + 14.0 * b + 1.0))))


def compute_causal_spectrum(
    transition_matrix_A: np.ndarray,
    noise_covariance_Sigma: np.ndarray,
    regularization: float = 1e-8,
    zero_threshold: float = 1e-12,
    n_eff: Optional[float] = None,
    min_eig_cov_x: Optional[float] = None,
    denoising: Optional[str] = "gavish_donoho"
) -> CausalSpectrumReport:
    """
    Compute exact Causal Information Spectrum and Dynamic Causal Dimensionality (DCD).
    
    For linear-Gaussian dynamics X_{t+1} = A X_t + epsilon, epsilon ~ N(0, Sigma):
        lambda_i = eigvals(Sigma^{-1} A A^T) >= 0
        e_i = 0.5 * ln(1 + lambda_i) >= 0
        EI = sum_{i=1}^d e_i
        DCD^{PR} = (sum e_i)^2 / sum e_i^2
        DCD^{entropy} = exp(-sum pi_i ln pi_i) where pi_i = e_i / EI
        
    Numerically stable via Cholesky factor L of Sigma and SVD of M = L^{-1} A,
    guaranteeing real non-negative eigenvalues, Fisher causal projection vectors V,
    and exact rotational invariance.
    When n_eff is provided and denoising is enabled, applies a Gavish & Donoho (2014)-inspired
    finite-sample spectral threshold calibrated to the local empirical noise floor.
    """
    A_arr = np.asarray(transition_matrix_A, dtype=np.float64)
    Sigma_arr = np.asarray(noise_covariance_Sigma, dtype=np.float64)
    
    if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(Sigma_arr)):
        raise ValueError("Transition matrix A and noise covariance Sigma must contain finite numbers.")
        
    d = A_arr.shape[0]
    if d == 0:
        raise ValueError("Cannot compute spectrum for zero-dimensional state.")
        
    # Contract: A=0 makes intervention and output independent, EI is exactly zero
    if np.allclose(A_arr, 0.0):
        return CausalSpectrumReport(
            spectrum=np.zeros(d, dtype=np.float64),
            eigenvalues=np.zeros(d, dtype=np.float64),
            effective_information=0.0,
            dcd_pr=0.0,
            dcd_entropy=0.0,
            normalized_weights=np.zeros(d, dtype=np.float64),
            projection_v=np.eye(d, dtype=np.float64)
        )
        
    # Symmetrize and regularize Sigma
    Sigma_sym = 0.5 * (Sigma_arr + Sigma_arr.T)
    if regularization > 0:
        Sigma_sym = Sigma_sym + regularization * np.eye(d)
        
    # Cholesky factorization of Sigma
    try:
        L = np.linalg.cholesky(Sigma_sym)
        M = scipy.linalg.solve_triangular(L, A_arr, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        w_eigs, v_eigs = np.linalg.eigh(Sigma_sym)
        w_clipped = np.maximum(w_eigs, max(regularization, 1e-12))
        M = (v_eigs.T @ A_arr) / np.sqrt(w_clipped)[:, np.newaxis]
    
    # Singular values s_i and right-singular vectors V of M:
    # M = U S Vt, so V = Vt.T has columns corresponding to the eigenvectors of
    # Fisher causal operator F = A^T Sigma^{-1} A = V diag(s^2) V^T
    try:
        U, s, Vt = scipy.linalg.svd(M, full_matrices=False)
        V = Vt.T
    except np.linalg.LinAlgError:
        s = scipy.linalg.svdvals(M)
        V = np.eye(d, dtype=np.float64)
        
    lambdas = np.maximum(s ** 2, 0.0)
    
    # Finite-sample causal spectrum denoising
    if n_eff is not None and denoising is not None and n_eff > 0:
        beta = min(0.99, float(d) / float(n_eff))
        sig_x_min = max(float(min_eig_cov_x), 1e-4) if min_eig_cov_x is not None else 1.0
        
        if denoising in ("gavish_donoho", "gavish_donoho_inspired", "empirical_spectral_threshold", "hard", "optimal"):
            lam_star = gavish_donoho_lambda_star(beta)
            lambda_cut = (lam_star ** 2) * beta / sig_x_min
            lambdas = np.where(lambdas > lambda_cut, lambdas, 0.0)
        elif denoising in ("marchenko_pastur", "soft"):
            lambda_cut = ((1.0 + np.sqrt(beta)) ** 2) * beta / sig_x_min
            lambdas = np.maximum(0.0, lambdas - lambda_cut)
    
    # Causal spectrum e_i = 0.5 * ln(1 + lambda_i)
    e_spectrum = 0.5 * np.log1p(lambdas)
    total_ei = float(np.sum(e_spectrum))
    
    if total_ei <= zero_threshold:
        dcd_pr = 0.0
        dcd_entropy = 0.0
        pi = np.zeros(d, dtype=np.float64)
    else:
        sum_sq = float(np.sum(e_spectrum ** 2))
        dcd_pr = float((total_ei ** 2) / sum_sq) if sum_sq > 0 else 0.0
        
        pi = e_spectrum / total_ei
        pos_pi = pi[pi > 0.0]
        h_entropy = -float(np.sum(pos_pi * np.log(pos_pi)))
        dcd_entropy = float(np.exp(h_entropy))
        
    return CausalSpectrumReport(
        spectrum=e_spectrum,
        eigenvalues=lambdas,
        effective_information=total_ei,
        dcd_pr=dcd_pr,
        dcd_entropy=dcd_entropy,
        normalized_weights=pi,
        projection_v=V
    )



def compute_dce_raw(micro_ei: float, macro_ei: float) -> float:
    """
    Compute Primary Dynamic Causal Emergence (strict Hoel definition):
        DCE_t(q) = EI_t(V^{(q)}) - EI_t(X)
        
    Emergence occurs if and only if DCE_t(q) > 0.
    """
    return float(macro_ei - micro_ei)


def compute_dce_density(micro_ei: float, macro_ei: float, p_dim: int, q_dim: int) -> float:
    """
    Compute Secondary Dynamic Causal Efficiency Gain / Information Density Gain:
        DCE_t^{density}(q) = EI_t(V^{(q)}) / q - EI_t(X) / p
    """
    if p_dim <= 0 or q_dim <= 0:
        raise ValueError(f"Dimensions must be positive, got p={p_dim}, q={q_dim}")
    return float((macro_ei / float(q_dim)) - (micro_ei / float(p_dim)))


def compute_dce(
    micro_ei: float,
    macro_ei: float,
    p_dim: int = 1,
    q_dim: int = 1,
    normalized: bool = False
) -> float:
    """
    Compute Dynamic Causal Emergence metrics:
        Raw (Primary): DCE = EI(V^{(q)}) - EI(X^{(p)})
        Density (Secondary): DCE^{density} = EI(V^{(q)}) / q - EI(X^{(p)}) / p
    """
    if normalized:
        return compute_dce_density(micro_ei, macro_ei, p_dim, q_dim)
    return compute_dce_raw(micro_ei, macro_ei)


def estimate_local_affine_dynamics(
    x_past: np.ndarray,
    x_future: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit local weighted affine transition via Weighted Ridge Regression with explicit intercept:
        x_{s+1} = A_t * x_s + c_t + epsilon_s,  weighted by w_{t,s}
        
    Returns:
        A_matrix (d, d), intercept_c (d,), Sigma_matrix (d, d)
    """
    d = x_past.shape[1]
    
    # Clean weights: ensure positive and normalized
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    sum_w = np.sum(w)
    if sum_w <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / sum_w
        
    w_sqrt = np.sqrt(w)[:, np.newaxis]
    
    # Weighted centers to capture affine drift / intercept
    x_bar = np.sum(x_past * w[:, np.newaxis], axis=0, keepdims=True)
    y_bar = np.sum(x_future * w[:, np.newaxis], axis=0, keepdims=True)
    
    x_centered = x_past - x_bar
    y_centered = x_future - y_bar
    
    x_past_w = x_centered * w_sqrt
    y_future_w = y_centered * w_sqrt
    
    adaptive_ridge = ridge_alpha
    
    # XtX = X_past^T W X_past + ridge * I_d
    xtx = x_past_w.T @ x_past_w + adaptive_ridge * np.eye(d)
    # XtY = X_past^T W X_future
    xty = x_past_w.T @ y_future_w
    
    # Solve for transition matrix A: A_t = (XtX)^(-1) XtY -> A_t^T = solve(XtX, xty)
    try:
        a_matrix_t = np.linalg.solve(xtx, xty).T
    except np.linalg.LinAlgError:
        a_matrix_t = np.linalg.lstsq(xtx, xty, rcond=1e-5)[0].T
        
    intercept_c = (y_bar - x_bar @ a_matrix_t.T).ravel()
        
    # Weighted residuals with explicit intercept
    pred_future = x_past @ a_matrix_t.T + intercept_c
    residuals = x_future - pred_future
    residuals_w = residuals * w_sqrt
    
    sigma_t = residuals_w.T @ residuals_w
    sigma_t = 0.5 * (sigma_t + sigma_t.T) + max(adaptive_ridge, 1e-12) * np.eye(d)
    
    return a_matrix_t, intercept_c, sigma_t


def estimate_local_gaussian_dynamics(
    x_past: np.ndarray,
    x_future: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit local weighted linear transition via Weighted Ridge Regression:
        x_{s+1} = A_t * x_s + epsilon_s,  weighted by w_{t,s}
        
    Returns:
        A_matrix (d, d) [clean standard ndarray], Sigma_matrix (d, d)
    """
    a_mat, _, sigma_mat = estimate_local_affine_dynamics(x_past, x_future, weights, ridge_alpha)
    return a_mat, sigma_mat

