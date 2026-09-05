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


def _stable_logdet(matrix: np.ndarray, reg: float = 1e-8) -> float:
    """
    Compute log determinant using Cholesky with fallback to slogdet.
    Guarantees symmetry and positive-definiteness.
    """
    d = matrix.shape[0]
    sym = 0.5 * (matrix + matrix.T)
    min_eig = np.min(np.linalg.eigvalsh(sym)) if d <= 64 else 0.0
    if min_eig < reg:
        sym += (reg - min_eig + 1e-8) * np.eye(d)
        
    try:
        # Cholesky is faster and numerically stabler
        L = np.linalg.cholesky(sym)
        return float(2.0 * np.sum(np.log(np.diag(L))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(sym)
        if sign <= 0:
            # Fallback with higher regularization
            sym += 1e-4 * np.eye(d)
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
        
    Args:
        transition_matrix_A: (d, d) local transition matrix A
        noise_covariance_Sigma: (d, d) transition noise covariance Sigma
        intervention: BaseIntervention instance or name ('gaussian', 'uniform', 'whitened').
                      If domain_bound is passed for backward compatibility, uses uniform with bound.
        domain_bound: (Deprecated) Hypercube half-width for uniform intervention.
        regularization: Numerical ridge for covariance conditioning.
        
    Returns:
        CausalDecomposition with effective_information, determinism, degeneracy, effectiveness.
    """
    A_arr = np.asarray(transition_matrix_A, dtype=np.float64)
    Sigma_arr = np.asarray(noise_covariance_Sigma, dtype=np.float64)
    
    d = A_arr.shape[0]
    if d == 0:
        raise ValueError("Cannot compute EI for zero-dimensional state.")
        
    # Clean non-finite elements if any
    A_clean = np.nan_to_num(A_arr, nan=0.0, posinf=1.0, neginf=-1.0)
    Sigma_clean = np.nan_to_num(Sigma_arr, nan=1e-4, posinf=1.0, neginf=1e-4)
    Sigma_sym = 0.5 * (Sigma_clean + Sigma_clean.T) + regularization * np.eye(d)
    
    # Resolve intervention
    if intervention is None:
        if domain_bound is not None:
            interv_obj = UniformCompactIntervention(dim=d, bound=domain_bound)
        else:
            interv_obj = GaussianMaxEntropyIntervention(dim=d, sigma_do=1.0)
    elif isinstance(intervention, str):
        if intervention == "uniform" and domain_bound is not None:
            interv_obj = UniformCompactIntervention(dim=d, bound=domain_bound)
        else:
            interv_obj = get_intervention(intervention, dim=d)
    else:
        interv_obj = intervention
        
    sigma_do = interv_obj.covariance_matrix
    
    # Output covariance under intervention: Sigma_{out} = A * Sigma_{do} * A^T + Sigma
    sigma_interventional = A_clean @ sigma_do @ A_clean.T + Sigma_sym
    sigma_interventional = 0.5 * (sigma_interventional + sigma_interventional.T) + regularization * np.eye(d)
    
    # Determinism: H(X_{t+1} | do(X_t)) = 0.5 * ln((2*pi*e)^d * det(Sigma_{out}))
    logdet_interventional = _stable_logdet(sigma_interventional, reg=regularization)
    h_interventional = 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet_interventional)
    
    # Degeneracy: H(epsilon) = 0.5 * ln((2*pi*e)^d * det(Sigma))
    logdet_noise = _stable_logdet(Sigma_sym, reg=regularization)
    h_noise = 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet_noise)
    
    # Effective Information: EI = H(X_{t+1} | do(X_t)) - H(epsilon) = 0.5 * (logdet_out - logdet_noise)
    ei = 0.5 * (logdet_interventional - logdet_noise)
    
    # Effectiveness normalization relative to intervention theoretical maximum entropy
    max_h = interv_obj.max_entropy
    effectiveness = float(ei / max_h) if max_h > 0 else 0.0
    
    return CausalDecomposition(
        effective_information=float(ei),
        determinism=float(h_interventional),
        degeneracy=float(h_noise),
        effectiveness=float(effectiveness)
    )


def compute_dce(
    micro_ei: float,
    macro_ei: float,
    p_dim: int,
    q_dim: int,
    normalized: bool = False
) -> float:
    """
    Compute Dynamic Causal Emergence:
        Raw: DCE = EI(V^{(q)}) - EI(X^{(p)})
        Normalized: DCE^{norm} = EI(V^{(q)}) / q - EI(X^{(p)}) / p
    """
    if normalized:
        return float((macro_ei / float(q_dim)) - (micro_ei / float(p_dim)))
    return float(macro_ei - micro_ei)


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
        A_matrix (d, d), Sigma_matrix (d, d)
    """
    d = x_past.shape[1]
    
    # Clean weights: ensure positive and normalized
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    sum_w = np.sum(w)
    if sum_w <= 1e-12:
        # Uniform fallback if weights collapsed
        w = np.ones_like(w) / len(w)
        sum_w = 1.0
    else:
        w = w / sum_w
        
    w_sqrt = np.sqrt(w)[:, np.newaxis]
    
    x_past_w = x_past * w_sqrt
    x_future_w = x_future * w_sqrt
    
    # XtX = X_past^T W X_past + ridge * I_d
    xtx = x_past_w.T @ x_past_w + ridge_alpha * np.eye(d)
    # XtY = X_past^T W X_future
    xty = x_past_w.T @ x_future_w
    
    # Solve for transition matrix A: A_t = (XtX)^(-1) XtY -> A_t^T = solve(XtX, xty)
    try:
        a_matrix_t = np.linalg.solve(xtx, xty).T
    except np.linalg.LinAlgError:
        a_matrix_t = np.linalg.lstsq(xtx, xty, rcond=1e-5)[0].T
        
    # Weighted residuals
    pred_future = x_past @ a_matrix_t.T
    residuals = x_future - pred_future
    residuals_w = residuals * w_sqrt
    
    sigma_t = residuals_w.T @ residuals_w
    sigma_t = 0.5 * (sigma_t + sigma_t.T) + max(ridge_alpha, 1e-6) * np.eye(d)
    
    return a_matrix_t, sigma_t
