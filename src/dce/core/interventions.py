"""
Intervention Distributions and Pearlian do-calculus for Continuous Effective Information.

Formalizes the intervention measure nu_t^{(d)} over continuous manifolds:
1. GaussianMaxEntropyIntervention: do(Z ~ N(0, sigma^2 * I_d))
2. UniformCompactIntervention: do(Z ~ U([-a, a]^d))
3. EmpiricalWhitenedIntervention: Data-driven whitened reference
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class BaseIntervention(ABC):
    """Abstract base class for interventional distributions nu^{(d)}."""
    
    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"Dimension must be strictly positive, got {dim}.")
        self.dim = int(dim)

    @property
    @abstractmethod
    def covariance_matrix(self) -> np.ndarray:
        """Interventional covariance matrix Sigma_{do} of shape (dim, dim)."""
        pass

    @property
    @abstractmethod
    def max_entropy(self) -> float:
        """Theoretical differential entropy H(nu) of the intervention measure."""
        pass

    @abstractmethod
    def sample(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Draw n_samples from the intervention distribution nu."""
        pass


class GaussianMaxEntropyIntervention(BaseIntervention):
    """
    Maximum-entropy intervention under second-moment constraint:
        do(Z ~ N(0, sigma_do^2 * I_d))
        
    Sigma_{do} = sigma_do^2 * I_d
    H(nu) = 0.5 * d * ln(2 * pi * e * sigma_do^2)
    """
    def __init__(self, dim: int, sigma_do: float = 1.0) -> None:
        super().__init__(dim)
        if sigma_do <= 0:
            raise ValueError(f"sigma_do must be positive, got {sigma_do}.")
        self.sigma_do = float(sigma_do)
        self._cov = (self.sigma_do ** 2) * np.eye(self.dim, dtype=np.float64)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def max_entropy(self) -> float:
        return float(0.5 * self.dim * np.log(2.0 * np.pi * np.e * (self.sigma_do ** 2)))

    def sample(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        r = rng or np.random.RandomState()
        return r.randn(n_samples, self.dim) * self.sigma_do


class UniformCompactIntervention(BaseIntervention):
    """
    Uniform intervention over compact hypercube:
        do(Z ~ U([-a, a]^d))
        
    Var(Z_i) = a^2 / 3
    Sigma_{do} = (a^2 / 3) * I_d
    H(nu) = d * ln(2 * a)
    """
    def __init__(self, dim: int, bound: float = 1.0) -> None:
        super().__init__(dim)
        if bound <= 0:
            raise ValueError(f"bound 'a' must be positive, got {bound}.")
        self.bound = float(bound)
        var = (self.bound ** 2) / 3.0
        self._cov = var * np.eye(self.dim, dtype=np.float64)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def max_entropy(self) -> float:
        return float(self.dim * np.log(2.0 * self.bound))

    def sample(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        r = rng or np.random.RandomState()
        return r.uniform(-self.bound, self.bound, size=(n_samples, self.dim))


class EmpiricalWhitenedIntervention(BaseIntervention):
    """
    Empirical whitened intervention matching unit variance:
    Sigma_{do} = I_d
    """
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._cov = np.eye(self.dim, dtype=np.float64)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def max_entropy(self) -> float:
        return float(0.5 * self.dim * np.log(2.0 * np.pi * np.e))

    def sample(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        r = rng or np.random.RandomState()
        return r.randn(n_samples, self.dim)


INTERVENTION_REGISTRY = {
    "gaussian": GaussianMaxEntropyIntervention,
    "gaussian_identity": GaussianMaxEntropyIntervention,
    "uniform": UniformCompactIntervention,
    "whitened": EmpiricalWhitenedIntervention,
}


def get_intervention(
    name: Union[str, BaseIntervention],
    dim: int,
    **kwargs
) -> BaseIntervention:
    """Factory function for intervention distributions."""
    if isinstance(name, BaseIntervention):
        if name.dim != dim:
            raise ValueError(f"Intervention dimension {name.dim} does not match expected {dim}.")
        return name
        
    key = str(name).lower()
    if key not in INTERVENTION_REGISTRY:
        raise ValueError(f"Unknown intervention type '{name}'. Options: {list(INTERVENTION_REGISTRY.keys())}")
        
    cls = INTERVENTION_REGISTRY[key]
    return cls(dim=dim, **kwargs)
