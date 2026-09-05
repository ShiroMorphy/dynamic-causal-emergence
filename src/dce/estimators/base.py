"""
Base Abstract Class for Dynamic Causal Emergence Estimators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseDynamicCE(ABC):
    """
    Abstract Base Class for Dynamic Causal Emergence estimators.
    
    Adheres to Scikit-Learn estimator conventions:
    - fit(X)
    - transform(X)
    - fit_transform(X)
    """
    
    def __init__(
        self,
        macro_dims: Optional[List[int]] = None,
        bandwidth: float = 24.0,
        kernel_type: str = "gaussian",
        causal_only: bool = False
    ) -> None:
        self.macro_dims = macro_dims or [1, 2, 4]
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        self.causal_only = causal_only
        
        # Output attributes populated after fit
        self.emergence_: Optional[np.ndarray] = None          # (T,) DCE_t
        self.causal_dimension_: Optional[np.ndarray] = None   # (T,) q_t^*
        self.micro_ei_: Optional[np.ndarray] = None           # (T,) EI_t^{(p)}
        self.macro_ei_: Optional[Dict[int, np.ndarray]] = {}  # {q: (T,) EI_t^{(q)}}
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> "BaseDynamicCE":
        """Fit the dynamic causal emergence model to data X."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the optimal time-varying macroscopic representation V_t."""
        if not self.is_fitted_:
            raise RuntimeError("Estimator must be fitted before calling transform().")
        return self._transform_impl(X)

    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """Fit model and return macroscopic representation."""
        return self.fit(X, y).transform(X)

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        """Default transform implementation."""
        raise NotImplementedError("Transform not implemented for this estimator.")
