import numpy as np
import pytest
from dce.estimators.local_kernel import LocalKernelDCE
from dce.estimators.linear_svd import LinearSVDDCE
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
from dce.estimators.dyn_nis import DynamicNIS
from dce.estimators.multiscale_ce2 import MultiscaleCE2Apportioner


def test_local_linear_gaussian_dce():
    np.random.seed(42)
    X = np.random.randn(80, 4)
    model = LocalLinearGaussianDCE(macro_dims=[1, 2], bandwidth=10.0)
    model.fit(X)
    assert model.emergence_ is not None
    assert len(model.emergence_) == 79
    assert len(model.causal_dimension_) == 79
    assert len(model.causal_dimension_confset_) == 79


def test_local_kernel_dce():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    model = LocalKernelDCE(macro_dims=[1, 2], bandwidth=10.0)
    model.fit(X)
    assert model.emergence_ is not None
    assert len(model.emergence_) == 99
    assert len(model.causal_dimension_) == 99


def test_linear_svd_dce():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    model = LinearSVDDCE(macro_dims=[1, 2], bandwidth=10.0)
    model.fit(X)
    assert model.emergence_ is not None
    assert len(model.emergence_) == 99


def test_dyn_nis_estimator():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    model = DynamicNIS(macro_dims=[1, 2], bandwidth=10.0, steps_per_window=2, hidden_dim=16)
    model.fit(X)
    assert model.emergence_ is not None
    assert len(model.emergence_) == 49
    assert len(model.causal_dimension_) == 49
    assert len(model.causal_dimension_confset_) == 49


def test_static_nis_plus_windowed():
    from dce.estimators.static_nis_plus import StaticNISPlusWindowed
    np.random.seed(42)
    X = np.random.randn(40, 4)
    model = StaticNISPlusWindowed(macro_dims=[1, 2], bandwidth=10.0, steps_per_window=2, hidden_dim=16)
    model.fit(X)
    assert model.emergence_ is not None
    assert len(model.emergence_) == 39
    assert len(model.causal_dimension_) == 39



def test_multiscale_ce2_apportioner():
    np.random.seed(42)
    multiscale_data = {
        "micro": np.random.randn(60, 8),
        "rto": np.random.randn(60, 4),
        "grid": np.random.randn(60, 2)
    }
    apportioner = MultiscaleCE2Apportioner(hierarchy_levels=["micro", "rto", "grid"], bandwidth=10.0)
    apportioner.fit(multiscale_data)
    assert apportioner.total_causality_ is not None
    assert len(apportioner.total_causality_) == 59
    assert "micro" in apportioner.apportioned_causality_
