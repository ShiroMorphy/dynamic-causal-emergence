"""
Unit test for EIA-930 balance calculation and sign convention.
Interchange > 0 indicates net export.
Generation = Demand + Interchange  =>  Generation - Interchange - Demand = 0.
"""

import numpy as np
import pandas as pd
import pytest

from dce.datasets.eia930.balance import compute_balance_residuals, validate_physical_balance


def test_balance_sign_convention():
    # Case: Demand = 100 MW, Generation = 120 MW, Net Export (Interchange) = 20 MW.
    # Generation - Interchange - Demand = 120 - 20 - 100 = 0.
    df = pd.DataFrame({
        "demand": [100.0],
        "generation": [120.0],
        "interchange": [20.0],
        "interconnection": ["Eastern"],
        "ba_code": ["PJM"]
    })
    
    res = compute_balance_residuals(df)
    assert np.isclose(res["accounting_residual"].iloc[0], 0.0), (
        f"Expected 0.0, got {res['accounting_residual'].iloc[0]}"
    )
    assert np.isclose(res["relative_residual"].iloc[0], 0.0)


def test_balance_reconcile_default_false():
    # Reconcile anomalies should default to False to avoid unproven data mutation
    df = pd.DataFrame({
        "demand": [100.0],
        "generation": [120.0],
        "interchange": [50.0], # Mismatch: 120 - 50 - 100 = -30 != 0
        "interconnection": ["Eastern"],
        "ba_code": ["PJM"]
    })
    
    df_clean, anomalies = validate_physical_balance(df, tolerance_ratio=0.10)
    # df_clean interchange should NOT be modified when reconcile_anomalies is False
    assert df_clean["interchange"].iloc[0] == 50.0
    assert len(anomalies) == 1
