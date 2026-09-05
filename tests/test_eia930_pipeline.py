import numpy as np
import pytest
from dce.datasets.eia930.client import generate_synthetic_eia930_benchmark
from dce.datasets.eia930.balance import validate_physical_balance
from dce.datasets.eia930.microstate import build_power_grid_microstate


def test_eia930_synthetic_generation():
    df = generate_synthetic_eia930_benchmark(n_hours=100)
    assert len(df) > 0
    assert "demand" in df.columns
    assert "generation" in df.columns
    assert "wind" in df.columns
    assert "solar" in df.columns


def test_eia930_balance_validation():
    df = generate_synthetic_eia930_benchmark(n_hours=50)
    df_clean, report = validate_physical_balance(df)
    assert len(df_clean) == len(df)


def test_eia930_microstate_construction():
    df = generate_synthetic_eia930_benchmark(n_hours=120)
    df_clean, _ = validate_physical_balance(df)
    grid_data = build_power_grid_microstate(df_clean)
    assert grid_data.microstate_matrix.shape[0] == 120
    assert len(grid_data.vre_penetration) == 120
    assert "micro" in grid_data.multiscale_representations
    assert "grid" in grid_data.multiscale_representations


def test_real_eia930_interconnection_isolation():
    from dce.datasets.eia930.client import load_real_eia930_archive
    df = load_real_eia930_archive("2021")
    assert len(df) > 50000
    
    # Test Eastern Interconnection
    eastern = build_power_grid_microstate(df, interconnection="Eastern", spec="core5", scaling="causal_rolling")
    assert eastern.microstate_matrix.shape[1] == 55  # 11 BAs * 5 variables
    assert np.all(np.isfinite(eastern.microstate_matrix))
    
    # Test ERCOT Interconnection
    ercot = build_power_grid_microstate(df, interconnection="ERCOT", spec="fuel_extended")
    assert ercot.microstate_matrix.shape[1] == 9   # 9 fuel features
    assert np.all(np.isfinite(ercot.microstate_matrix))
    assert len(ercot.net_load) == ercot.microstate_matrix.shape[0]

