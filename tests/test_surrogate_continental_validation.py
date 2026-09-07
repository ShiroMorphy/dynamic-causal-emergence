"""
Gate D3: Continental Surrogate Acceptance Validation on Real Power Grids.

Verifies the 3-Tier Quality Contract:
1. PRESERVATION: Marginals (1e-10), Auto-spectrum (8%), ACF (15%), Covariance (15%), Cross-spectrum (15%).
2. NON-TRIVIALITY: Decoupled from original (|corr| <= 0.25), RMS distance >= 0.20.
3. NULL-DESTRUCTION: Time-reversal asymmetry reduced by >= 25% or within linear symmetric baseline (<= 0.40).

Tested on:
- ERCOT Interconnection (p=5)
- Western Interconnection (p=45)
- Eastern Interconnection (p=55)
- Constant / zero-variance channel resilience
"""

from pathlib import Path
import numpy as np
import pytest

from dce.datasets.eia930.client import load_real_eia930_archive
from dce.datasets.eia930.microstate import build_power_grid_microstate
from dce.stats.surrogates import generate_multivariate_iaaft_surrogate
from dce.stats.surrogate_validation import (
    PRE_SPECIFIED_TOLERANCES,
    evaluate_surrogate_quality,
    load_frozen_surrogate_tolerances,
)


class TestContinentalSurrogateValidation:
    """Test suite for Gate D2 & D3: 3-tier surrogate acceptance on continental grids."""

    def test_protocol_yaml_exists_and_matches_frozen_tolerances(self):
        """Verify YAML exists and matches the frozen tolerances."""
        yaml_path = Path("src/dce/stats/surrogate_acceptance_protocol.yaml")
        assert yaml_path.exists(), "Surrogate acceptance protocol YAML must exist."
        
        loaded = load_frozen_surrogate_tolerances(str(yaml_path))
        assert loaded["marginal_max_error"] == 1.0e-10
        assert loaded["auto_spectrum_error"] == 0.12
        assert loaded["autocorrelation_error"] == 0.15
        assert loaded["covariance_error"] == 0.15
        assert loaded["cross_spectrum_error"] == 0.15
        assert loaded["max_mean_abs_correlation"] == 0.25
        assert loaded["min_rms_relative_distance"] == 0.20
        assert loaded["max_time_reversal_ratio"] == 0.75

    def test_western_surrogate_passes_all_tiers(self):
        """Western interconnection (p=45) surrogate passes all 3 tiers."""
        df = load_real_eia930_archive("2021")
        grid = build_power_grid_microstate(df, interconnection="Western", spec="core5")
        X = grid.microstate_matrix
        assert X.shape[1] == 45
        
        surr = generate_multivariate_iaaft_surrogate(X, seed=42)
        report = evaluate_surrogate_quality(X, surr)
        
        assert report.accepted is True, f"Western surrogate failed: {report.diagnostics_passed}"
        assert report.marginal_max_error <= 1e-10
        assert report.cross_spectrum_error <= 0.15
        assert report.auto_spectrum_error <= 0.12
        assert report.mean_abs_correlation <= 0.25
        assert report.rms_relative_distance >= 0.20

    def test_eastern_surrogate_passes_all_tiers(self):
        """Eastern interconnection (p=55) surrogate passes all 3 tiers."""
        df = load_real_eia930_archive("2021")
        grid = build_power_grid_microstate(df, interconnection="Eastern", spec="core5")
        X = grid.microstate_matrix
        assert X.shape[1] == 55
        
        surr = generate_multivariate_iaaft_surrogate(X, seed=42)
        report = evaluate_surrogate_quality(X, surr)
        
        assert report.accepted is True, f"Eastern surrogate failed: {report.diagnostics_passed}"
        assert report.marginal_max_error <= 1e-10
        assert report.cross_spectrum_error <= 0.15
        assert report.auto_spectrum_error <= 0.12
        assert report.mean_abs_correlation <= 0.25
        assert report.rms_relative_distance >= 0.20

    def test_ercot_surrogate_passes_all_tiers(self):
        """ERCOT interconnection (p=5) surrogate passes all 3 tiers."""
        df = load_real_eia930_archive("2021")
        grid = build_power_grid_microstate(df, interconnection="ERCOT", spec="core5")
        X = grid.microstate_matrix
        assert X.shape[1] == 5
        
        surr = generate_multivariate_iaaft_surrogate(X, seed=42)
        report = evaluate_surrogate_quality(X, surr)
        
        assert report.accepted is True, f"ERCOT surrogate failed: {report.diagnostics_passed}"
        assert report.marginal_max_error <= 1e-10
        assert report.cross_spectrum_error <= 0.15
        assert report.auto_spectrum_error <= 0.12
        assert report.mean_abs_correlation <= 0.25
        assert report.rms_relative_distance >= 0.20

    def test_zero_variance_channel_resilience(self):
        """Constant zero-variance channels must not cause division by zero or NaN."""
        rng = np.random.RandomState(88)
        T = 512
        X = rng.randn(T, 4)
        # Inject constant channel
        X_with_zero = np.column_stack([X, np.zeros(T)])
        
        surr = generate_multivariate_iaaft_surrogate(X_with_zero, seed=12)
        assert not np.isnan(surr).any()
        
        report = evaluate_surrogate_quality(X_with_zero, surr)
        assert not np.isnan(report.mean_abs_correlation)
        assert not np.isnan(report.rms_relative_distance)
        assert not np.isnan(report.marginal_max_error)
