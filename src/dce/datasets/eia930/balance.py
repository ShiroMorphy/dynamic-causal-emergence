"""
Physical Balance Validation and Quality Assurance for EIA-930 Data.

Per Section 13.5 of Master Plan:
Does not treat G - I - D = 0 as an exact nodal Kirchhoff law, but as an operational
accounting diagnostic:
    Residual_{i, t} = Generation_{i, t} - Interchange_{i, t} - Demand_{i, t}
Accounts for transmission losses, reporting discrepancies, pumping storage, and timing lags.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def compute_balance_residuals(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute operational balance residuals:
        residual = generation - interchange - demand
        relative_residual = |residual| / (demand + 1e-6)
    """
    df_res = df.copy()
    # In EIA-930: Interchange > 0 represents net export.
    # Therefore, Generation = Demand + Interchange, or Demand_implied = Generation - Interchange.
    df_res["accounting_residual"] = df_res["generation"] - df_res["interchange"] - df_res["demand"]
    df_res["relative_residual"] = np.abs(df_res["accounting_residual"]) / (np.maximum(df_res["demand"], 1.0))
    return df_res


def get_balance_summary_statistics(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute summary statistics of accounting residuals per Balancing Authority and Interconnection.
    """
    df_res = compute_balance_residuals(df)
    grouped = df_res.groupby(["interconnection", "ba_code"])["relative_residual"]
    summary = grouped.agg(
        mean="mean",
        std="std",
        median="median",
        p95=lambda x: np.percentile(x, 95),
        p99=lambda x: np.percentile(x, 99),
        max="max"
    ).reset_index()
    return summary


def validate_physical_balance(
    df: pd.DataFrame,
    tolerance_ratio: float = 0.20,
    reconcile_anomalies: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and clean physical power balance per Balancing Authority.
    
    Args:
        df: DataFrame with demand, generation, interchange.
        tolerance_ratio: Maximum allowable relative residual before flagging.
        reconcile_anomalies: Whether to adjust interchange for extreme outliers (default False).
        
    Returns:
        (df_cleaned, anomaly_report)
    """
    df_clean = compute_balance_residuals(df)
    
    anomalies = df_clean["relative_residual"] > tolerance_ratio
    anomaly_report = df_clean[anomalies].copy()
    
    if reconcile_anomalies and anomalies.any():
        # Reconcile interchange (net export = generation - demand)
        df_clean.loc[anomalies, "interchange"] = (
            df_clean.loc[anomalies, "generation"] - df_clean.loc[anomalies, "demand"]
        )
        df_clean.loc[anomalies, "accounting_residual"] = 0.0
        df_clean.loc[anomalies, "relative_residual"] = 0.0
        
    return df_clean, anomaly_report

