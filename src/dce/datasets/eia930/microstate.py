"""
Construction of Multivariate Microstate Matrix X_t and Multiscale Hierarchies.

Constructs:
1. Microstate matrix X_t: [D_1, G_1, W_1, S_1, I_1, ..., D_N, G_N, W_N, S_N, I_N] in R^{T x p}
   with support for isolated interconnections (Eastern, Western, ERCOT per Section 16).
2. Variable Renewable Energy Penetration: VRE_t = (Wind_t + Solar_t) / Generation_t
3. Net Load: NetLoad_t = Demand_t - (Wind_t + Solar_t)
4. Demand Forecast Error: FE_t = |Demand_t - Forecast_t| / Demand_t
5. Causal rolling scaling (Section 13.8) to prevent look-ahead bias in online analysis.
6. Nested Hierarchical Aggregations (BA -> RTO -> Interconnection -> Grid).
"""

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd


class GridMicrostateData(NamedTuple):
    """Container for processed grid dataset."""
    timestamps: pd.DatetimeIndex
    microstate_matrix: np.ndarray        # (T, p) Microstate X_t
    feature_names: List[str]             # Names of features in X_t
    vre_penetration: np.ndarray          # (T,) VRE_t
    net_load: np.ndarray                 # (T,) Net Load in MW
    forecast_error: np.ndarray           # (T,) FE_t
    multiscale_representations: Dict[str, np.ndarray]  # {'micro', 'rto', 'interconnection', 'grid'}
    ba_list: List[str]
    interconnection: str


def apply_causal_rolling_scaling(
    matrix: np.ndarray,
    window: int = 168,
    min_periods: int = 24,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Apply strictly causal rolling z-score scaling using past observations only.
    
    X_scaled[t] = (X[t] - mean(X[t-window:t])) / (std(X[t-window:t]) + eps)
    No future leakage.
    """
    T, p = matrix.shape
    scaled = np.zeros_like(matrix)
    
    for t in range(T):
        if t < min_periods:
            # For cold start, scale by available history up to t (inclusive)
            hist = matrix[:max(t+1, 1), :]
        else:
            start_idx = max(0, t - window)
            hist = matrix[start_idx:t, :]
            
        mu = np.mean(hist, axis=0)
        sigma = np.std(hist, axis=0)
        scaled[t, :] = (matrix[t, :] - mu) / (sigma + eps)
        
    return scaled


def build_power_grid_microstate(
    df: pd.DataFrame,
    interconnection: Optional[str] = None,
    spec: str = "core5",
    scaling: str = "none",
    rolling_window: int = 168
) -> GridMicrostateData:
    """
    Build structured microstate matrix and hierarchical representations from EIA-930 dataframe.
    
    Args:
        df: Processed EIA-930 hourly DataFrame.
        interconnection: Filter to specific interconnection ('Eastern', 'Western', 'ERCOT') or None for all.
        spec: Feature specification: 'core5' (demand, generation, wind, solar, interchange) or
              'fuel_extended' (adds coal, gas, nuclear, hydro).
        scaling: Scaling method: 'none', 'causal_rolling', or 'standard' (retrospective-only).
        rolling_window: Window in hours for causal rolling scaling.
    """
    df_work = df.copy()
    if interconnection is not None and interconnection.lower() not in ["all", "none"]:
        df_work = df_work[df_work["interconnection"].str.lower() == interconnection.lower()].copy()
        if len(df_work) == 0:
            raise ValueError(f"No records found for interconnection '{interconnection}'")
            
    df_sorted = df_work.sort_values(by=["timestamp", "ba_code"]).reset_index(drop=True)
    timestamps = pd.DatetimeIndex(sorted(df_sorted["timestamp"].unique()))
    ba_codes = sorted(df_sorted["ba_code"].unique())
    
    n_timestamps = len(timestamps)
    n_bas = len(ba_codes)
    
    if spec == "fuel_extended":
        variables = ["demand", "generation", "wind", "solar", "coal", "gas", "nuclear", "hydro", "interchange"]
    else:
        variables = ["demand", "generation", "wind", "solar", "interchange"]
        
    p_dim = n_bas * len(variables)
    micro_mat = np.zeros((n_timestamps, p_dim), dtype=np.float64)
    feature_names = []
    
    is_causal = (scaling == "causal_rolling")
    for idx, var in enumerate(variables):
        if var in df_sorted.columns:
            pivoted = df_sorted.pivot(index="timestamp", columns="ba_code", values=var).reindex(timestamps)
            if is_causal:
                pivoted = pivoted.ffill().fillna(0.0)
            else:
                pivoted = pivoted.ffill().bfill().fillna(0.0)
            values = pivoted.values
        else:
            values = np.zeros((n_timestamps, n_bas), dtype=np.float64)
            
        start_col = idx * n_bas
        end_col = (idx + 1) * n_bas
        micro_mat[:, start_col:end_col] = values
        for ba in ba_codes:
            feature_names.append(f"{ba}_{var}")
            
    # Compute aggregate metrics for this panel
    total_wind = df_sorted.groupby("timestamp")["wind"].sum().reindex(timestamps).fillna(0.0).values
    total_solar = df_sorted.groupby("timestamp")["solar"].sum().reindex(timestamps).fillna(0.0).values
    total_gen = df_sorted.groupby("timestamp")["generation"].sum().reindex(timestamps).fillna(0.0).values
    total_demand = df_sorted.groupby("timestamp")["demand"].sum().reindex(timestamps).fillna(0.0).values
    
    if "demand_forecast" in df_sorted.columns:
        total_forecast = df_sorted.groupby("timestamp")["demand_forecast"].sum().reindex(timestamps).fillna(0.0).values
    else:
        total_forecast = total_demand
        
    vre_penetration = (total_wind + total_solar) / np.maximum(total_gen, 1e-3)
    net_load = total_demand - (total_wind + total_solar)
    forecast_error = np.abs(total_demand - total_forecast) / np.maximum(total_demand, 1e-3)
    
    # Optional scaling
    if scaling == "causal_rolling":
        micro_mat = apply_causal_rolling_scaling(micro_mat, window=rolling_window)
    elif scaling == "standard":
        mu = np.mean(micro_mat, axis=0)
        sigma = np.std(micro_mat, axis=0) + 1e-6
        micro_mat = (micro_mat - mu) / sigma
        
    # Multiscale hierarchical representations
    # 1. Micro
    # 2. RTO (if rto column exists)
    if "rto" in df_sorted.columns and df_sorted["rto"].nunique() > 1:
        rto_df = df_sorted.groupby(["timestamp", "rto"])[variables].sum().reset_index()
        rto_cols = []
        for var in variables:
            if var in rto_df.columns:
                piv = rto_df.pivot(index="timestamp", columns="rto", values=var).reindex(timestamps)
                piv = piv.ffill().fillna(0.0) if is_causal else piv.ffill().bfill().fillna(0.0)
                rto_cols.append(piv.values)
        rto_mat = np.column_stack(rto_cols)
    else:
        rto_mat = micro_mat
        
    # 3. Interconnection
    if "interconnection" in df_sorted.columns and df_sorted["interconnection"].nunique() > 1:
        inter_df = df_sorted.groupby(["timestamp", "interconnection"])[variables].sum().reset_index()
        inter_cols = []
        for var in variables:
            if var in inter_df.columns:
                piv = inter_df.pivot(index="timestamp", columns="interconnection", values=var).reindex(timestamps)
                piv = piv.ffill().fillna(0.0) if is_causal else piv.ffill().bfill().fillna(0.0)
                inter_cols.append(piv.values)
        inter_mat = np.column_stack(inter_cols)
    else:
        inter_mat = micro_mat
        
    # 4. Grid
    grid_mat = np.column_stack([
        df_sorted.groupby("timestamp")[var].sum().reindex(timestamps).fillna(0.0).values
        for var in variables if var in df_sorted.columns
    ])
    
    multiscale_dict = {
        "micro": micro_mat,
        "rto": rto_mat,
        "interconnection": inter_mat,
        "grid": grid_mat
    }
    
    return GridMicrostateData(
        timestamps=timestamps,
        microstate_matrix=micro_mat,
        feature_names=feature_names,
        vre_penetration=vre_penetration,
        net_load=net_load,
        forecast_error=forecast_error,
        multiscale_representations=multiscale_dict,
        ba_list=ba_codes,
        interconnection=interconnection or "all"
    )

