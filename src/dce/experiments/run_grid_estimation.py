"""
Main Empirical Pipeline for Dynamic Causal Emergence on Real US Power Grids.

Executes Milestone M6:
- Processes real EIA-930 hourly data for Eastern, Western, and ERCOT.
- Fits LocalLinearGaussianDCE in both retrospective (symmetric) and causal (one-sided) modes.
- Evaluates raw emergence, normalized emergence, optimal causal dimension q_t^*,
  and confidence sets C_t^{(q)}.
- Exports reproducible parquets to results/empirical/.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd

from dce.datasets.eia930.client import load_real_eia930_archive
from dce.datasets.eia930.microstate import build_power_grid_microstate
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE


def run_estimation_for_panel(
    df_raw: pd.DataFrame,
    interconnection: str,
    spec: str = "core5",
    causal_only: bool = False,
    bandwidth: float = 48.0,
    ridge_alpha: float = 0.01,
    output_dir: str = "results/empirical",
    tag: str = "2021"
) -> pd.DataFrame:
    """
    Run DCE estimation for a specific interconnection panel.
    """
    print(f"\n=======================================================")
    mode_str = "causal" if causal_only else "retrospective"
    print(f"Running DCE [{mode_str}] on {interconnection} ({tag}, spec={spec})")
    print(f"=======================================================")
    
    scaling = "causal_rolling" if causal_only else "standard"
    grid_data = build_power_grid_microstate(
        df_raw,
        interconnection=interconnection,
        spec=spec,
        scaling=scaling
    )
    
    X = grid_data.microstate_matrix
    T_total, p_dim = X.shape
    print(f"Microstate matrix shape: T={T_total}, p={p_dim}, BAs={len(grid_data.ba_list)}")
    
    if interconnection.upper() == "ERCOT":
        macro_dims = [1, 2, 3, 4] if spec == "core5" else [1, 2, 3, 4, 6]
    else:
        macro_dims = [1, 2, 4, 8, 16]
        
    model = LocalLinearGaussianDCE(
        macro_dims=macro_dims,
        bandwidth=bandwidth,
        kernel_type="gaussian",
        causal_only=causal_only,
        ridge_alpha=ridge_alpha,
        selection_criterion="normalized"
    )
    
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    print(f"Estimation finished in {elapsed:.2f}s ({elapsed/len(model.emergence_)*1000:.2f} ms/step)")
    
    # Align timestamps (transitions correspond to t = 0 ... T-2)
    timestamps = grid_data.timestamps[:-1]
    vre = grid_data.vre_penetration[:-1]
    net_load = grid_data.net_load[:-1]
    fe = grid_data.forecast_error[:-1]
    
    # Compile confsets as string
    confsets_str = []
    confsets_size = []
    if model.causal_dimension_confset_ is not None:
        for c in model.causal_dimension_confset_:
            confsets_str.append(",".join(map(str, sorted(c))))
            confsets_size.append(len(c))
    else:
        confsets_str = [str(q) for q in model.causal_dimension_]
        confsets_size = [1] * len(model.causal_dimension_)
        
    # Macro EI for optimal q
    macro_ei_optimal = np.zeros_like(model.micro_ei_)
    for t_idx, q_star in enumerate(model.causal_dimension_):
        macro_ei_optimal[t_idx] = model.macro_ei_[q_star][t_idx]
        
    # Raw emergence and density gain
    raw_dce = model.optimal_dce_raw_ if hasattr(model, "optimal_dce_raw_") else (macro_ei_optimal - model.micro_ei_)
    density_dce = model.optimal_dce_density_ if hasattr(model, "optimal_dce_density_") else model.emergence_
    
    res_df = pd.DataFrame({
        "timestamp": timestamps,
        "interconnection": interconnection,
        "mode": mode_str,
        "spec": spec,
        "vre_penetration": vre,
        "net_load_mw": net_load,
        "forecast_error": fe,
        "micro_ei": model.micro_ei_,
        "macro_ei": macro_ei_optimal,
        "dcd_pr": model.dcd_pr_,
        "dcd_entropy": model.dcd_entropy_,
        "q90": model.q90_,
        "ccg_optimal": density_dce,
        "dce_raw_optimal": raw_dce,
        "dce_raw": raw_dce,
        "dce_density": density_dce,
        "dce_norm": density_dce,
        "q_star": model.causal_dimension_,
        "q_star_raw": getattr(model, "causal_dimension_raw_", model.causal_dimension_),
        "q_star_density": getattr(model, "causal_dimension_density_", model.causal_dimension_),
        "q_confset": confsets_str,
        "q_confset_size": confsets_size
    })
    
    for i in range(p_dim):
        res_df[f"causal_spectrum_{i+1}"] = model.causal_spectrum_[:, i]
    
    for q in macro_dims:
        res_df[f"macro_ei_q{q}"] = model.macro_ei_[q]
        res_df[f"dce_raw_q{q}"] = model.macro_ei_[q] - model.micro_ei_
        res_df[f"dce_density_q{q}"] = (model.macro_ei_[q] / q) - (model.micro_ei_ / p_dim)
        res_df[f"dce_norm_q{q}"] = res_df[f"dce_density_q{q}"]
        
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{interconnection.lower()}_dce_{tag}_{mode_str}.parquet"
    out_path = os.path.join(output_dir, fname)
    res_df.to_parquet(out_path, index=False)
    print(f"Saved results to {out_path} ({len(res_df)} timestamps)")
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Run Empirical DCE estimation on real EIA-930 grids.")
    parser.add_argument("--period", default="2021", choices=["2021", "2022h2", "all"])
    parser.add_argument("--interconnection", default="all", choices=["all", "ERCOT", "Western", "Eastern"])
    parser.add_argument("--mode", default="both", choices=["both", "retrospective", "causal"])
    parser.add_argument("--ridge-alpha", type=float, default=0.01, help="Regularization ridge parameter")
    parser.add_argument("--output_dir", default="results/empirical")
    args = parser.parse_args()
    
    periods = ["2021"] if args.period == "2021" else (["2022h2"] if args.period == "2022h2" else ["2021", "2022h2"])
    interconnections = ["ERCOT", "Western", "Eastern"] if args.interconnection == "all" else [args.interconnection]
    causal_modes = [False, True] if args.mode == "both" else ([True] if args.mode == "causal" else [False])
    
    for period in periods:
        print(f"\n>>> Loading EIA-930 archive for period: {period}")
        df_raw = load_real_eia930_archive(period=period)
        
        for inter in interconnections:
            spec = "fuel_extended" if inter == "ERCOT" else "core5"
            for causal_flag in causal_modes:
                run_estimation_for_panel(
                    df_raw=df_raw,
                    interconnection=inter,
                    spec=spec,
                    causal_only=causal_flag,
                    bandwidth=48.0,
                    ridge_alpha=args.ridge_alpha,
                    output_dir=args.output_dir,
                    tag=period
                )
                
    print("\nAll empirical DCE estimations completed successfully!")


if __name__ == "__main__":
    main()
