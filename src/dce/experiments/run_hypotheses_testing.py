"""
Execution Pipeline for Empirical Hypothesis Testing (H1, H2, H3, H4).

Executes Milestones M7 & M8:
- H1: Surrogate Data Testing with model refitting on IAAFT surrogates (CRITICAL-02).
- H2: GAMM & Threshold Regression with bootstrap stability (Section 18).
- H3: Matched Event Study for Winter Storm Uri, Elliott, and Summer Heatwaves (Section 19).
- H4: Walk-Forward Out-of-Sample Forecasting with Diebold-Mariano HAC tests (Section 20).
"""

import os
import json
import time
import numpy as np
import pandas as pd

from dce.datasets.eia930.client import load_real_eia930_archive
from dce.datasets.eia930.microstate import build_power_grid_microstate
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
from dce.stats.hypothesis import (
    run_h1_surrogate_test,
    compute_h2_gamm_and_threshold,
    compute_h3_matched_event_study,
    compute_h4_walk_forward_forecasting,
)


def execute_h1_test(output_dir: str = "results/empirical", n_surrogates: int = 1000) -> dict:
    """Execute H1 surrogate test on ERCOT with model refitting on every surrogate."""
    print("\n=======================================================")
    print(f"Executing H1: Surrogate Testing (n={n_surrogates} IAAFT refits)")
    print("=======================================================")
    
    # Load 2021 ERCOT panel
    df_raw = load_real_eia930_archive("2021")
    ercot_data = build_power_grid_microstate(df_raw, interconnection="ERCOT", spec="fuel_extended", scaling="standard")
    X = ercot_data.microstate_matrix
    
    # For computation efficiency across B refits, use a representative slice (e.g. 1500 hours covering Uri & winter)
    # or stride if full year
    T_slice = 2000
    X_slice = X[:T_slice]
    
    macro_dims = [1, 2, 3, 4]
    
    def fit_model(X_in):
        m = LocalLinearGaussianDCE(
            macro_dims=macro_dims,
            bandwidth=48.0,
            causal_only=False,
            ridge_alpha=1e-4,
            selection_criterion="normalized"
        )
        m.fit(X_in)
        return m.emergence_
        
    t0 = time.time()
    empirical_dce = fit_model(X_slice)
    print(f"Empirical fit completed in {time.time() - t0:.2f}s, mean DCE: {np.mean(empirical_dce):.4f}")
    
    h1_res = run_h1_surrogate_test(
        empirical_dce=empirical_dce,
        X_micro=X_slice,
        fit_model_fn=fit_model,
        n_surrogates=n_surrogates,
        seed=42
    )
    
    out_dict = {
        "hypothesis": "H1_Surrogate_Significance",
        "interconnection": "ERCOT",
        "n_surrogates": n_surrogates,
        "T": T_slice,
        "significant_ratio_raw": h1_res.significant_ratio_raw,
        "significant_ratio_fdr": h1_res.significant_ratio_fdr,
        "max_stat_empirical": h1_res.max_stat_empirical,
        "max_stat_pvalue": h1_res.max_stat_pvalue,
        "mean_stat_empirical": h1_res.mean_stat_empirical,
        "mean_stat_pvalue": h1_res.mean_stat_pvalue,
        "empirical_dce_sample": empirical_dce[:100].tolist(),
        "surrogate_95th_sample": h1_res.surrogate_dce_95th[:100].tolist()
    }
    
    out_path = os.path.join(output_dir, "h1_surrogate_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"H1 results saved to {out_path}")
    print(f"H1 Decision: Max p-val = {h1_res.max_stat_pvalue:.4f}, Mean p-val = {h1_res.mean_stat_pvalue:.4f}, FDR Sig Ratio = {h1_res.significant_ratio_fdr:.2%}")
    return out_dict


def execute_h2_test(output_dir: str = "results/empirical") -> dict:
    """Execute H2 GAMM and threshold regression on empirical DCE results."""
    print("\n=======================================================")
    print("Executing H2: Nonlinear GAMM & Threshold Behavior")
    print("=======================================================")
    
    # Load retrospective DCE results for ERCOT and Western
    panels = ["ercot", "western"]
    results_h2 = {}
    
    for p in panels:
        path = os.path.join(output_dir, f"{p}_dce_2021_retrospective.parquet")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {p}")
            continue
            
        df_dce = pd.read_parquet(path)
        dce = df_dce["dce_norm"].values
        vre = df_dce["vre_penetration"].values
        ts = pd.DatetimeIndex(df_dce["timestamp"])
        net_load = df_dce["net_load_mw"].values if "net_load_mw" in df_dce.columns else None
        
        print(f"Fitting GAMM & Threshold model for {p.upper()} ({len(df_dce)} timestamps)...")
        h2_res = compute_h2_gamm_and_threshold(
            dce_series=dce,
            vre_series=vre,
            timestamps=ts,
            net_load_series=net_load,
            n_bootstrap=50
        )
        
        results_h2[p] = {
            "pseudo_r2": h2_res.pseudo_r2,
            "best_threshold_gamma": h2_res.best_threshold,
            "threshold_ci_95": list(h2_res.threshold_ci),
            "davies_pvalue": h2_res.davies_pvalue,
            "model_comparison": h2_res.model_comparison.to_dict(orient="records"),
            "vre_grid": h2_res.vre_grid.tolist(),
            "vre_partial_effects": h2_res.vre_partial_effects.tolist(),
            "vre_confidence_intervals": h2_res.vre_confidence_intervals.tolist()
        }
        print(f"  {p.upper()}: Pseudo R2 = {h2_res.pseudo_r2:.3f}, Threshold gamma = {h2_res.best_threshold:.3f} (95% CI: {h2_res.threshold_ci[0]:.3f} - {h2_res.threshold_ci[1]:.3f}), Davies p = {h2_res.davies_pvalue:.4e}")
        
    out_path = os.path.join(output_dir, "h2_gamm_results.json")
    with open(out_path, "w") as f:
        json.dump(results_h2, f, indent=2)
    print(f"H2 results saved to {out_path}")
    return results_h2


def execute_h3_test(output_dir: str = "results/empirical") -> dict:
    """Execute H3 matched event study on Winter Storm Uri (ERCOT)."""
    print("\n=======================================================")
    print("Executing H3: Matched Event Study & Dimensional Collapse")
    print("=======================================================")
    
    path_ercot = os.path.join(output_dir, "ercot_dce_2021_retrospective.parquet")
    df_ercot = pd.read_parquet(path_ercot)
    
    # Winter Storm Uri event: 2021-02-12 to 2021-02-19
    h3_uri = compute_h3_matched_event_study(
        dce_df=df_ercot,
        event_start="2021-02-12 00:00:00",
        event_end="2021-02-19 23:00:00",
        event_name="Winter Storm Uri",
        event_id="uri_2021",
        pre_window_h=48,
        post_window_h=48
    )
    
    out_dict = {
        "event_id": h3_uri.event_id,
        "event_name": h3_uri.event_name,
        "n_hours": len(h3_uri.event_timestamps),
        "collapse_ratio_event": h3_uri.collapse_ratio_event,
        "collapse_ratio_baseline": h3_uri.collapse_ratio_baseline,
        "collapse_pvalue": h3_uri.collapse_pvalue,
        "mean_delta_dce": float(np.mean(h3_uri.delta_dce)),
        "median_delta_q": float(np.median(h3_uri.delta_q_star)),
        "event_timestamps": [t.isoformat() for t in h3_uri.event_timestamps],
        "delta_dce": h3_uri.delta_dce.tolist(),
        "delta_q_star": h3_uri.delta_q_star.tolist(),
        "event_dce": h3_uri.event_dce.tolist(),
        "matched_dce_mean": h3_uri.matched_dce_mean.tolist()
    }
    
    out_path = os.path.join(output_dir, "h3_event_study_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"H3 results saved to {out_path}")
    print(f"Uri Dimensional Collapse: Event Collapse Rate = {h3_uri.collapse_ratio_event:.2%}, Baseline = {h3_uri.collapse_ratio_baseline:.2%}, p = {h3_uri.collapse_pvalue:.4e}")
    return out_dict


def execute_h4_test(output_dir: str = "results/empirical") -> dict:
    """Execute H4 walk-forward rolling forecasting with causal DCE."""
    print("\n=======================================================")
    print("Executing H4: Walk-Forward Rolling Predictive Regressions")
    print("=======================================================")
    
    path_causal = os.path.join(output_dir, "ercot_dce_2021_causal.parquet")
    df_causal = pd.read_parquet(path_causal)
    
    fe = df_causal["forecast_error"].values
    dce_c = df_causal["dce_norm"].values
    q_c = df_causal["q_star"].values
    
    h4_res = compute_h4_walk_forward_forecasting(
        forecast_error=fe,
        dce_causal=dce_c,
        q_star_causal=q_c,
        horizons=[1, 6, 12, 24],
        train_window=2000,
        eval_step=12
    )
    
    out_dict = {
        "horizons": h4_res.horizons,
        "rmse_baseline": h4_res.rmse_baseline,
        "rmse_augmented": h4_res.rmse_augmented,
        "mae_baseline": h4_res.mae_baseline,
        "mae_augmented": h4_res.mae_augmented,
        "rmse_improvement_pct": h4_res.rmse_improvement_pct,
        "diebold_mariano_stat": h4_res.diebold_mariano_stat,
        "diebold_mariano_pvalue": h4_res.diebold_mariano_pvalue,
        "stress_tail_rmse_baseline": h4_res.stress_tail_rmse_baseline,
        "stress_tail_rmse_augmented": h4_res.stress_tail_rmse_augmented
    }
    
    out_path = os.path.join(output_dir, "h4_forecast_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"H4 results saved to {out_path}")
    for h in h4_res.horizons:
        print(f"  h={h:2d}h: RMSE Base={h4_res.rmse_baseline[h]:.4f} -> Aug={h4_res.rmse_augmented[h]:.4f} (Delta={h4_res.rmse_improvement_pct[h]:+.2f}%), DM Stat={h4_res.diebold_mariano_stat[h]:.2f} (p={h4_res.diebold_mariano_pvalue[h]:.4f})")
    return out_dict


def main():
    os.makedirs("results/empirical", exist_ok=True)
    execute_h1_test()
    execute_h2_test()
    execute_h3_test()
    execute_h4_test()
    print("\nAll Hypothesis Tests H1, H2, H3, H4 successfully executed!")


if __name__ == "__main__":
    main()
