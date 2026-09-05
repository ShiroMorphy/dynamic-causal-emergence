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
from scipy import stats

from dce.datasets.eia930.client import load_real_eia930_archive
from dce.datasets.eia930.microstate import build_power_grid_microstate
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
from typing import Tuple
from dce.stats.surrogates import generate_multivariate_surrogates
from dce.stats.hypothesis import (
    run_h1_surrogate_test,
    compute_surrogate_significance_from_ensemble,
    compute_h2_gamm_and_threshold,
    compute_h3_matched_event_study,
    compute_h4_walk_forward_forecasting,
)


def _fit_single_surrogate(X_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-level picklable surrogate model fitting routine returning (dcd_pr, ccg)."""
    m = LocalLinearGaussianDCE(
        macro_dims=[2],
        bandwidth=48.0,
        causal_only=False,
        ridge_alpha=1e-4,
        selection_criterion="density"
    )
    m.fit(X_in)
    return m.dcd_pr_, m.optimal_dce_density_


def execute_h1_test(output_dir: str = "results/empirical", n_surrogates: int = 1000) -> dict:
    """Execute H1 surrogate test on ERCOT with model refitting on every surrogate."""
    print("\n=======================================================")
    print(f"Executing H1: Surrogate Testing on DCD & CCG (n={n_surrogates} IAAFT refits)")
    print("=======================================================")
    
    # Load 2021 ERCOT panel
    df_raw = load_real_eia930_archive("2021")
    ercot_data = build_power_grid_microstate(df_raw, interconnection="ERCOT", spec="fuel_extended", scaling="standard")
    X = ercot_data.microstate_matrix
    
    # Use representative 2000-hour slice covering Winter Storm Uri & winter stress
    T_slice = 2000
    X_slice = X[:T_slice]
    
    t0 = time.time()
    m_emp = LocalLinearGaussianDCE(
        macro_dims=[2],
        bandwidth=48.0,
        causal_only=False,
        ridge_alpha=1e-4,
        selection_criterion="density"
    )
    m_emp.fit(X_slice)
    empirical_dcd = m_emp.dcd_pr_
    empirical_ccg = m_emp.optimal_dce_density_
    print(f"Empirical fit completed in {time.time() - t0:.2f}s: Mean DCD_PR = {np.mean(empirical_dcd):.4f}, Mean CCG = {np.mean(empirical_ccg):.4f}")
    
    print(f"Generating {n_surrogates} strictly accepted multivariate IAAFT surrogates (T={X_slice.shape[0]}, p={X_slice.shape[1]})...")
    t_gen = time.time()
    surrogates = generate_multivariate_surrogates(X_slice, n_surrogates=n_surrogates, seed=42, strictly_accepted=True)
    print(f"Surrogate generation completed in {time.time() - t_gen:.2f}s")
    
    print(f"Refitting DCE model on {n_surrogates} surrogates in parallel across worker processes...")
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    workers = min(n_surrogates, max(1, (os.cpu_count() or 4) - 1))
    t_refit = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            surr_results = list(executor.map(_fit_single_surrogate, surrogates))
    except Exception as e:
        print(f"ProcessPoolExecutor fallback ({e}), using ThreadPoolExecutor...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            surr_results = list(executor.map(_fit_single_surrogate, surrogates))
            
    surr_dcd_ensemble = np.array([r[0] for r in surr_results], dtype=np.float64)
    surr_ccg_ensemble = np.array([r[1] for r in surr_results], dtype=np.float64)
    dt_total = time.time() - t_refit
    print(f"All {n_surrogates} surrogate model refits completed in {dt_total:.2f}s ({n_surrogates / max(dt_total, 0.001):.2f} fits/sec)")
    
    # Run H1 on DCD_PR (Primary Dimensionality Metric - Contraction)
    print(f"\n--- Evaluating Primary Metric: DCD_PR (Dynamic Causal Dimensionality) ---")
    h1_res_dcd = compute_surrogate_significance_from_ensemble(
        empirical_dce=empirical_dcd,
        surr_dce_ensemble=surr_dcd_ensemble,
        test_direction="less"
    )
    
    # Run H1 on CCG (Secondary Concentration Metric - Emergence)
    print(f"\n--- Evaluating Secondary Metric: CCG (Causal Concentration Gain) ---")
    h1_res_ccg = compute_surrogate_significance_from_ensemble(
        empirical_dce=empirical_ccg,
        surr_dce_ensemble=surr_ccg_ensemble,
        test_direction="greater"
    )
    
    out_dict = {
        "hypothesis": "H1_Surrogate_Significance",
        "interconnection": "ERCOT",
        "n_surrogates": n_surrogates,
        "T": T_slice,
        "dcd_pr": {
            "mean_stat_empirical": h1_res_dcd.mean_stat_empirical,
            "mean_stat_pvalue": h1_res_dcd.mean_stat_pvalue,
            "max_stat_empirical": h1_res_dcd.max_stat_empirical,
            "max_stat_pvalue": h1_res_dcd.max_stat_pvalue,
            "significant_ratio_raw": h1_res_dcd.significant_ratio_raw,
            "significant_ratio_fdr": h1_res_dcd.significant_ratio_fdr,
            "empirical_sample": empirical_dcd[:100].tolist(),
            "surrogate_95th_sample": h1_res_dcd.surrogate_dce_95th[:100].tolist()
        },
        "ccg": {
            "mean_stat_empirical": h1_res_ccg.mean_stat_empirical,
            "mean_stat_pvalue": h1_res_ccg.mean_stat_pvalue,
            "max_stat_empirical": h1_res_ccg.max_stat_empirical,
            "max_stat_pvalue": h1_res_ccg.max_stat_pvalue,
            "significant_ratio_raw": h1_res_ccg.significant_ratio_raw,
            "significant_ratio_fdr": h1_res_ccg.significant_ratio_fdr,
            "empirical_sample": empirical_ccg[:100].tolist(),
            "surrogate_95th_sample": h1_res_ccg.surrogate_dce_95th[:100].tolist()
        },
        # Backward-compatibility root fields based on CCG / emergence
        "significant_ratio_raw": h1_res_ccg.significant_ratio_raw,
        "significant_ratio_fdr": h1_res_ccg.significant_ratio_fdr,
        "max_stat_empirical": h1_res_ccg.max_stat_empirical,
        "max_stat_pvalue": h1_res_ccg.max_stat_pvalue,
        "mean_stat_empirical": h1_res_ccg.mean_stat_empirical,
        "mean_stat_pvalue": h1_res_ccg.mean_stat_pvalue,
        "empirical_dce_sample": empirical_ccg[:100].tolist(),
        "surrogate_95th_sample": h1_res_ccg.surrogate_dce_95th[:100].tolist()
    }
    
    out_path = os.path.join(output_dir, "h1_surrogate_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nH1 results saved to {out_path}")
    print(f"H1 Summary:")
    print(f"  DCD_PR: Mean={h1_res_dcd.mean_stat_empirical:.4f} (p={h1_res_dcd.mean_stat_pvalue:.4f}), FDR Sig={h1_res_dcd.significant_ratio_fdr:.2%}")
    print(f"  CCG:    Mean={h1_res_ccg.mean_stat_empirical:.4f} (p={h1_res_ccg.mean_stat_pvalue:.4f}), FDR Sig={h1_res_ccg.significant_ratio_fdr:.2%}")
    return out_dict


def execute_h2_test(output_dir: str = "results/empirical") -> dict:
    """Execute H2 GAMM and threshold regression on empirical DCD and CCG results."""
    print("\n=======================================================")
    print("Executing H2: Nonlinear GAMM & Threshold Behavior (DCD_PR & CCG)")
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
        dcd_pr = df_dce["dcd_pr"].values
        ccg = df_dce["ccg_optimal"].values
        vre = df_dce["vre_penetration"].values
        ts = pd.DatetimeIndex(df_dce["timestamp"])
        net_load = df_dce["net_load_mw"].values if "net_load_mw" in df_dce.columns else None
        
        print(f"Fitting GAMM & Threshold model for {p.upper()} on DCD_PR ({len(df_dce)} timestamps)...")
        h2_res_pr = compute_h2_gamm_and_threshold(
            dce_series=dcd_pr,
            vre_series=vre,
            timestamps=ts,
            net_load_series=net_load,
            n_bootstrap=50
        )
        
        print(f"Fitting GAMM & Threshold model for {p.upper()} on CCG ({len(df_dce)} timestamps)...")
        h2_res_ccg = compute_h2_gamm_and_threshold(
            dce_series=ccg,
            vre_series=vre,
            timestamps=ts,
            net_load_series=net_load,
            n_bootstrap=50
        )
        
        results_h2[p] = {
            # Primary: DCD_PR
            "dcd_pr": {
                "pseudo_r2": h2_res_pr.pseudo_r2,
                "best_threshold_gamma": h2_res_pr.best_threshold,
                "threshold_ci_95": list(h2_res_pr.threshold_ci),
                "davies_pvalue": h2_res_pr.davies_pvalue,
                "model_comparison": h2_res_pr.model_comparison.to_dict(orient="records"),
                "vre_grid": h2_res_pr.vre_grid.tolist(),
                "vre_partial_effects": h2_res_pr.vre_partial_effects.tolist(),
                "vre_confidence_intervals": h2_res_pr.vre_confidence_intervals.tolist()
            },
            # Secondary: CCG
            "ccg": {
                "pseudo_r2": h2_res_ccg.pseudo_r2,
                "best_threshold_gamma": h2_res_ccg.best_threshold,
                "threshold_ci_95": list(h2_res_ccg.threshold_ci),
                "davies_pvalue": h2_res_ccg.davies_pvalue,
                "model_comparison": h2_res_ccg.model_comparison.to_dict(orient="records"),
                "vre_grid": h2_res_ccg.vre_grid.tolist(),
                "vre_partial_effects": h2_res_ccg.vre_partial_effects.tolist(),
                "vre_confidence_intervals": h2_res_ccg.vre_confidence_intervals.tolist()
            },
            # Root-level backward-compatible fields matching primary DCD_PR
            "pseudo_r2": h2_res_pr.pseudo_r2,
            "best_threshold_gamma": h2_res_pr.best_threshold,
            "threshold_ci_95": list(h2_res_pr.threshold_ci),
            "davies_pvalue": h2_res_pr.davies_pvalue,
            "model_comparison": h2_res_pr.model_comparison.to_dict(orient="records"),
            "vre_grid": h2_res_pr.vre_grid.tolist(),
            "vre_partial_effects": h2_res_pr.vre_partial_effects.tolist(),
            "vre_confidence_intervals": h2_res_pr.vre_confidence_intervals.tolist()
        }
        print(f"  {p.upper()} [DCD_PR]: Pseudo R2 = {h2_res_pr.pseudo_r2:.3f}, Threshold gamma = {h2_res_pr.best_threshold:.3f} (95% CI: {h2_res_pr.threshold_ci[0]:.3f} - {h2_res_pr.threshold_ci[1]:.3f}), Davies p = {h2_res_pr.davies_pvalue:.4e}")
        print(f"  {p.upper()} [CCG]:    Pseudo R2 = {h2_res_ccg.pseudo_r2:.3f}, Threshold gamma = {h2_res_ccg.best_threshold:.3f} (95% CI: {h2_res_ccg.threshold_ci[0]:.3f} - {h2_res_ccg.threshold_ci[1]:.3f}), Davies p = {h2_res_ccg.davies_pvalue:.4e}")
        
    out_path = os.path.join(output_dir, "h2_gamm_results.json")
    with open(out_path, "w") as f:
        json.dump(results_h2, f, indent=2)
    print(f"H2 results saved to {out_path}")
    return results_h2


def execute_h3_test(output_dir: str = "results/empirical", n_permutations: int = 1000) -> dict:
    """Execute H3 matched event study and block permutation testing across 5 major grid stress events with 14-day global exclusion."""
    print("\n=======================================================")
    print(f"Executing H3: Multi-Event Study & Block Permutation Tests (n_events=5, B={n_permutations}, 14d exclusion)")
    print("=======================================================")
    
    events_config = [
        {
            "event_id": "uri_2021",
            "event_name": "Winter Storm Uri",
            "interconnection": "ERCOT",
            "parquet_file": "ercot_dce_2021_retrospective.parquet",
            "event_start": "2021-02-12 00:00:00",
            "event_end": "2021-02-19 23:00:00",
            "pre_window_h": 48,
            "post_window_h": 48
        },
        {
            "event_id": "heat_dome_2021",
            "event_name": "PNW Heat Dome",
            "interconnection": "Western",
            "parquet_file": "western_dce_2021_retrospective.parquet",
            "event_start": "2021-06-25 00:00:00",
            "event_end": "2021-07-02 23:00:00",
            "pre_window_h": 48,
            "post_window_h": 48
        },
        {
            "event_id": "ida_2021",
            "event_name": "Hurricane Ida",
            "interconnection": "Eastern",
            "parquet_file": "eastern_dce_2021_retrospective.parquet",
            "event_start": "2021-08-29 00:00:00",
            "event_end": "2021-09-03 23:00:00",
            "pre_window_h": 48,
            "post_window_h": 48
        },
        {
            "event_id": "tx_heatwave_2022",
            "event_name": "Texas Summer Heatwave",
            "interconnection": "ERCOT",
            "parquet_file": "ercot_dce_2022h2_retrospective.parquet",
            "event_start": "2022-07-08 00:00:00",
            "event_end": "2022-07-20 23:00:00",
            "pre_window_h": 48,
            "post_window_h": 48
        },
        {
            "event_id": "elliott_2022",
            "event_name": "Winter Storm Elliott",
            "interconnection": "Eastern",
            "parquet_file": "eastern_dce_2022h2_retrospective.parquet",
            "event_start": "2022-12-22 00:00:00",
            "event_end": "2022-12-26 23:00:00",
            "pre_window_h": 48,
            "post_window_h": 48
        }
    ]
    
    all_events = {}
    summary_list = []
    uri_result_dict = None
    rng = np.random.RandomState(42)
    p_blocks_primary = []
    
    for cfg in events_config:
        parquet_path = os.path.join(output_dir, cfg["parquet_file"])
        if not os.path.exists(parquet_path):
            print(f"Warning: {parquet_path} not found, skipping {cfg['event_name']}")
            continue
            
        df_panel = pd.read_parquet(parquet_path)
        ts = pd.to_datetime(df_panel["timestamp"], utc=True)
        t_start = pd.to_datetime(cfg["event_start"], utc=True)
        t_end = pd.to_datetime(cfg["event_end"], utc=True)
        
        # 14-day global exclusion window around this event
        t_excl_start = t_start - pd.Timedelta(days=14)
        t_excl_end = t_end + pd.Timedelta(days=14)
        
        is_event = (ts >= t_start) & (ts <= t_end)
        is_excluded = (ts >= t_excl_start) & (ts <= t_excl_end)
        n_ev_hours = int(is_event.sum())
        
        # Primary quantities
        dcd_vals = df_panel["dcd_pr"].values
        ccg_vals = df_panel["ccg_optimal"].values
        q90_vals = df_panel["q90"].values
        q_star_vals = df_panel["q_star"].values if "q_star" in df_panel.columns else q90_vals
        
        # Baseline pool strictly outside the 14-day exclusion window
        base_mask = ~is_excluded
        base_dcd = dcd_vals[base_mask]
        base_ccg = ccg_vals[base_mask]
        base_q90 = q90_vals[base_mask]
        
        mean_base_ccg = float(np.mean(base_ccg))
        obs_delta_ccg = float(np.mean(ccg_vals[is_event]) - mean_base_ccg)
        
        mean_base_dcd = float(np.mean(base_dcd))
        obs_delta_dcd = float(np.mean(dcd_vals[is_event]) - mean_base_dcd)
        
        # Binary coordination rate (q* = 2) for ERCOT / grids
        binary_event = float(np.mean(q_star_vals[is_event] == 2))
        binary_base = float(np.mean(q_star_vals[base_mask] == 2))
        
        # Continuous Dimensional Collapse threshold: 10th percentile of baseline DCD_PR
        dcd_thresh = float(np.percentile(base_dcd, 10.0))
        collapse_rate_event = float(np.mean(dcd_vals[is_event] <= dcd_thresh))
        collapse_rate_base = float(np.mean(base_dcd <= dcd_thresh))
        
        # Matched event study window
        h3_res = compute_h3_matched_event_study(
            dce_df=df_panel,
            event_start=cfg["event_start"],
            event_end=cfg["event_end"],
            event_name=cfg["event_name"],
            event_id=cfg["event_id"],
            pre_window_h=cfg["pre_window_h"],
            post_window_h=cfg["post_window_h"]
        )
        
        # Contiguous Block Permutation Test across non-excluded baseline
        valid_starts = []
        is_excl_arr = is_excluded.values
        for i in range(len(df_panel) - n_ev_hours + 1):
            if not np.any(is_excl_arr[i:i + n_ev_hours]):
                valid_starts.append(i)
                
        valid_starts = np.array(valid_starts)
        if len(valid_starts) > 0 and n_permutations > 0:
            sampled_starts = rng.choice(
                valid_starts,
                size=min(n_permutations, len(valid_starts)),
                replace=(len(valid_starts) < n_permutations)
            )
            
            # Permutation for CCG (concentration gain)
            perm_ccg_deltas = np.array([
                np.mean(ccg_vals[s:s + n_ev_hours]) - mean_base_ccg for s in sampled_starts
            ])
            p_val_ccg_two_sided = float(
                (np.sum(np.abs(perm_ccg_deltas) >= np.abs(obs_delta_ccg)) + 1.0) / (len(perm_ccg_deltas) + 1.0)
            )
            p_val_ccg_surge = float(
                (np.sum(perm_ccg_deltas >= obs_delta_ccg) + 1.0) / (len(perm_ccg_deltas) + 1.0)
            )
            
            # Permutation for DCD_PR
            perm_dcd_deltas = np.array([
                np.mean(dcd_vals[s:s + n_ev_hours]) - mean_base_dcd for s in sampled_starts
            ])
            p_val_dcd_two_sided = float(
                (np.sum(np.abs(perm_dcd_deltas) >= np.abs(obs_delta_dcd)) + 1.0) / (len(perm_dcd_deltas) + 1.0)
            )
            
            # Permutation for dimensional collapse
            perm_collapses = np.array([
                np.mean(dcd_vals[s:s + n_ev_hours] <= dcd_thresh) for s in sampled_starts
            ])
            p_val_collapse_perm = float(
                (np.sum(perm_collapses >= collapse_rate_event) + 1.0) / (len(perm_collapses) + 1.0)
            )
            
            # Primary inference p-value is the block permutation p-value
            primary_p_block = p_val_ccg_two_sided
        else:
            primary_p_block = 1.0
            p_val_ccg_two_sided = 1.0
            p_val_ccg_surge = 1.0
            p_val_dcd_two_sided = 1.0
            p_val_collapse_perm = 1.0
            
        p_blocks_primary.append(primary_p_block)
        
        # Event trajectory slices for plotting
        w_start = max(0, np.where(is_event)[0][0] - cfg["pre_window_h"])
        w_end = min(len(df_panel), np.where(is_event)[0][-1] + cfg["post_window_h"] + 1)
        w_slice = df_panel.iloc[w_start:w_end]
        
        ev_dict = {
            "event_id": cfg["event_id"],
            "event_name": cfg["event_name"],
            "interconnection": cfg["interconnection"],
            "event_start": cfg["event_start"],
            "event_end": cfg["event_end"],
            "n_hours": len(w_slice),
            "event_core_hours": n_ev_hours,
            "exclusion_window_hours": int(is_excluded.sum()),
            "ccg_event_mean": float(np.mean(ccg_vals[is_event])),
            "ccg_baseline_mean": mean_base_ccg,
            "obs_delta_ccg": obs_delta_ccg,
            "p_val_ccg_block_perm": p_val_ccg_two_sided,
            "p_val_ccg_surge_block_perm": p_val_ccg_surge,
            "dcd_pr_event_mean": float(np.mean(dcd_vals[is_event])),
            "dcd_pr_baseline_mean": mean_base_dcd,
            "obs_delta_dcd_pr": obs_delta_dcd,
            "p_val_dcd_block_perm": p_val_dcd_two_sided,
            "binary_coordination_event": binary_event,
            "binary_coordination_baseline": binary_base,
            "collapse_threshold_dcd10": dcd_thresh,
            "collapse_ratio_event": collapse_rate_event,
            "collapse_ratio_baseline": collapse_rate_base,
            "collapse_pvalue_analytical": h3_res.collapse_pvalue,
            "collapse_pvalue_block_perm": p_val_collapse_perm,
            "primary_p_block": primary_p_block,
            "event_timestamps": [t.isoformat() for t in pd.to_datetime(w_slice["timestamp"], utc=True)],
            "event_dce": w_slice["ccg_optimal"].tolist(),
            "matched_dce_mean": [mean_base_ccg] * len(w_slice),
            "delta_dce": (w_slice["ccg_optimal"].values - mean_base_ccg).tolist(),
            "event_dcd_pr": w_slice["dcd_pr"].tolist(),
            "event_q90": w_slice["q90"].tolist(),
            "event_q_star": w_slice["q_star"].tolist() if "q_star" in w_slice.columns else w_slice["q90"].tolist()
        }
        
        all_events[cfg["event_id"]] = ev_dict
        summary_list.append({
            "event_id": cfg["event_id"],
            "name": cfg["event_name"],
            "grid": cfg["interconnection"],
            "hours": n_ev_hours,
            "delta_ccg": f"{obs_delta_ccg:+.3f}",
            "ccg_event": f"{float(np.mean(ccg_vals[is_event])):.3f}",
            "ccg_base": f"{mean_base_ccg:.3f}",
            "p_block": f"{primary_p_block:.4f}",
            "delta_dcd_pr": f"{obs_delta_dcd:+.3f}",
            "p_block_dcd": f"{p_val_dcd_two_sided:.4f}"
        })
        
        if cfg["event_id"] == "uri_2021":
            uri_result_dict = ev_dict
            
        print(f"  {cfg['event_name']} ({cfg['interconnection']}): "
              f"CCG: {float(np.mean(ccg_vals[is_event])):.3f} vs Base {mean_base_ccg:.3f} (Delta={obs_delta_ccg:+.3f}, p_block={primary_p_block:.4f}) | "
              f"DCD_PR: {float(np.mean(dcd_vals[is_event])):.2f} vs Base {mean_base_dcd:.2f} (p_block={p_val_dcd_two_sided:.4f})")
              
    # Aggregate Multi-Event Fisher Panel Statistic across all 5 events
    p_arr = np.array(p_blocks_primary, dtype=np.float64)
    t_fisher = float(-2.0 * np.sum(np.log(np.clip(p_arr, 1.0 / (n_permutations + 1.0), 1.0))))
    df_fisher = 2 * len(p_arr)
    p_fisher = float(stats.chi2.sf(t_fisher, df=df_fisher))
    
    fisher_panel = {
        "t_statistic": t_fisher,
        "degrees_of_freedom": df_fisher,
        "p_value": p_fisher,
        "n_events": len(p_arr),
        "event_ids": [cfg["event_id"] for cfg in events_config]
    }
    print(f"\nMulti-Event Aggregate Fisher Panel Statistic: T_Fisher = {t_fisher:.2f}, df = {df_fisher}, p = {p_fisher:.4f}")
    
    # Root dictionary: backward compatible with uri_2021 while containing all new artifacts
    out_dict = dict(uri_result_dict) if uri_result_dict is not None else {}
    out_dict["events"] = all_events
    out_dict["summary_table"] = summary_list
    out_dict["fisher_panel"] = fisher_panel
    
    out_path = os.path.join(output_dir, "h3_event_study_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"H3 multi-event results saved to {out_path}")
    print(pd.DataFrame(summary_list).to_string(index=False))
    return out_dict


def execute_h4_test(output_dir: str = "results/empirical") -> dict:
    """Execute H4 walk-forward rolling forecasting with causal DCD and CCG."""
    print("\n=======================================================")
    print("Executing H4: Walk-Forward Rolling Predictive Regressions (Causal DCD & CCG)")
    print("=======================================================")
    
    path_causal = os.path.join(output_dir, "ercot_dce_2021_causal.parquet")
    df_causal = pd.read_parquet(path_causal)
    
    fe = df_causal["forecast_error"].values
    dcd_c = df_causal["dcd_pr"].values
    ccg_c = df_causal["ccg_optimal"].values
    
    h4_res = compute_h4_walk_forward_forecasting(
        forecast_error=fe,
        dce_causal=dcd_c,
        q_star_causal=ccg_c,
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
    import argparse
    parser = argparse.ArgumentParser(description="Empirical Hypothesis Testing Pipeline (H1 to H4).")
    parser.add_argument("--test", default="all", choices=["all", "h1", "h2", "h3", "h4"], help="Which hypothesis test to execute")
    parser.add_argument("--n-surrogates", type=int, default=200, help="Number of surrogate refits for H1")
    parser.add_argument("--output-dir", default="results/empirical", help="Directory for output JSON artifacts")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test in ("all", "h1"):
        execute_h1_test(output_dir=args.output_dir, n_surrogates=args.n_surrogates)
    if args.test in ("all", "h2"):
        execute_h2_test(output_dir=args.output_dir)
    if args.test in ("all", "h3"):
        execute_h3_test(output_dir=args.output_dir)
    if args.test in ("all", "h4"):
        execute_h4_test(output_dir=args.output_dir)
        
    print("\nHypothesis Tests completed successfully!")


if __name__ == "__main__":
    main()
