#!/usr/bin/env python3
"""
Surrogate Quality Protocol Sensitivity Analysis.
Evaluates independent surrogate acceptance ensembles across auto-spectrum tolerances
tol in {0.08, 0.10, 0.12, 0.15}, recording exact B, N_eval, acceptance yield,
critical values, and p-values for ERCOT and Western interconnections.
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dce.datasets.eia930.client import load_real_eia930_archive
from dce.datasets.eia930.microstate import build_power_grid_microstate
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
from dce.stats.surrogates import generate_multivariate_iaaft_surrogate
from dce.stats.surrogate_validation import evaluate_surrogate_quality, load_frozen_surrogate_tolerances


def _fit_surrogate(s: np.ndarray):
    """Worker function to fit LocalLinearGaussianDCE on a surrogate realization."""
    m = LocalLinearGaussianDCE(
        macro_dims=[2],
        bandwidth=48.0,
        causal_only=False,
        ridge_alpha=0.01,
        selection_criterion="density"
    )
    m.fit(s)
    return float(np.mean(m.dcd_pr_)), float(np.mean(m.optimal_dce_density_))


def run_sensitivity_analysis(target_b: int = 100, output_path: str = "results/empirical/h1_surrogate_sensitivity_results.json"):
    tolerances = [0.08, 0.10, 0.12, 0.15]
    grids = [
        ("ERCOT", "fuel_extended"),
        ("Western", "core5")
    ]
    
    print("Loading 2021 EIA-930 operational data...")
    df_raw = load_real_eia930_archive("2021")
    
    base_tols = load_frozen_surrogate_tolerances()
    results = {}
    
    workers = min(16, max(1, (os.cpu_count() or 4) - 1))
    
    for inter, spec in grids:
        print(f"\n==================================================")
        print(f"Running Surrogate Sensitivity for {inter} ({spec})")
        print(f"==================================================")
        
        m_data = build_power_grid_microstate(df_raw, interconnection=inter, spec=spec, scaling="standard")
        X = m_data.microstate_matrix
        T_len, p = X.shape
        
        # Empirical fit
        m_emp = LocalLinearGaussianDCE(macro_dims=[2], bandwidth=48.0, causal_only=False, ridge_alpha=0.01, selection_criterion="density")
        m_emp.fit(X)
        emp_dcd = float(np.mean(m_emp.dcd_pr_))
        emp_ccg = float(np.mean(m_emp.optimal_dce_density_))
        print(f"Empirical: mean DCD_PR = {emp_dcd:.4f}, mean CCG = {emp_ccg:.4f}")
        
        grid_results = {
            "interconnection": inter,
            "T": T_len,
            "p": p,
            "empirical_dcd_mean": emp_dcd,
            "empirical_ccg_mean": emp_ccg,
            "target_B": target_b,
            "tolerance_evaluations": {}
        }
        
        for tau in tolerances:
            print(f"\n--- Testing tolerance tau = {tau:.2f} ---")
            t_start = time.time()
            
            custom_tols = dict(base_tols)
            custom_tols["auto_spectrum_error"] = tau
            
            accepted_surrogates = []
            attempt = 0
            max_evals = target_b * 15
            
            while len(accepted_surrogates) < target_b and attempt < max_evals:
                surr_seed = 42 + attempt * 1000 + 7
                surr = generate_multivariate_iaaft_surrogate(X, seed=surr_seed)
                report = evaluate_surrogate_quality(X, surr, tolerances=custom_tols)
                if report.accepted:
                    accepted_surrogates.append(surr)
                attempt += 1
                
            n_eval = attempt
            acc_rate = len(accepted_surrogates) / max(n_eval, 1)
            print(f"Accepted {len(accepted_surrogates)}/{target_b} after {n_eval} candidate evaluations ({acc_rate*100:.1f}% yield in {time.time()-t_start:.1f}s)")
            
            t_refit = time.time()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                fit_results = list(executor.map(_fit_surrogate, accepted_surrogates))
                
            surr_dcd_means = np.array([r[0] for r in fit_results])
            surr_ccg_means = np.array([r[1] for r in fit_results])
            
            crit_dcd_05 = float(np.percentile(surr_dcd_means, 5.0))
            crit_ccg_95 = float(np.percentile(surr_ccg_means, 95.0))
            p_val_dcd = float((np.sum(surr_dcd_means <= emp_dcd) + 1) / (len(surr_dcd_means) + 1))
            p_val_ccg = float((np.sum(surr_ccg_means >= emp_ccg) + 1) / (len(surr_ccg_means) + 1))
            
            print(f"tau={tau:.2f}: crit_dcd_05={crit_dcd_05:.4f}, emp_dcd={emp_dcd:.4f}, p_dcd={p_val_dcd:.4f} | min_surr={np.min(surr_dcd_means):.4f}")
            
            grid_results["tolerance_evaluations"][f"{tau:.2f}"] = {
                "tolerance": tau,
                "B_accepted": len(accepted_surrogates),
                "N_evaluated": n_eval,
                "acceptance_yield": acc_rate,
                "critical_value_dcd_05": crit_dcd_05,
                "critical_value_ccg_95": crit_ccg_95,
                "p_value_dcd": p_val_dcd,
                "p_value_ccg": p_val_ccg,
                "surrogate_dcd_mean": float(np.mean(surr_dcd_means)),
                "surrogate_dcd_min": float(np.min(surr_dcd_means)),
                "surrogate_dcd_max": float(np.max(surr_dcd_means)),
                "surrogate_ccg_mean": float(np.mean(surr_ccg_means)),
                "significant": p_val_dcd < 0.05
            }
            
        results[inter.lower()] = grid_results
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSensitivity analysis complete and saved to {output_path}")
    return results


if __name__ == "__main__":
    run_sensitivity_analysis()
