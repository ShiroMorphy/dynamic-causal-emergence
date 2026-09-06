"""
Monte Carlo Calibration and Falsification Harness for Dynamic Causal Emergence (Milestone M3).

Executes rigorous multi-seed Monte Carlo simulations across DGPs A-I to compute:
- Empirical False Positive Rate (FPR) under null stationary (DGP-A) and null nonstationary (DGP-B)
- Volatility shock robustness (DGP-F)
- Tracking RMSE and bias on smooth drift (DGP-D)
- Transition detection delay on abrupt emergence (DGP-C)
- Dynamic causal dimensionality recovery accuracy P(qhat = q_true) (DGP-E)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import beta

from dce.datasets.synthetic import SYNTHETIC_DGP_REGISTRY, get_synthetic_benchmark
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def run_mc_replication(
    dgp_name: str,
    seed: int,
    estimator_name: str = "linear_gaussian",
    macro_dims: Optional[List[int]] = None,
    bandwidth: float = 24.0,
    causal_only: bool = False,
    n_steps: int = 600,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """Execute a single Monte Carlo evaluation on given DGP with fixed seed."""
    macro_dims = macro_dims or [1, 2, 4, 8]
    data = get_synthetic_benchmark(dgp_name, n_steps=n_steps, seed=seed)
    
    # Instantiate estimator
    if estimator_name == "linear_gaussian":
        model = LocalLinearGaussianDCE(
            macro_dims=macro_dims,
            bandwidth=bandwidth,
            causal_only=causal_only
        )
    elif estimator_name in ("dyn_nis", "dynamic_nis"):
        from dce.estimators.dyn_nis import DynamicNIS
        model = DynamicNIS(
            macro_dims=macro_dims,
            bandwidth=bandwidth,
            causal_only=causal_only,
            steps_per_window=6,
            eval_step=max(1, int(bandwidth // 4)),
            hidden_dim=32,
        )
    elif estimator_name in ("static_nis", "static_nis_windowed"):
        from dce.estimators.static_nis_plus import StaticNISPlusWindowed
        model = StaticNISPlusWindowed(
            macro_dims=macro_dims,
            bandwidth=bandwidth,
            causal_only=causal_only,
            steps_per_window=6,
            eval_step=max(1, int(bandwidth // 4)),
            hidden_dim=32,
        )
    else:
        raise ValueError(f"Estimator '{estimator_name}' not supported in MC harness.")

    model.fit(data.states)
    
    # Extract dual metrics
    if hasattr(model, "optimal_dce_raw_") and hasattr(model, "optimal_dce_density_"):
        dce_hat_raw = model.optimal_dce_raw_
        dce_hat_density = model.optimal_dce_density_
        q_hat_raw = model.causal_dimension_raw_
        q_hat_density = model.causal_dimension_density_
    else:
        dce_hat_raw = model.emergence_
        dce_hat_density = getattr(model, "normalized_emergence_", model.emergence_)
        q_hat_raw = model.causal_dimension_
        q_hat_density = model.causal_dimension_

    T_eval = len(dce_hat_raw)
    dce_true_raw = data.true_dce_raw[:T_eval] if data.true_dce_raw is not None else np.zeros(T_eval)
    dce_true_density = data.true_dce_density[:T_eval] if data.true_dce_density is not None else data.true_dce[:T_eval]
    q_true = data.true_optimal_dim[:T_eval]
    
    # 1. Estimation Metrics - Raw
    error_raw = dce_hat_raw - dce_true_raw
    rmse_raw = float(np.sqrt(np.mean(error_raw ** 2)))
    mean_bias_raw = float(np.mean(error_raw))
    max_dce_raw = float(np.max(dce_hat_raw))
    
    # Estimation Metrics - Density
    error_density = dce_hat_density - dce_true_density
    rmse_density = float(np.sqrt(np.mean(error_density ** 2)))
    mean_bias_density = float(np.mean(error_density))
    max_dce_density = float(np.max(dce_hat_density))
    
    # 2. Causal Dimension Accuracy
    dim_accuracy_raw = float(np.mean(q_hat_raw == q_true))
    dim_accuracy_density = float(np.mean(q_hat_density == q_true))
    
    # 2b. Dynamic Causal Dimensionality Metrics
    dcd_pr_hat = getattr(model, "dcd_pr_", None)
    dcd_entropy_hat = getattr(model, "dcd_entropy_", None)
    q90_hat = getattr(model, "q90_", None)

    if dcd_pr_hat is not None and getattr(data, "true_dcd_pr", None) is not None:
        err_dcd_pr = dcd_pr_hat[:T_eval] - data.true_dcd_pr[:T_eval]
        rmse_dcd_pr = float(np.sqrt(np.mean(err_dcd_pr ** 2)))
        mean_bias_dcd_pr = float(np.mean(err_dcd_pr))
    else:
        rmse_dcd_pr = None
        mean_bias_dcd_pr = None

    if dcd_entropy_hat is not None and getattr(data, "true_dcd_entropy", None) is not None:
        err_dcd_entropy = dcd_entropy_hat[:T_eval] - data.true_dcd_entropy[:T_eval]
        rmse_dcd_entropy = float(np.sqrt(np.mean(err_dcd_entropy ** 2)))
        mean_bias_dcd_entropy = float(np.mean(err_dcd_entropy))
    else:
        rmse_dcd_entropy = None
        mean_bias_dcd_entropy = None

    if q90_hat is not None and getattr(data, "true_q90", None) is not None:
        dim_accuracy_q90 = float(np.mean(q90_hat[:T_eval] == data.true_q90[:T_eval]))
    else:
        dim_accuracy_q90 = None
    
    # 3. Detection Delay & Detection Success (for transition DGPs)
    detection_delay_density = None
    detection_success_density = None
    is_null_dgp = ("null" in dgp_name.lower()) or ("shock" in dgp_name.lower())
    
    if data.transition_timestamp is not None and isinstance(data.transition_timestamp, int):
        tau = data.transition_timestamp
        detected_indices = np.where((dce_hat_density > threshold) & (np.arange(T_eval) >= tau - 20))[0]
        if len(detected_indices) > 0:
            tau_hat = int(detected_indices[0])
            detection_delay_density = float(tau_hat - tau)
            detection_success_density = True
        else:
            detection_success_density = False
            
    # 4. False Positive / Exceedance
    if is_null_dgp:
        is_false_alarm_raw = bool(max_dce_raw > 0.0) # Raw strict emergence impossible under linear coarse-graining
        is_false_alarm_density = bool(max_dce_density > threshold)
    else:
        # For emergence DGPs, false alarm means detecting emergence well before transition
        if data.transition_timestamp is not None and isinstance(data.transition_timestamp, int):
            tau = data.transition_timestamp
            pre_trans_raw = dce_hat_raw[:max(0, tau - 30)]
            pre_trans_dens = dce_hat_density[:max(0, tau - 30)]
            is_false_alarm_raw = bool(len(pre_trans_raw) > 0 and np.max(pre_trans_raw) > 0.0)
            is_false_alarm_density = bool(len(pre_trans_dens) > 0 and np.max(pre_trans_dens) > threshold)
        else:
            is_false_alarm_raw = False
            is_false_alarm_density = False
            
    return {
        "seed": seed,
        "rmse_raw": rmse_raw,
        "mean_bias_raw": mean_bias_raw,
        "max_dce_raw": max_dce_raw,
        "rmse_density": rmse_density,
        "mean_bias_density": mean_bias_density,
        "max_dce_density": max_dce_density,
        "dim_accuracy_raw": dim_accuracy_raw,
        "dim_accuracy_density": dim_accuracy_density,
        "rmse_dcd_pr": rmse_dcd_pr,
        "mean_bias_dcd_pr": mean_bias_dcd_pr,
        "rmse_dcd_entropy": rmse_dcd_entropy,
        "mean_bias_dcd_entropy": mean_bias_dcd_entropy,
        "dim_accuracy_q90": dim_accuracy_q90,
        "detection_delay_density": detection_delay_density,
        "detection_success_density": detection_success_density,
        "is_false_alarm_raw": is_false_alarm_raw,
        "is_false_alarm_density": is_false_alarm_density,
        # Backward-compatible keys
        "rmse": rmse_density,
        "mean_bias": mean_bias_density,
        "dim_accuracy": dim_accuracy_density,
        "is_false_alarm": is_false_alarm_density
    }


def run_monte_carlo_suite(
    dgp_name: str,
    n_reps: int = 100,
    estimator_name: str = "linear_gaussian",
    macro_dims: Optional[List[int]] = None,
    bandwidth: float = 24.0,
    causal_only: bool = False,
    n_steps: int = 600,
    threshold: float = 0.05,
    base_seed: int = 1000,
    n_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Run full Monte Carlo experiment across n_reps independent random seeds in parallel."""
    from concurrent.futures import ProcessPoolExecutor
    
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
        
    seeds = [base_seed + i for i in range(n_reps)]
    
    if n_workers > 1 and n_reps > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    run_mc_replication,
                    dgp_name,
                    seed,
                    estimator_name,
                    macro_dims,
                    bandwidth,
                    causal_only,
                    n_steps,
                    threshold
                )
                for seed in seeds
            ]
            reps_data = [f.result() for f in futures]
    else:
        reps_data = [
            run_mc_replication(
                dgp_name,
                seed,
                estimator_name,
                macro_dims,
                bandwidth,
                causal_only,
                n_steps,
                threshold
            )
            for seed in seeds
        ]
        
    fa_raw = [1 if r["is_false_alarm_raw"] else 0 for r in reps_data]
    fa_dens = [1 if r["is_false_alarm_density"] else 0 for r in reps_data]
    k_raw = sum(fa_raw)
    k_dens = sum(fa_dens)
    ci_raw_low, ci_raw_high = clopper_pearson_ci(k_raw, n_reps)
    ci_dens_low, ci_dens_high = clopper_pearson_ci(k_dens, n_reps)
    
    dcd_pr_rmses = [r["rmse_dcd_pr"] for r in reps_data if r["rmse_dcd_pr"] is not None]
    dcd_pr_biases = [r["mean_bias_dcd_pr"] for r in reps_data if r["mean_bias_dcd_pr"] is not None]
    dcd_ent_rmses = [r["rmse_dcd_entropy"] for r in reps_data if r["rmse_dcd_entropy"] is not None]
    dcd_ent_biases = [r["mean_bias_dcd_entropy"] for r in reps_data if r["mean_bias_dcd_entropy"] is not None]
    q90_accs = [r["dim_accuracy_q90"] for r in reps_data if r["dim_accuracy_q90"] is not None]
    delays = [r["detection_delay_density"] for r in reps_data if r["detection_delay_density"] is not None]

    summary = {
        "dgp": dgp_name,
        "n_reps": n_reps,
        "estimator": estimator_name,
        "bandwidth": bandwidth,
        "causal_only": causal_only,
        "threshold": threshold,
        "fpr_raw": float(k_raw / n_reps),
        "fpr_raw_ci95": [ci_raw_low, ci_raw_high],
        "fpr_density": float(k_dens / n_reps),
        "fpr_density_ci95": [ci_dens_low, ci_dens_high],
        "mean_rmse_dcd_pr": float(np.mean(dcd_pr_rmses)) if dcd_pr_rmses else None,
        "std_rmse_dcd_pr": float(np.std(dcd_pr_rmses)) if dcd_pr_rmses else None,
        "mean_bias_dcd_pr": float(np.mean(dcd_pr_biases)) if dcd_pr_biases else None,
        "mean_rmse_dcd_entropy": float(np.mean(dcd_ent_rmses)) if dcd_ent_rmses else None,
        "std_rmse_dcd_entropy": float(np.std(dcd_ent_rmses)) if dcd_ent_rmses else None,
        "mean_bias_dcd_entropy": float(np.mean(dcd_ent_biases)) if dcd_ent_biases else None,
        "mean_dim_accuracy_q90": float(np.mean(q90_accs)) if q90_accs else None,
        "mean_rmse_density": float(np.mean([r["rmse_density"] for r in reps_data])),
        "std_rmse_density": float(np.std([r["rmse_density"] for r in reps_data])),
        "mean_bias_density": float(np.mean([r["mean_bias_density"] for r in reps_data])),
        "mean_rmse_raw": float(np.mean([r["rmse_raw"] for r in reps_data])),
        "mean_bias_raw": float(np.mean([r["mean_bias_raw"] for r in reps_data])),
        "mean_dim_accuracy_density": float(np.mean([r["dim_accuracy_density"] for r in reps_data])),
        "mean_dim_accuracy_raw": float(np.mean([r["dim_accuracy_raw"] for r in reps_data])),
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "median_detection_delay": float(np.median(delays)) if delays else None,
        # Backward-compatible summary fields
        "fpr": float(k_dens / n_reps),
        "mean_rmse": float(np.mean([r["rmse_density"] for r in reps_data])),
        "std_rmse": float(np.std([r["rmse_density"] for r in reps_data])),
        "mean_bias": float(np.mean([r["mean_bias_density"] for r in reps_data])),
        "mean_dim_accuracy": float(np.mean([r["dim_accuracy_density"] for r in reps_data]))
    }
    
    return {
        "summary": summary,
        "replications": reps_data
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Benchmark Harness for Dynamic Causal Emergence.")
    parser.add_argument("--dgp", type=str, default="dgp_b", help="DGP identifier (e.g. dgp_a, dgp_b, dgp_c, dgp_e)")
    parser.add_argument("--all-dgps", action="store_true", help="Run across all canonical DGPs A-I")
    parser.add_argument("--estimator", type=str, default="linear_gaussian", help="Estimator to benchmark (linear_gaussian, dyn_nis, static_nis_windowed)")
    parser.add_argument("--reps", type=int, default=100, help="Number of Monte Carlo replications")
    parser.add_argument("--n-steps", type=int, default=500, help="Trajectory length T")
    parser.add_argument("--bandwidth", type=float, default=24.0, help="Temporal kernel bandwidth")
    parser.add_argument("--causal", action="store_true", help="Use one-sided causal temporal weights")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON output summary")
    args = parser.parse_args()
    
    dgps_to_run = list(SYNTHETIC_DGP_REGISTRY.keys()) if args.all_dgps else [args.dgp]
    consolidated = {}
    
    for dgp_name in dgps_to_run:
        print(f"--> Running Monte Carlo on {dgp_name} (R={args.reps}, T={args.n_steps})...")
        full_res = run_monte_carlo_suite(
            dgp_name=dgp_name,
            n_reps=args.reps,
            estimator_name=args.estimator,
            bandwidth=args.bandwidth,
            causal_only=args.causal,
            n_steps=args.n_steps
        )
        consolidated[dgp_name] = full_res["summary"]
        
        # Save individual DGP full artifact
        dgp_out_path = f"results/synthetic/{dgp_name}_{args.estimator}.json"
        os.makedirs(os.path.dirname(os.path.abspath(dgp_out_path)), exist_ok=True)
        with open(dgp_out_path, "w") as f:
            json.dump(full_res, f, indent=2)
            
        print(f"Finished {dgp_name}: FPR_raw={full_res['summary']['fpr_raw']:.3f}, FPR_dens={full_res['summary']['fpr_density']:.3f}, RMSE={full_res['summary']['mean_rmse_density']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(consolidated, f, indent=2)
        print(f"Saved consolidated benchmark results to {args.output}")
    else:
        consolidated_path = f"results/synthetic/mc_benchmark_consolidated_{args.estimator}.json"
        os.makedirs(os.path.dirname(os.path.abspath(consolidated_path)), exist_ok=True)
        with open(consolidated_path, "w") as f:
            json.dump(consolidated, f, indent=2)
        print(f"Saved consolidated benchmark results to {consolidated_path}")


if __name__ == "__main__":
    main()
