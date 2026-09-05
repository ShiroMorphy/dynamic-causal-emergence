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
from typing import Any, Dict, List, Optional
import numpy as np

from dce.datasets.synthetic import get_synthetic_benchmark
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE


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
    dce_hat = model.emergence_
    q_hat = model.causal_dimension_
    
    T_eval = len(dce_hat)
    dce_true = data.true_dce[:T_eval]
    q_true = data.true_optimal_dim[:T_eval]
    
    # 1. Estimation Metrics
    error = dce_hat - dce_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_bias = float(np.mean(error))
    max_dce = float(np.max(dce_hat))
    
    # 2. Causal Dimension Accuracy
    dim_accuracy = float(np.mean(q_hat == q_true))
    
    # 3. Detection Delay & Detection Success (for transition DGPs)
    detection_delay = None
    detection_success = None
    is_null_dgp = ("null" in dgp_name.lower()) or ("shock" in dgp_name.lower())
    
    if data.transition_timestamp is not None and isinstance(data.transition_timestamp, int):
        tau = data.transition_timestamp
        detected_indices = np.where((dce_hat > threshold) & (np.arange(T_eval) >= tau - 20))[0]
        if len(detected_indices) > 0:
            tau_hat = int(detected_indices[0])
            detection_delay = float(tau_hat - tau)
            detection_success = True
        else:
            detection_success = False
            
    # 4. False Positive / Exceedance
    if is_null_dgp:
        is_false_alarm = bool(max_dce > threshold)
    else:
        # For emergence DGPs, false alarm means detecting emergence well before transition
        if data.transition_timestamp is not None and isinstance(data.transition_timestamp, int):
            tau = data.transition_timestamp
            pre_trans = dce_hat[:max(0, tau - 30)]
            is_false_alarm = bool(len(pre_trans) > 0 and np.max(pre_trans) > threshold)
        else:
            is_false_alarm = False
            
    return {
        "seed": seed,
        "rmse": rmse,
        "mean_bias": mean_bias,
        "max_dce": max_dce,
        "dim_accuracy": dim_accuracy,
        "detection_delay": detection_delay,
        "detection_success": detection_success,
        "is_false_alarm": is_false_alarm
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
    base_seed: int = 1000
) -> Dict[str, Any]:
    """Run full Monte Carlo experiment across n_reps independent random seeds."""
    reps_data = []
    for i in range(n_reps):
        seed = base_seed + i
        res = run_mc_replication(
            dgp_name=dgp_name,
            seed=seed,
            estimator_name=estimator_name,
            macro_dims=macro_dims,
            bandwidth=bandwidth,
            causal_only=causal_only,
            n_steps=n_steps,
            threshold=threshold
        )
        reps_data.append(res)
        
    rmse_vals = [r["rmse"] for r in reps_data]
    bias_vals = [r["mean_bias"] for r in reps_data]
    acc_vals = [r["dim_accuracy"] for r in reps_data]
    fa_vals = [1 if r["is_false_alarm"] else 0 for r in reps_data]
    delays = [r["detection_delay"] for r in reps_data if r["detection_delay"] is not None]
    
    summary = {
        "dgp": dgp_name,
        "n_reps": n_reps,
        "estimator": estimator_name,
        "bandwidth": bandwidth,
        "causal_only": causal_only,
        "threshold": threshold,
        "fpr": float(np.mean(fa_vals)),
        "mean_rmse": float(np.mean(rmse_vals)),
        "std_rmse": float(np.std(rmse_vals)),
        "mean_bias": float(np.mean(bias_vals)),
        "mean_dim_accuracy": float(np.mean(acc_vals)),
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "median_detection_delay": float(np.median(delays)) if delays else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Benchmark Harness for Dynamic Causal Emergence.")
    parser.add_argument("--dgp", type=str, default="dgp_b", help="DGP identifier (e.g. dgp_a, dgp_b, dgp_c, dgp_e)")
    parser.add_argument("--estimator", type=str, default="linear_gaussian", help="Estimator to benchmark (linear_gaussian, dyn_nis, static_nis_windowed)")
    parser.add_argument("--reps", type=int, default=50, help="Number of Monte Carlo replications")
    parser.add_argument("--n-steps", type=int, default=500, help="Trajectory length T")
    parser.add_argument("--bandwidth", type=float, default=24.0, help="Temporal kernel bandwidth")
    parser.add_argument("--causal", action="store_true", help="Use one-sided causal temporal weights")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON output summary")
    args = parser.parse_args()
    
    summary = run_monte_carlo_suite(
        dgp_name=args.dgp,
        n_reps=args.reps,
        estimator_name=args.estimator,
        bandwidth=args.bandwidth,
        causal_only=args.causal,
        n_steps=args.n_steps
    )

    
    print(json.dumps(summary, indent=2))
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
