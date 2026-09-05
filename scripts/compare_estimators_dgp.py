"""
Comparative Benchmark Script: Dyn-NIS+ vs Static Windowed NIS+ vs Linear-Gaussian DCE.

Verifies Claim C02:
Dyn-NIS+ with representation-space Procrustes temporal regularization and warm-starting
improves tracking stability, reduces chattering variance Var(DCE_t - DCE_{t-1}),
and detects dynamic causal transitions with calibrated latency.
"""

import json
import os
import numpy as np

from dce.datasets.synthetic import generate_dgp_c_abrupt_emergence
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE
from dce.estimators.dyn_nis import DynamicNIS
from dce.estimators.static_nis_plus import StaticNISPlusWindowed


def evaluate_stability_and_tracking(dce_series: np.ndarray, true_dce: np.ndarray) -> dict:
    diffs = np.diff(dce_series)
    chattering_variance = float(np.var(diffs))
    rmse = float(np.sqrt(np.mean((dce_series - true_dce) ** 2)))
    mean_val = float(np.mean(dce_series))
    return {
        "chattering_variance": chattering_variance,
        "rmse": rmse,
        "mean": mean_val
    }


def main():
    print("Running comparative benchmark on DGP-C (Abrupt Emergence)...")
    seed = 42
    n_steps = 300
    tau = 150
    bandwidth = 20.0
    macro_dims = [2, 4, 8]
    eval_step = 5
    
    data = generate_dgp_c_abrupt_emergence(n_steps=n_steps, transition_t=tau, p_dim=8, q_dim=2, seed=seed)
    T_trans = n_steps - 1
    true_dce = data.true_dce[:T_trans]
    true_q = data.true_optimal_dim[:T_trans]
    
    # 1. Local Linear Gaussian Reference
    print("1. Fitting LocalLinearGaussianDCE...")
    model_linear = LocalLinearGaussianDCE(macro_dims=macro_dims, bandwidth=bandwidth)
    model_linear.fit(data.states)
    res_linear = evaluate_stability_and_tracking(model_linear.normalized_emergence_, true_dce)
    acc_linear = float(np.mean(model_linear.causal_dimension_[tau:] == 2))
    res_linear["post_transition_q_accuracy"] = acc_linear
    
    # 2. Dynamic NIS+ (With Procrustes Alignment & Warm Start)
    print("2. Fitting DynamicNIS (Dyn-NIS+)...")
    model_dyn = DynamicNIS(
        macro_dims=macro_dims,
        bandwidth=bandwidth,
        steps_per_window=5,
        eval_step=eval_step,
        hidden_dim=32,
        eta_phi=0.5,
        eta_f=0.2
    )
    model_dyn.fit(data.states)
    res_dyn = evaluate_stability_and_tracking(model_dyn.normalized_emergence_, true_dce)
    acc_dyn = float(np.mean(model_dyn.causal_dimension_[tau:] == 2))
    res_dyn["post_transition_q_accuracy"] = acc_dyn
    
    # 3. Static Windowed NIS+ (Cold Start, No Temporal Alignment)
    print("3. Fitting StaticNISPlusWindowed...")
    model_static = StaticNISPlusWindowed(
        macro_dims=macro_dims,
        bandwidth=bandwidth,
        steps_per_window=5,
        eval_step=eval_step,
        hidden_dim=32
    )
    model_static.fit(data.states)
    res_static = evaluate_stability_and_tracking(model_static.normalized_emergence_, true_dce)
    acc_static = float(np.mean(model_static.causal_dimension_[tau:] == 2))
    res_static["post_transition_q_accuracy"] = acc_static
    
    comparison = {
        "dgp": "dgp_c_abrupt_emergence",
        "n_steps": n_steps,
        "transition_tau": tau,
        "bandwidth": bandwidth,
        "results": {
            "local_linear_gaussian": res_linear,
            "dyn_nis": res_dyn,
            "static_nis_windowed": res_static
        },
        "dyn_nis_vs_static_chattering_reduction_pct": float(
            (res_static["chattering_variance"] - res_dyn["chattering_variance"]) / res_static["chattering_variance"] * 100
        ) if res_static["chattering_variance"] > 0 else 0.0
    }
    
    print("\n--- Benchmark Summary ---")
    print(json.dumps(comparison, indent=2))
    
    out_path = "results/synthetic/mc_delay_comparison.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison artifact to {out_path}")


if __name__ == "__main__":
    main()
