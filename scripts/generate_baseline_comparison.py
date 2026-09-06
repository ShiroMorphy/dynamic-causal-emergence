#!/usr/bin/env python3
"""
Baseline Comparison Suite for Dynamic Causal Emergence.
Benchmarks Local Linear-Gaussian DCE against:
1. Sliding-Window PCA Rank (Participation Ratio of local sample covariance)
2. Sliding-Window Static CE (standard unregularized CE on sliding windows)
3. Local Linear-Gaussian DCE (Proposed Method)
"""

import json
import numpy as np
from pathlib import Path
from dce.datasets.synthetic import (
    generate_dgp_c_abrupt_emergence,
    generate_dgp_d_smooth_drift,
    generate_dgp_e_changing_dimension,
    generate_dgp_j_hierarchical_transition
)
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "results" / "synthetic" / "baseline_comparison_results.json"


def evaluate_baselines(n_reps: int = 10):
    dgps = {
        "DGP-C": (generate_dgp_c_abrupt_emergence, 1200),
        "DGP-D": (generate_dgp_d_smooth_drift, 1200),
        "DGP-E": (generate_dgp_e_changing_dimension, 2400),
        "DGP-J": (generate_dgp_j_hierarchical_transition, 2400)
    }

    results = {}

    for dgp_name, (gen_fn, n_steps) in dgps.items():
        print(f"--> Evaluating baselines on {dgp_name} (R={n_reps}, T={n_steps})...")
        dce_rmses, dce_accs = [], []
        pca_rmses, pca_accs = [], []
        static_rmses, static_accs = [], []

        for rep in range(n_reps):
            data = gen_fn(n_steps=n_steps, seed=1000 + rep)
            X = data.states
            T_trans = len(X) - 1
            w_len = 48

            # 1. Proposed DCE
            est_dce = LocalLinearGaussianDCE(bandwidth=48.0, ridge_alpha=0.01)
            est_dce.fit(X)

            # 2. Sliding-Window PCA Rank
            pca_prs = []
            pca_q90s = []
            for t in range(T_trans):
                start = max(0, t - w_len // 2)
                end = min(len(X), t + w_len // 2 + 1)
                cov = np.cov(X[start:end], rowvar=False)
                eigs = np.maximum(np.linalg.eigvalsh(cov), 0.0)
                sum_e = np.sum(eigs)
                if sum_e > 1e-12:
                    pr = float((sum_e ** 2) / np.sum(eigs ** 2))
                    cum = np.cumsum(np.sort(eigs)[::-1]) / sum_e
                    q90 = int(np.searchsorted(cum, 0.90) + 1)
                else:
                    pr = 0.0
                    q90 = 0
                pca_prs.append(pr)
                pca_q90s.append(q90)

            pca_prs = np.array(pca_prs)
            pca_q90s = np.array(pca_q90s)

            # 3. Sliding-Window Static CE (unregularized OLS)
            static_prs = []
            static_q90s = []
            for t in range(T_trans):
                start = max(0, t - w_len // 2)
                end = min(len(X) - 1, t + w_len // 2 + 1)
                x_p = X[start:end]
                x_f = X[start+1:end+1]
                try:
                    A_hat = np.linalg.lstsq(x_p, x_f, rcond=None)[0].T
                    res = x_f - (A_hat @ x_p.T).T
                    Sig_hat = np.cov(res, rowvar=False) + 1e-4 * np.eye(X.shape[1])
                    import scipy.linalg
                    L = np.linalg.cholesky(Sig_hat)
                    M = scipy.linalg.solve_triangular(L, A_hat, lower=True, check_finite=False)
                    s = scipy.linalg.svdvals(M)
                    lambdas = s ** 2
                    e = 0.5 * np.log1p(lambdas)
                    sum_e = np.sum(e)
                    pr = float((sum_e ** 2) / np.sum(e ** 2)) if sum_e > 1e-12 else 0.0
                    cum = np.cumsum(e) / max(sum_e, 1e-12)
                    q90 = int(np.searchsorted(cum, 0.90) + 1)
                except Exception:
                    pr = 0.0
                    q90 = 0
                static_prs.append(pr)
                static_q90s.append(q90)

            static_prs = np.array(static_prs)
            static_q90s = np.array(static_q90s)

            if data.true_dcd_pr is not None:
                dce_rmses.append(float(np.sqrt(np.mean((est_dce.dcd_pr_ - data.true_dcd_pr) ** 2))))
                pca_rmses.append(float(np.sqrt(np.mean((pca_prs - data.true_dcd_pr) ** 2))))
                static_rmses.append(float(np.sqrt(np.mean((static_prs - data.true_dcd_pr) ** 2))))

            if data.true_q90 is not None:
                dce_accs.append(float(np.mean(est_dce.q90_ == data.true_q90)))
                pca_accs.append(float(np.mean(pca_q90s == data.true_q90)))
                static_accs.append(float(np.mean(static_q90s == data.true_q90)))

        results[dgp_name] = {
            "dce": {
                "rmse": float(np.mean(dce_rmses)) if dce_rmses else None,
                "acc": float(np.mean(dce_accs)) if dce_accs else None
            },
            "sliding_pca": {
                "rmse": float(np.mean(pca_rmses)) if pca_rmses else None,
                "acc": float(np.mean(pca_accs)) if pca_accs else None
            },
            "sliding_static_ce": {
                "rmse": float(np.mean(static_rmses)) if static_rmses else None,
                "acc": float(np.mean(static_accs)) if static_accs else None
            }
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline results to {OUTPUT_PATH}")
    return results


def format_latex_baseline_table():
    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{\textbf{Comparative Baseline Benchmark on Dynamic Scale Transition Systems.} Root-Mean-Square Error (RMSE) and exact $q_{90}$ dimensional recovery accuracy comparing the proposed Local Linear-Gaussian DCE against Sliding-Window PCA Rank and Sliding-Window Static Causal Emergence across dynamic benchmarks.}")
    lines.append(r"\label{tab:baseline_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Benchmark} & \multicolumn{2}{c}{\textbf{Sliding PCA}} & \multicolumn{2}{c}{\textbf{Static CE (Window)}} & \multicolumn{2}{c}{\textbf{Proposed DCD}} \\")
    lines.append(r" & \textbf{RMSE} & \textbf{Acc.} & \textbf{RMSE} & \textbf{Acc.} & \textbf{RMSE} & \textbf{Acc.} \\")
    lines.append(r"\midrule")

    for dgp_name, res in results.items():
        pca = res["sliding_pca"]
        st = res["sliding_static_ce"]
        dce = res["dce"]
        
        p_r = pca.get("rmse")
        p_a = pca.get("acc")
        s_r = st.get("rmse")
        s_a = st.get("acc")
        d_r = dce.get("rmse")
        d_a = dce.get("acc")
        
        pca_rmse = f"{p_r:.3f}" if p_r is not None else "--"
        pca_acc = f"{p_a * 100:.1f}\\%" if p_a is not None else "--"
        st_rmse = f"{s_r:.3f}" if s_r is not None else "--"
        st_acc = f"{s_a * 100:.1f}\\%" if s_a is not None else "--"
        dce_rmse = f"\\textbf{{{d_r:.3f}}}" if d_r is not None else "--"
        dce_acc = f"\\textbf{{{d_a * 100:.1f}\\%}}" if d_a is not None else "--"

        lines.append(f"{dgp_name} & {pca_rmse} & {pca_acc} & {st_rmse} & {st_acc} & {dce_rmse} & {dce_acc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    evaluate_baselines(n_reps=10)
    print("\n" + format_latex_baseline_table())
