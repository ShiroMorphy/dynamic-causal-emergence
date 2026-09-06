#!/usr/bin/env python3
"""
Baseline Comparison Suite for Dynamic Causal Emergence.
Benchmarks Local Linear-Gaussian DCE against:
1. Sliding-Window PCA Rank (Participation Ratio of local sample covariance)
2. Sliding-Window Unregularized Causal Spectrum (unregularized OLS causal spectrum)
3. Proposed DCD (Spectral-Regularized Local Linear-Gaussian)

Information budget is strictly equalized:
- Gaussian kernel bandwidth h = 36.0 yields N_eff = 2 * sqrt(pi) * 36 approx 128
- Sliding window baselines use window length w_len = 128.
- Monte Carlo replications: R = 100 per benchmark.
"""

import os
import json
import numpy as np
import scipy.linalg
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dce.datasets.synthetic import (
    generate_dgp_c_abrupt_emergence,
    generate_dgp_d_smooth_drift,
    generate_dgp_e_changing_dimension,
    generate_dgp_j_hierarchical_transition,
    generate_dgp_k_holdout_transition,
)
from dce.estimators.linear_gaussian import LocalLinearGaussianDCE

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "results" / "synthetic" / "baseline_comparison_results.json"


def evaluate_single_rep(args):
    dgp_name, gen_fn, n_steps, seed, w_len, bandwidth = args
    data = gen_fn(n_steps=n_steps, seed=seed)
    X = data.states
    T_trans = len(X) - 1

    # 1. Proposed DCD (Spectral-Regularized)
    est_dce = LocalLinearGaussianDCE(bandwidth=bandwidth, ridge_alpha=0.01, macro_dims=[])
    est_dce.fit(X)

    # 2. Sliding-Window PCA Rank (w_len = 128)
    pca_prs = np.zeros(T_trans)
    pca_q90s = np.zeros(T_trans, dtype=int)
    for t in range(T_trans):
        start = max(0, t - w_len // 2)
        end = min(len(X), t + w_len // 2 + 1)
        cov = np.cov(X[start:end], rowvar=False)
        eigs = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        sum_e = np.sum(eigs)
        if sum_e > 1e-12:
            pca_prs[t] = float((sum_e ** 2) / np.sum(eigs ** 2))
            cum = np.cumsum(np.sort(eigs)[::-1]) / sum_e
            pca_q90s[t] = int(np.searchsorted(cum, 0.90) + 1)

    # 3. Sliding-Window Unregularized Causal Spectrum (w_len = 128)
    static_prs = np.zeros(T_trans)
    static_q90s = np.zeros(T_trans, dtype=int)
    for t in range(T_trans):
        start = max(0, t - w_len // 2)
        end = min(len(X) - 1, t + w_len // 2 + 1)
        x_p = X[start:end]
        x_f = X[start + 1 : end + 1]
        try:
            A_hat = np.linalg.lstsq(x_p, x_f, rcond=None)[0].T
            res = x_f - (A_hat @ x_p.T).T
            Sig_hat = np.cov(res, rowvar=False) + 1e-4 * np.eye(X.shape[1])
            L = np.linalg.cholesky(Sig_hat)
            M = scipy.linalg.solve_triangular(L, A_hat, lower=True, check_finite=False)
            s = scipy.linalg.svdvals(M)
            lambdas = s ** 2
            e = 0.5 * np.log1p(lambdas)
            sum_e = np.sum(e)
            if sum_e > 1e-12:
                static_prs[t] = float((sum_e ** 2) / np.sum(e ** 2))
                cum = np.cumsum(e) / sum_e
                static_q90s[t] = int(np.searchsorted(cum, 0.90) + 1)
        except Exception:
            pass

    return {
        "dce_rmse": float(np.sqrt(np.mean((est_dce.dcd_pr_ - data.true_dcd_pr) ** 2))),
        "dce_acc": float(np.mean(est_dce.q90_ == data.true_q90)),
        "pca_rmse": float(np.sqrt(np.mean((pca_prs - data.true_dcd_pr) ** 2))),
        "pca_acc": float(np.mean(pca_q90s == data.true_q90)),
        "static_rmse": float(np.sqrt(np.mean((static_prs - data.true_dcd_pr) ** 2))),
        "static_acc": float(np.mean(static_q90s == data.true_q90)),
    }


def evaluate_baselines(n_reps: int = 100, w_len: int = 128, bandwidth: float = 36.0):
    dgps = {
        "DGP-C": (generate_dgp_c_abrupt_emergence, 1200),
        "DGP-D": (generate_dgp_d_smooth_drift, 1200),
        "DGP-E": (generate_dgp_e_changing_dimension, 2400),
        "DGP-J": (generate_dgp_j_hierarchical_transition, 2400),
        "DGP-K": (generate_dgp_k_holdout_transition, 2400),
    }

    results = {}
    n_workers = min(10, os.cpu_count() or 4)

    for dgp_name, (gen_fn, n_steps) in dgps.items():
        print(f"--> Evaluating baselines on {dgp_name} (R={n_reps}, T={n_steps}, w_len={w_len}, h={bandwidth}, workers={n_workers})...")
        tasks = [
            (dgp_name, gen_fn, n_steps, 1000 + rep, w_len, bandwidth)
            for rep in range(n_reps)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            rep_results = list(executor.map(evaluate_single_rep, tasks))

        dce_rmses = [r["dce_rmse"] for r in rep_results]
        dce_accs = [r["dce_acc"] for r in rep_results]
        pca_rmses = [r["pca_rmse"] for r in rep_results]
        pca_accs = [r["pca_acc"] for r in rep_results]
        static_rmses = [r["static_rmse"] for r in rep_results]
        static_accs = [r["static_acc"] for r in rep_results]

        results[dgp_name] = {
            "dce": {
                "rmse": float(np.mean(dce_rmses)),
                "rmse_std": float(np.std(dce_rmses)),
                "acc": float(np.mean(dce_accs)),
                "acc_std": float(np.std(dce_accs)),
            },
            "sliding_pca": {
                "rmse": float(np.mean(pca_rmses)),
                "rmse_std": float(np.std(pca_rmses)),
                "acc": float(np.mean(pca_accs)),
                "acc_std": float(np.std(pca_accs)),
            },
            "sliding_unreg_spectrum": {
                "rmse": float(np.mean(static_rmses)),
                "rmse_std": float(np.std(static_rmses)),
                "acc": float(np.mean(static_accs)),
                "acc_std": float(np.std(static_accs)),
            },
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline results to {OUTPUT_PATH}")
    return results


def format_latex_baseline_table(results=None):
    if results is None:
        with open(OUTPUT_PATH, "r") as f:
            results = json.load(f)

    b = "\\"
    descriptions = {
        "DGP-C": f"DGP-C ($8 {b}to 2$)",
        "DGP-D": f"DGP-D ($8 {b}to 2$)",
        "DGP-E": f"DGP-E ($16 {b}to 8,4,2$)",
        "DGP-J": f"DGP-J ($12 {b}to 6,3,2$)",
        "DGP-K": f"DGP-K (Hold-out $10 {b}to 5,2,1$)",
    }

    lines = [
        f"{b}begin{{table}}[t]",
        f"{b}centering",
        f"{b}footnotesize",
        f"{b}setlength{{{b}tabcolsep}}{{3.2pt}}",
        f"{b}caption{{{b}textbf{{Comparative Baseline Benchmark on Dynamic Scale Transition Systems ($R=100$).}} Root-Mean-Square Error (RMSE) and exact $q_{{90}}$ dimensional recovery accuracy comparing Proposed DCD (Spectral-Regularized, Gaussian $h=36$, $N_{{{b}text{{eff}}}} {b}approx 128$) against Sliding-Window PCA Rank ($w_{{{b}text{{len}}}}=128$) and Sliding-Window Unregularized Causal Spectrum ($w_{{{b}text{{len}}}}=128$) under strictly equalized information budgets. DGPs C, D, E, J represent derivation benchmarks; DGP-K is a frozen out-of-sample hold-out benchmark ($p=10$, $5 {b}to 2 {b}to 1$).}}",
        f"{b}label{{tab:baseline_comparison}}",
        f"{b}resizebox{{{b}columnwidth}}{{!}}{{%",
        f"{b}begin{{tabular}}{{lcccccc}}",
        f"{b}toprule",
        f"{b}textbf{{Benchmark}} & {b}multicolumn{{2}}{{c}}{{{b}textbf{{Sliding PCA}}}} & {b}multicolumn{{2}}{{c}}{{{b}textbf{{Sliding Spectrum}}}} & {b}multicolumn{{2}}{{c}}{{{b}textbf{{Proposed DCD}}}} {b}{b}",
        f" & {b}textbf{{RMSE}} & {b}textbf{{Acc.}} & {b}textbf{{RMSE}} & {b}textbf{{Acc.}} & {b}textbf{{RMSE}} & {b}textbf{{Acc.}} {b}{b}",
        f"{b}midrule",
    ]

    for dgp_name, res in results.items():
        pca = res["sliding_pca"]
        st = res.get("sliding_unreg_spectrum", res.get("sliding_static_ce", {}))
        dce = res["dce"]
        label = descriptions.get(dgp_name, dgp_name)
        p_r = pca.get("rmse")
        p_a = pca.get("acc")
        s_r = st.get("rmse")
        s_a = st.get("acc")
        d_r = dce.get("rmse")
        d_a = dce.get("acc")

        rmses = [p_r, s_r, d_r]
        valid_rmses = [x for x in rmses if x is not None]
        min_rmse = min(valid_rmses) if valid_rmses else None

        accs = [p_a, s_a, d_a]
        valid_accs = [x for x in accs if x is not None]
        max_acc = max(valid_accs) if valid_accs else None

        def fmt_r(val):
            if val is None:
                return "--"
            s = f"{val:.3f}"
            return f"{b}textbf{{{s}}}" if min_rmse is not None and abs(val - min_rmse) < 1e-6 else s

        def fmt_a(val):
            if val is None:
                return "--"
            s = f"{val * 100:.1f}\\%"
            return f"{b}textbf{{{s}}}" if max_acc is not None and abs(val - max_acc) < 1e-6 else s

        pca_rmse = fmt_r(p_r)
        pca_acc = fmt_a(p_a)
        st_rmse = fmt_r(s_r)
        st_acc = fmt_a(s_a)
        dce_rmse = fmt_r(d_r)
        dce_acc = fmt_a(d_a)

        lines.append(f"{label} & {pca_rmse} & {pca_acc} & {st_rmse} & {st_acc} & {dce_rmse} & {dce_acc} {b}{b}")

    lines.extend([
        f"{b}bottomrule",
        f"{b}end{{tabular}}%",
        "}",
        f"{b}end{{table}}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    res = evaluate_baselines(n_reps=100, w_len=128, bandwidth=36.0)
    print("\n" + format_latex_baseline_table(res))
