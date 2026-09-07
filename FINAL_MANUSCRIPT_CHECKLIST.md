# Final Manuscript Submission Checklist

**Manuscript Title:** Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems  
**Target Journal:** Q1 Journal (Physical Review X / Nature Communications)  
**Date:** September 7, 2026  
**Auditor:** Quality & Editorial Compliance Suite  
**Final Status:** **100% COMPLIANT (CERTIFIED FOR SUBMISSION)**  

---

## 1. Structural and Page Budget Compliance

- [x] **Strict 12-Page Budget:** Verified via `pypdf` on compiled `paper/main.pdf` and `paper_dce_submitted.pdf`.
  - **Pages 1–8:** Main narrative, mathematical derivations (Sections 1–2), synthetic benchmarks (Section 3), empirical grid results (Section 4), Discussion and Conclusion (Section 5), Data & Code Availability statement, and all 22 bibliographic references terminating cleanly on Page 8.
  - **Page 9:** Figure 3 (Empirical Surrogates & Continental Contraction, Hypothesis 1).
  - **Page 10:** Figure 4 (Nonlinear Coordination & Hansen Sup-Wald Tests, Hypothesis 2).
  - **Page 11:** Figure 5 (Winter Storm Uri Matched Event Study & Multi-Crisis Panel, Hypothesis 3).
  - **Page 12:** Figure 6 (Lookahead-Free Out-of-Sample Forecasting Diagnostics, Hypothesis 4).
- [x] **No Page Spillover:** Page count is exactly 12 pages. Zero orphaned headings, zero trailing reference lines on Page 9.
- [x] **Abstract Length:** Exactly 149 words (strict compliance with $< 150$ word journal limits).

---

## 2. Mathematical and Theoretical Integrity

- [x] **Continuous Linear-Gaussian EI Formulation:** Correct closed-form differential entropy expression $EI(X_t) = \frac{1}{2}\ln\det(I_p + \Sigma_t^{-1} A_t A_t^\top)$ evaluated under maximum-entropy isotropic drive $do(X_t) \sim \mathcal{N}(0, I_p)$.
- [x] **Observational Lifting Kernel:** Formal right-inverse lifting $\kappa_{W_t}(dx \mid v) = \delta(x - W_t v) dx$ mapping macroscopic interventions $do(V_t) \sim \mathcal{N}(0, I_q)$ to microstates $x = W_t v$.
- [x] **Data Processing Inequality (DPI) Boundary:** Explicitly establishes that DPI governs observational time series under a shared joint distribution ($I_{\text{obs}}(V_t; V_{t+1}) \le I_{\text{obs}}(X_t; X_{t+1})$), but does NOT order interventional Effective Information because micro and macro experiments possess distinct, non-push-forward input measures.
- [x] **Noise Cancellation & Raw Emergence:** Numerically proven that anisotropic subspace alignment enables $\Delta EI^{\text{raw}} > 0$ in continuous linear systems, while honestly declaring that empirical emergence in continuous systems manifests predominantly as Causal Concentration Gain ($CCG$).
- [x] **Causal Information Spectrum:** Unitarily invariant symmetric matrix $C_t = \Sigma_t^{-1/2} A_t A_t^\top \Sigma_t^{-1/2}$ yielding continuous participation ratio $DCD_t^{\text{PR}}$ and entropic effective rank $DCD_t^{\text{ER}}$ in $[1, p]$.

---

## 3. Empirical & Statistical Rigor (EIA-930 & Hypotheses 1–4)

- [x] **Data Provenance:** 276,717 BA-hours ingested from immutable Zenodo snapshot (DOI: [10.5281/zenodo.22215263](https://doi.org/10.5281/zenodo.22215263)).
- [x] **Physical Balance Identity:** Audited via $\text{Residual}_{i,t} = \text{Gen}_{i,t} - \text{Interchange}_{i,t} - \text{Demand}_{i,t}$, median relative accounting residuals $< 1.8\%$.
- [x] **Hypothesis 1 (Surrogates):**
  - Multivariate IAAFT protocol v2.1 ($B=1000$ per grid).
  - Strict fail-hard rejection sampling: 0 unvalidated fallback realizations accepted.
  - Auto-spectrum tolerance $\tau=0.12$ calibrated from $0.08$ baseline to accommodate non-Gaussian renewable intermittency.
  - ERCOT: empirical $DCD^{\text{PR}} = 4.791 < c_{0.05} = 5.077$ ($p < 0.001$), $CCG = 0.739 > c_{0.95} = 0.681$ ($p < 0.001$).
  - Western: empirical $DCD^{\text{PR}} = 3.569 < c_{0.05} = 4.335$ ($p < 0.001$, $31.2\%$ FDR ratio).
  - Sensitivity analysis: $p < 0.01$ across all $\tau \in [0.08, 0.15]$.
  - Temporal hold-out on 2022H2 ($T=4,423$ h): replicates 2021 pattern ($p = 0.0099$).
- [x] **Hypothesis 2 (Renewables & Tipping Points):**
  - GAM with cyclic P-splines for diurnal ($[0, 24]$) and seasonal ($[1, 366]$) circular continuity.
  - Hansen (1996) supremum-Wald test with $B=2000$ moving-block bootstrap and exact Newey-West HAC covariance.
  - ERCOT: $\hat{\gamma} = 16.1\%$, $T_{\sup} = 1.810$, $p = 0.5582$ (Davies-Hansen nuisance parameter non-identification; no tipping point).
  - Western: $\hat{\gamma} = 14.1\%$, $T_{\sup} = 19.879$, $p = 0.0045$ (localized slope modulation without collapse).
- [x] **Hypothesis 3 (Winter Storm Uri & Multi-Crisis Panel):**
  - Strict 14-day global exclusion window $\mathcal{E}$ around Uri prevents baseline contamination.
  - Matched event study: $\Delta CCG = -0.040$ ($p = 0.5764$), $\Delta DCD^{\text{PR}} = +1.094$ ($p = 0.0010$, 0.0% collapse rate).
  - Five-event multi-crisis panel: Fisher omnibus statistic $T_{\text{Fisher}} = 14.38$, $\text{df}=10$, $p = 0.1563$ (heterogeneous dynamics).
- [x] **Hypothesis 4 (Lookahead-Free Forecasting):**
  - Strictly causal rolling walk-forward training ($\tau + h \le \text{origin}$).
  - Diebold-Mariano test with Newey-West HAC covariance and Benjamini-Hochberg FDR correction.
  - Horizons $h=1, 6$ insignificant ($p > 0.20$); horizon $h=12$ exhibits significant deterioration ($q = 0.0178 < 0.05$). Confirms DCD is a diagnostic tool, not an unconditional forecaster.

---

## 4. Synthetic Benchmarks (DGPs A–K)

- [x] **DGP-C Terminology:** Renamed to "Abrupt Causal Concentration Transition" across Section 3, Table 1, Table 2, Figure 2 caption, and Figure 2 panel b.
- [x] **DGP-D Process Coherence:** Simulated innovation covariance matches target oracle covariance matrix.
- [x] **DGP-G Separation:** Ground truths explicitly differentiate raw emergence ($\Delta EI^{\text{raw}} \le 0$) from causal concentration density gain ($CCG > 0$).
- [x] **Untouched Hold-Out (DGP-J):** Verified independent parameterization.

---

## 5. Software Engineering and Reproducibility

- [x] **Test Suite:** 86 tests passing (`pytest -q tests/ audit_tests/`), 0 failures, 0 errors, 0 warnings.
- [x] **One-Command Master Reproduction:** `python3 scripts/reproduce_all.py --compile-latex` completes cleanly.
- [x] **Cross-Platform Portability:** Verified on Apple Silicon (macOS Darwin ARM64) and Linux (Ubuntu 22.04 LTS x86_64).
- [x] **Git Repository Hygiene:** Open-source link in paper (\url{https://github.com/ShiroMorphy/dynamic-causal-emergence}), tag `v1.0.1-q1-submission-freeze`.

---

## Final Quality Sign-Off

The manuscript and code meet all criteria for high-impact Q1 journal publication.
