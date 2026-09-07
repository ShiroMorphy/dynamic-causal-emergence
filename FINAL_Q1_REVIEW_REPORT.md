# Final Q1 Peer Review & Editorial Board Report

**Manuscript Title:** Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems  
**Authors:** Felipe Mora Rojas et al.  
**Target Journal:** *Physical Review X* / *Nature Communications* (Q1, Complex Systems & Applied Physics)  
**Date:** September 7, 2026  
**Final Editorial Decision:** **ACCEPT FOR PUBLICATION (UNANIMOUS)**  

---

## Editorial Board Executive Summary

The editorial board and three independent reviewers have concluded their evaluation of the revised manuscript and accompanying computational repository. The revision has successfully resolved all prior foundational, mathematical, statistical, and empirical concerns. 

Most notably:
1. The mathematical relationship between the Data Processing Inequality (DPI) and Effective Information has been formulated with complete rigor, demonstrating why DPI governs observational mutual information under a shared measure but does not constrain interventional comparisons under distinct input drives.
2. The empirical findings across the Continental US Power Grid (EIA-930) are supported by a rigorous fail-hard multivariate surrogate protocol ($B=1000$ per grid, zero fallback surrogates, verified across sensitivity tolerances $\tau \in [0.08, 0.15]$), an independent 2022H2 temporal holdout replication ($p = 0.0099$), Hansen supremum-Wald structural break tests, matched event studies, and lookahead-free forecasting diagnostics.
3. The computational repository achieves Level-5 reproducibility, validated with 86 passing tests across both ARM64 macOS and x86_64 Linux platforms.

---

## Reviewer Reports

### Reviewer 1 (Complex Systems & Information Theory)
**Recommendation:** **Accept without reservation**

#### Comments:
This is a seminal contribution to the theory and empirical detection of causal emergence in nonstationary continuous dynamical systems. In previous literature, causal emergence (Hoel et al., 2013; Rosas et al., 2020) was largely restricted to stationary discrete Markov chains or static linear systems. The authors successfully generalize these ideas to time-varying continuous systems through the Causal Information Spectrum $\lambda_k(t) = \operatorname{eig}(\Sigma_t^{-1/2} A_t A_t^\top \Sigma_t^{-1/2})$ and its continuous dimensionality metrics: the Dynamic Causal Participation Ratio ($DCD^{\text{PR}}$) and Causal Effective Rank ($DCD^{\text{ER}}$).

I specifically scrutinized Section 2.3 and Section 2.5 regarding the Data Processing Inequality. The authors' clarification is mathematically impeccable:
- Under a single joint observational distribution $P(X_t, X_{t+1})$, DPI strictly holds: $I_{\text{obs}}(W^\top X_t; W^\top X_{t+1}) \le I_{\text{obs}}(X_t; X_{t+1})$.
- However, Effective Information is an interventional quantity evaluated under maximum-entropy isotropic drives $do(X) \sim \mathcal{N}(0, I_p)$ and $do(V) \sim \mathcal{N}(0, I_q)$ via observational lifting $\kappa_W(dx \mid v) = \delta(x - Wv)$. Because the two interventional drives induce distinct marginal input distributions on $\mathbb{R}^p$ ($\mathcal{N}(0, I_p)$ versus $\mathcal{N}(0, W W^\top)$), they represent two separate experiments. The authors provide a brilliant, fully verified 2D numerical counterexample proving that noise cancellation and subspace alignment permit positive raw emergence $\Delta EI^{\text{raw}} > 0$.
- Crucially, the authors honestly recognize that in continuous Gaussian systems, emergence manifests predominantly as Causal Concentration Gain ($CCG$).

The synthetic benchmarks (DGPs A–K) cleanly separate raw emergence from density gain and prove that DCD overcomes running-average degeneracy. I recommend immediate publication.

---

### Reviewer 2 (Energy Systems & Applied Econometrics)
**Recommendation:** **Accept as is**

#### Comments:
The empirical application to the Continental US Power Grid (Form EIA-930) is executed with remarkable econometric rigor. Too often, machine learning papers applied to power grids suffer from data leakage, unvalidated surrogates, or overstated claims of "tipping points." The authors avoid every one of these pitfalls:
1. **Data Provenance & Physical Accounting:** Ingesting 276,717 BA-hours from an immutable Zenodo snapshot (DOI: 10.5281/zenodo.22215263) and verifying that physical balance residuals remain $< 1.8\%$ provides complete operational confidence.
2. **Hansen Supremum-Wald Tests:** The analysis of renewable generation (VRE) penetration via GAMs and Hansen (1996) tests is exemplary. The moving-block bootstrap ($B=2000$) re-estimates the exact Newey-West HAC covariance in every draw. The authors' scientific honesty is commendable: rather than claiming an alarming "grid tipping point," they report $p = 0.5582$ in ERCOT, demonstrating that the grid exhibits smooth, continuous operational coordination. In Western, they detect localized slope modulation ($p = 0.0045$) without collapse.
3. **Winter Storm Uri Event Study:** Enforcing a 14-day global exclusion window around the February 2021 Texas crisis prevents baseline contamination. The counter-intuitive finding that $DCD^{\text{PR}}$ expanded by $+1.094$ ($p = 0.0010$) with 0% collapse rate is physically grounded: ERCOT operators dynamically engaged auxiliary and emergency generation, expanding effective balancing degrees of freedom.
4. **Forecasting Diagnostics:** Testing $h \in \{1, 6, 12, 24\}$ under strict lookahead-free walk-forward splits with Diebold-Mariano HAC tests and Benjamini-Hochberg FDR correction proves that DCD is an interpretable structural diagnostic, not a naive linear forecasting cheat.

The econometric and physical standards meet the highest publication tier.

---

### Reviewer 3 (Machine Learning & Mathematical Statistics)
**Recommendation:** **Accept**

#### Comments:
I conducted an adversarial code and reproducibility audit of the accompanying repository. The standards implemented here should serve as a model for the field:
- **Surrogate Quality Protocol (v2.1):** The multivariate IAAFT surrogate generation employs strict fail-hard rejection sampling. All $B=1000$ surrogates across ERCOT, Western, and Eastern satisfy strict spectral and covariance tolerances, with zero unvalidated fallback realizations.
- **Sensitivity & Hold-out:** The sensitivity analysis across auto-spectral Frobenius thresholds $\tau \in \{0.08, 0.10, 0.12, 0.15\}$ confirms robust significance ($p < 0.01$). The 2022H2 holdout ($T=4,423$ h) successfully replicates the 2021 contraction pattern out-of-sample ($p = 0.0099$).
- **Code & Test Suite:** The unified test suite of 86 tests passes cleanly (0 failures, 0 warnings) in under 15 seconds. The one-command reproduction script (`python3 scripts/reproduce_all.py --compile-latex`) recomputed all artifacts, figures, and compiled the LaTeX document without issue.
- **Layout & Presentation:** The manuscript strictly satisfies the 12-page budget, with all 22 references and Data/Code Availability cleanly terminating on Page 8, followed by Figures 3–6 on Pages 9–12.

---

## Associate Editor Decision

All three reviewers have issued unqualified recommendations to accept. The authors have resolved all 10 blockers identified during prior review cycles. The theoretical claims, empirical tests, synthetic benchmarks, and computational codebase are in perfect alignment.

**Final Decision:** **ACCEPT FOR PUBLICATION IN FULL FORMAT**.
