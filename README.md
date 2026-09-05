# Dynamic Causal Emergence (DCE)

[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-Zenodo%20Record%2022215263-orange.svg)](https://doi.org/10.5281/zenodo.22215263)

Official implementation and replication package for:
> **Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems: Evidence from Renewable-Dominated Continental Power Grids**  
> *Felipe Mora-Rojas* (2026).

---

## 1. Overview

**Dynamic Causal Emergence ($DCE_t$)** is a mathematically rigorous and neural variational framework designed to quantify time-varying macroscopic causality in nonstationary complex systems.

Existing Causal Emergence (CE) formulations and deep learning estimators implicitly assume stationarity in the underlying dynamical law $P(X_{t+1}|X_t)$, rendering them blind to the evolving operational regimes, critical phase transitions, and multiscale reorganizations ubiquitous in real-world systems. 

This repository provides:
- **Local Linear Gaussian DCE (`LocalLinearGaussianDCE`)**: Exact closed-form analytical reference estimator using locally-weighted linear regression, maximum-entropy interventional distributions, and analytical Effective Information ($EI$).
- **Dyn-NIS+ Neural Estimator (`DynNISPlus`)**: Deep neural variational architecture that jointly optimizes macroscopic prediction error, latent Effective Information, Procrustes representation alignment, and parameter-manifold smoothness, suppressing parameter chattering by 80.0%.
- **Nonstationary Benchmark Suite (DGPs A–I)**: 9 synthetic data-generating processes with analytical ground truths for Monte Carlo calibration and falsification.
- **Continental US Power Grid Pipeline (EIA-930)**: Ingestion of authentic hourly operational data from immutable Zenodo archive snapshots (Record `22215263`, CC-BY-4.0) across Eastern ($p=55$), Western ($p=45$), and ERCOT ($p=9$) interconnections with operational residual auditing and leak-free causal rolling scaling.
- **Statistical Inference Suite (H1–H4)**: Multivariate IAAFT surrogates with full model refitting, GAMM spline and threshold regressions, matched event studies for extreme weather (Winter Storm Uri), and walk-forward rolling out-of-sample forecast evaluation.
- **Multiscale CE 2.0 Apportioning**: Causal density decomposition across nested physical grid tiers (Micro BAs, RTOs, Continental Interconnections).

---

## 2. Repository Structure

```
.
├── CITATION.cff                      # Standard citation metadata
├── LICENSE                           # Open source MIT license
├── MASTER_Q1_IMPLEMENTATION_PLAN...  # Research protocol and implementation specification
├── README.md                         # This file
├── data/
│   ├── raw/                          # Authentic EIA-930 raw zip archives & manifest.json
│   └── processed/                    # Clean parquet panels (2021, 2022H2)
├── paper/
│   ├── main.tex                      # Publication LaTeX manuscript
│   ├── main.pdf                      # Compiled PDF (10 pages, 7 figures, Table 1)
│   ├── references.bib                # BibTeX database
│   ├── claim_ledger.csv              # Formal audit ledger of all scientific claims
│   └── figures/                      # High-resolution vector PDF publication figures
│       ├── fig1_conceptual_framework.pdf
│       ├── fig2_synthetic_validation.pdf
│       ├── fig3_power_grid_trajectory.pdf
│       ├── fig4_vre_nonlinear_phase_transition.pdf
│       ├── fig5_extreme_events_collapse.pdf
│       ├── fig6_multiscale_ce2_apportioning.pdf
│       └── fig7_baseline_comparison.pdf
├── results/
│   ├── synthetic/                    # Monte Carlo calibration results
│   └── empirical/                    # Grid DCE panels and hypothesis testing JSON artifacts
├── scripts/
│   ├── compare_estimators_dgp.py     # Benchmark comparison script
│   ├── generate_all_paper_figures.py # Script generating Figures 1-7 from frozen artifacts
│   └── reproduce_all.py              # Single-command master replication runner
├── src/dce/
│   ├── datasets/eia930/              # Real EIA-930 ingestion, balance auditing, and microstates
│   ├── estimators/                   # Local linear Gaussian, Dyn-NIS+, and Multiscale CE2
│   ├── synthetic/                    # DGPs A through I generators
│   └── stats/                        # IAAFT surrogates and H1-H4 hypothesis tests
└── tests/                            # Comprehensive unit test suite (38/38 passing)
```

---

## 3. Quick Start

### Installation
```bash
git clone https://github.com/anonymous/dynamic-causal-emergence.git
cd dynamic-causal-emergence
conda create -n dce python=3.11 -y
conda activate dce
pip install -e .
```

### Run Unit Tests
```bash
pytest tests/
```
All 38 unit tests run in ~12 seconds and verify:
- Analytical Gaussian oracle closed forms against numerical integrals.
- Orthonormality and Procrustes alignment invariance.
- Real EIA-930 raw ingestion and physical balance residuals.
- IAAFT surrogate spectral conservation.
- GAMM threshold estimation and walk-forward forecasting.

---

## 4. End-to-End Master Replication

To verify data integrity against Zenodo SHA-256 hashes, validate empirical artifacts, regenerate all publication figures, recompile the LaTeX manuscript, and display the claim audit ledger:

```bash
python scripts/reproduce_all.py
```

### Expected Output:
```
[INFO] Starting Dynamic Causal Emergence (DCE) Master Replication...
[INFO] Verifying raw EIA-930 data archives against Zenodo cryptographic hashes...
[INFO]   [OK] eia930-2021half1.zip (9ca67bd2e07d...)
[INFO]   [OK] eia930-2021half2.zip (83e0ad541664...)
[INFO]   [OK] eia930-2022half2.zip (531fb1bfb394...)
[INFO] Raw data verification PASSED (Zenodo CC-BY-4.0).
[INFO]   [FOUND] results/empirical/ercot_dce_2021_retrospective.parquet
[INFO]   [FOUND] results/empirical/ercot_dce_2021_causal.parquet
[INFO]   [FOUND] results/empirical/western_dce_2021_retrospective.parquet
[INFO]   [FOUND] results/empirical/eastern_dce_2021_retrospective.parquet
[INFO]   [FOUND] results/empirical/h1_surrogate_results.json
[INFO]   [FOUND] results/empirical/h2_gamm_results.json
[INFO]   [FOUND] results/empirical/h3_event_study_results.json
[INFO]   [FOUND] results/empirical/h4_forecast_results.json
[INFO]   [FOUND] results/empirical/ce2_apportioning_2021.parquet
[INFO] Empirical artifacts verification PASSED.
[INFO] Regenerating all paper figures from genuine experimental artifacts...
[INFO] All 7 figures generated successfully in paper/figures/
[INFO] Compiling publication manuscript paper/main.tex...
[INFO] Manuscript compiled successfully: paper/main.pdf (605,665 bytes)
=== SCIENTIFIC CLAIM VERIFICATION LEDGER ===
claim_id                  status                                                                                 claim_text
     C01                verified                                DCE is calibrated under null nonstationarity (FPR <= alpha)
     C02                verified                     Dyn-NIS+ improves tracking and detection delay over windowed baselines
     C03                verified     Dynamic causal dimension q_t^* is accurately recovered with calibrated confidence sets
     C04                verified                        Linear-Gaussian analytical oracle matches numerical local estimator
     C05                verified                              Statistically significant DCE episodes occur in US power grid
     C06                verified                  VRE has a nonlinear association with DCE with verified threshold behavior
     C07                verified Severe grid stress events exhibit significant dimensional collapse to rigid macro dynamics
     C08 falsified_reported_null One-sided causal DCE evaluation indicates no unconditional linear forecast error reduction
============================================
[INFO] Dynamic Causal Emergence pipeline replication COMPLETE and VERIFIED.
```

---

## 5. Summary of Empirical Findings

| Hypothesis / Metric | Statistical Result | Status & Interpretation |
|---|---|---|
| **H1 (Emergence)** | Empirical mean $DCE_t = 0.00392$ nats exceeds 50 IAAFT surrogates with full model refitting ($p = 0.0196 < 0.05$). | **Verified**: Macroscopic coarse-graining filters out microscopic stochasticity. |
| **H2 (Renewable Modulation)** | GAMM explains 34.8% deviance in ERCOT and 36.3% in Western. Structural breakpoints at $\hat{\gamma} = 21.6\%$ in ERCOT ($p = 5.94 \times 10^{-50}$) and $\hat{\gamma} = 37.2\%$ in Western ($p = 1.77 \times 10^{-6}$). | **Verified**: Non-linear transition from decentralized thermal dispatch to coordinated renewable ramping. |
| **H3 (Extreme Weather Collapse)** | Winter Storm Uri event study demonstrates dimensional collapse rate ($q^* \le Q_{0.10}$) surging from $11.04\%$ baseline to $19.27\%$ during freeze ($p = 1.77 \times 10^{-4}$). | **Verified**: Extreme grid stress forces decentralized subsystems into a single catastrophic macro-failure mode. |
| **H4 (Operational Predictability)** | Diebold-Mariano tests on walk-forward rolling regressions confirm unconditional linear demand forecast error gains are statistically indistinguishable from zero ($p \in [0.15, 0.71]$). | **Falsified / Honest Null Reported**: $DCE_t$ is a physical structural diagnostic of collective coordination and failure susceptibility, not an unconditional linear forecaster. |

---

## 6. Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{morarojas2026dynamic,
  title={Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems: Evidence from Renewable-Dominated Continental Power Grids},
  author={Mora-Rojas, Felipe},
  journal={Preprint / Under Review},
  year={2026},
  url={https://github.com/anonymous/dynamic-causal-emergence}
}
```

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Raw data archives are sourced from Catalyst Cooperative's Public Utility Data Liberation (PUDL) project on Zenodo under Creative Commons Attribution 4.0 International (CC-BY-4.0).
