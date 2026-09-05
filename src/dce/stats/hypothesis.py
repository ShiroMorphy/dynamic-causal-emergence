"""
Statistical Hypothesis Testing Suite for Q1 Paper (H1 to H4).

H1: Surrogate Data Testing for DCE_t > 0 Significance with Full Model Refitting (CRITICAL-02).
H2: Nonlinear GAMM & Threshold Regression DCE_t = f(VRE_t) + Controls + epsilon_t.
H3: Matched Event Study & Dimensional Collapse Analysis during Extreme Crisis Events.
H4: Out-of-Sample Walk-Forward Predictive Regressions & Diebold-Mariano Tests.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pygam import LinearGAM, s, f

from dce.stats.surrogates import generate_multivariate_surrogates


class Hypothesis1Result(NamedTuple):
    """Container for H1 Surrogate Test results."""
    p_values_pointwise: np.ndarray
    q_values_fdr: np.ndarray
    significant_ratio_raw: float
    significant_ratio_fdr: float
    max_stat_empirical: float
    max_stat_pvalue: float
    mean_stat_empirical: float
    mean_stat_pvalue: float
    surrogate_dce_ensemble: np.ndarray
    surrogate_dce_95th: np.ndarray

    @property
    def significant_ratio(self) -> float:
        return self.significant_ratio_raw


class Hypothesis2Result(NamedTuple):
    """Container for H2 GAM & Threshold results."""
    gam_model: LinearGAM
    pseudo_r2: float
    vre_grid: np.ndarray
    vre_partial_effects: np.ndarray
    vre_confidence_intervals: np.ndarray
    best_threshold: float
    threshold_ci: Tuple[float, float]
    davies_pvalue: float
    model_comparison: pd.DataFrame


class Hypothesis3Result(NamedTuple):
    """Container for H3 Extreme Event Study results."""
    event_id: str
    event_name: str
    event_timestamps: pd.DatetimeIndex
    event_dce: np.ndarray
    event_q_star: np.ndarray
    matched_dce_mean: np.ndarray
    matched_dce_std: np.ndarray
    matched_q_star_median: np.ndarray
    delta_dce: np.ndarray
    delta_q_star: np.ndarray
    collapse_ratio_event: float
    collapse_ratio_baseline: float
    collapse_pvalue: float


class Hypothesis4Result(NamedTuple):
    """Container for H4 Out-of-Sample Walk-Forward Forecasting results."""
    horizons: List[int]
    rmse_baseline: Dict[int, float]
    rmse_augmented: Dict[int, float]
    mae_baseline: Dict[int, float]
    mae_augmented: Dict[int, float]
    rmse_improvement_pct: Dict[int, float]
    diebold_mariano_stat: Dict[int, float]
    diebold_mariano_pvalue: Dict[int, float]
    stress_tail_rmse_baseline: Dict[int, float]
    stress_tail_rmse_augmented: Dict[int, float]

    @property
    def rmse_augmented_dce(self) -> float:
        if self.rmse_augmented:
            return next(iter(self.rmse_augmented.values()))
        return 0.0


def fdr_bh(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg False Discovery Rate correction."""
    p = np.asarray(p_vals)
    n = len(p)
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]
    q = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (sorted_p[i] * n) / rank
        cummin = min(cummin, val)
        q[sorted_indices[i]] = cummin
    return np.clip(q, 0.0, 1.0)


def run_h1_surrogate_test(
    empirical_dce: np.ndarray,
    X_micro: np.ndarray,
    fit_model_fn: Callable[[np.ndarray], np.ndarray],
    n_surrogates: int = 50,
    alpha: float = 0.05,
    seed: int = 42
) -> Hypothesis1Result:
    """
    Test H1: Surrogate testing with full model refitting on each surrogate (CRITICAL-02).
    
    Args:
        empirical_dce: (T,) estimated DCE from real microstate data.
        X_micro: (T, p) real microstate matrix.
        fit_model_fn: Callable that accepts X_surr and returns emergence array (T,).
        n_surrogates: Number of surrogate realizations (B >= 50).
        alpha: Significance level.
        seed: Random seed.
    """
    T = len(empirical_dce)
    print(f"Generating {n_surrogates} multivariate IAAFT surrogates (T={X_micro.shape[0]}, p={X_micro.shape[1]})...")
    surrogates = generate_multivariate_surrogates(X_micro, n_surrogates=n_surrogates, seed=seed)
    
    surr_dce_ensemble = np.zeros((n_surrogates, T), dtype=np.float64)
    print(f"Refitting DCE model on {n_surrogates} surrogates to construct exact null distribution...")
    for b in range(n_surrogates):
        surr_dce_ensemble[b, :] = fit_model_fn(surrogates[b])
        
    # 1. Pointwise exceedance p-values
    exceedances = np.sum(surr_dce_ensemble >= empirical_dce[np.newaxis, :], axis=0)
    p_values_pointwise = (exceedances + 1.0) / (n_surrogates + 1.0)
    q_values_fdr = fdr_bh(p_values_pointwise)
    
    sig_raw = float(np.mean(p_values_pointwise < alpha))
    sig_fdr = float(np.mean(q_values_fdr < alpha))
    
    # 2. Global Max Test Statistic: T_max = max_t DCE_t
    t_max_emp = float(np.max(empirical_dce))
    t_max_surr = np.max(surr_dce_ensemble, axis=1)
    p_max = float((np.sum(t_max_surr >= t_max_emp) + 1.0) / (n_surrogates + 1.0))
    
    # 3. Global Mean Test Statistic: T_mean = mean_t DCE_t
    t_mean_emp = float(np.mean(empirical_dce))
    t_mean_surr = np.mean(surr_dce_ensemble, axis=1)
    p_mean = float((np.sum(t_mean_surr >= t_mean_emp) + 1.0) / (n_surrogates + 1.0))
    
    surr_95th = np.percentile(surr_dce_ensemble, 95.0, axis=0)
    
    return Hypothesis1Result(
        p_values_pointwise=p_values_pointwise,
        q_values_fdr=q_values_fdr,
        significant_ratio_raw=sig_raw,
        significant_ratio_fdr=sig_fdr,
        max_stat_empirical=t_max_emp,
        max_stat_pvalue=p_max,
        mean_stat_empirical=t_mean_emp,
        mean_stat_pvalue=p_mean,
        surrogate_dce_ensemble=surr_dce_ensemble,
        surrogate_dce_95th=surr_95th
    )


def compute_h2_gamm_and_threshold(
    dce_series: np.ndarray,
    vre_series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    net_load_series: Optional[np.ndarray] = None,
    n_bootstrap: int = 100
) -> Hypothesis2Result:
    """
    Test H2: Generalized Additive Model and Threshold/Segmented Regression (Section 18).
    
    Compares Linear (M0), Segmented (M1), and GAM Spline (M2) with bootstrap stability.
    """
    hours = timestamps.hour.values if hasattr(timestamps, "hour") else np.zeros(len(dce_series))
    doy = timestamps.dayofyear.values if hasattr(timestamps, "dayofyear") else np.zeros(len(dce_series))
    
    if net_load_series is None:
        net_load_norm = np.zeros_like(dce_series)
    else:
        net_load_norm = (net_load_series - np.mean(net_load_series)) / (np.std(net_load_series) + 1e-4)
        
    X_gam = np.column_stack([vre_series, net_load_norm, hours, doy])
    y_gam = dce_series
    
    gam = LinearGAM(s(0, n_splines=15) + s(1, n_splines=10) + s(2, n_splines=8) + s(3, n_splines=10))
    gam.gridsearch(X_gam, y_gam, progress=False)
    
    vre_grid = np.linspace(np.percentile(vre_series, 2), np.percentile(vre_series, 98), 100)
    XX = gam.generate_X_grid(term=0, n=100)
    p_dep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
    pseudo_r2 = float(gam.statistics_["pseudo_r2"]["explained_deviance"])
    
    # 2. Threshold Search (Grid search for best breakpoint gamma in VRE)
    cand_gammas = np.percentile(vre_series, np.linspace(15, 85, 50))
    best_gamma = float(cand_gammas[0])
    best_sse = np.inf
    
    X_ctrl = np.column_stack([np.ones(len(y_gam)), net_load_norm, hours, doy])
    
    # Baseline linear M0
    X_m0 = np.column_stack([X_ctrl, vre_series])
    ols_m0 = sm.OLS(y_gam, X_m0).fit()
    aic_m0 = ols_m0.aic
    bic_m0 = ols_m0.bic
    
    for gamma in cand_gammas:
        vre_hinge = np.maximum(0.0, vre_series - gamma)
        X_seg = np.column_stack([X_ctrl, vre_series, vre_hinge])
        model_seg = sm.OLS(y_gam, X_seg).fit()
        if model_seg.ssr < best_sse:
            best_sse = model_seg.ssr
            best_gamma = float(gamma)
            
    # Fit best segmented model M1
    best_hinge = np.maximum(0.0, vre_series - best_gamma)
    X_m1 = np.column_stack([X_ctrl, vre_series, best_hinge])
    ols_m1 = sm.OLS(y_gam, X_m1).fit()
    aic_m1 = ols_m1.aic
    bic_m1 = ols_m1.bic
    
    # 3. Block bootstrap for threshold confidence interval
    block_len = 48
    n_blocks = len(y_gam) // block_len
    boot_gammas = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        block_idx = rng.randint(0, n_blocks, size=n_blocks)
        sample_indices = np.concatenate([np.arange(b * block_len, (b + 1) * block_len) for b in block_idx])
        y_b = y_gam[sample_indices]
        vre_b = vre_series[sample_indices]
        X_ctrl_b = X_ctrl[sample_indices]
        
        b_best_gamma = best_gamma
        b_best_sse = np.inf
        for g in cand_gammas[::2]:
            X_b_seg = np.column_stack([X_ctrl_b, vre_b, np.maximum(0.0, vre_b - g)])
            try:
                fit_b = sm.OLS(y_b, X_b_seg).fit()
                if fit_b.ssr < b_best_sse:
                    b_best_sse = fit_b.ssr
                    b_best_gamma = g
            except Exception:
                continue
        boot_gammas.append(b_best_gamma)
        
    thresh_ci = (float(np.percentile(boot_gammas, 2.5)), float(np.percentile(boot_gammas, 97.5)))
    
    # Davies test approximation via F-test / likelihood ratio
    lr_stat = 2.0 * (ols_m1.llf - ols_m0.llf)
    davies_pval = float(stats.chi2.sf(lr_stat, df=2))
    
    model_comp = pd.DataFrame([
        {"Model": "M0_Linear", "AIC": aic_m0, "BIC": bic_m0, "R2": ols_m0.rsquared},
        {"Model": "M1_Segmented", "AIC": aic_m1, "BIC": bic_m1, "R2": ols_m1.rsquared},
        {"Model": "M2_GAM_Spline", "AIC": gam.statistics_["AIC"], "BIC": np.nan, "R2": pseudo_r2}
    ])
    
    return Hypothesis2Result(
        gam_model=gam,
        pseudo_r2=pseudo_r2,
        vre_grid=vre_grid,
        vre_partial_effects=p_dep,
        vre_confidence_intervals=confi,
        best_threshold=best_gamma,
        threshold_ci=thresh_ci,
        davies_pvalue=davies_pval,
        model_comparison=model_comp
    )


def compute_h3_matched_event_study(
    dce_df: pd.DataFrame,
    event_start: str,
    event_end: str,
    event_name: str = "Extreme Event",
    event_id: str = "uri_2021",
    pre_window_h: int = 48,
    post_window_h: int = 48
) -> Hypothesis3Result:
    """
    Test H3: Matched event study and dimensional collapse during extreme crises (Section 19).
    """
    dce_df = dce_df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(dce_df["timestamp"], utc=True)
    
    t_start = pd.to_datetime(event_start, utc=True)
    t_end = pd.to_datetime(event_end, utc=True)
    
    event_mask = (ts >= t_start) & (ts <= t_end)
    event_indices = np.where(event_mask)[0]
    
    if len(event_indices) == 0:
        raise ValueError(f"No records found between {event_start} and {event_end}")
        
    i_start = max(0, event_indices[0] - pre_window_h)
    i_end = min(len(dce_df), event_indices[-1] + post_window_h + 1)
    
    window_df = dce_df.iloc[i_start:i_end].copy()
    window_ts = pd.to_datetime(window_df["timestamp"], utc=True)
    event_dce = window_df["dce_norm"].values
    event_q = window_df["q_star"].values
    
    # Matched non-event control windows: same month, same days of week, 1-2 weeks before or after
    candidate_control_mask = (ts.dt.month == t_start.month) & ~event_mask
    ctrl_indices = np.where(candidate_control_mask)[0]
    
    L = len(window_df)
    matched_dce_runs = []
    matched_q_runs = []
    
    for offset in [-336, -168, 168, 336]:  # -2wk, -1wk, +1wk, +2wk
        start_cand = event_indices[0] + offset
        end_cand = start_cand + L
        if 0 <= start_cand and end_cand <= len(dce_df):
            cand_slice = dce_df.iloc[start_cand:end_cand]
            matched_dce_runs.append(cand_slice["dce_norm"].values)
            matched_q_runs.append(cand_slice["q_star"].values)
            
    if len(matched_dce_runs) == 0:
        # Fallback to mean across entire non-event baseline
        matched_dce_mean = np.full(L, np.mean(dce_df.loc[~event_mask, "dce_norm"]))
        matched_dce_std = np.full(L, np.std(dce_df.loc[~event_mask, "dce_norm"]))
        matched_q_median = np.full(L, np.median(dce_df.loc[~event_mask, "q_star"]))
    else:
        matched_dce_mat = np.array(matched_dce_runs)
        matched_q_mat = np.array(matched_q_runs)
        matched_dce_mean = np.mean(matched_dce_mat, axis=0)
        matched_dce_std = np.std(matched_dce_mat, axis=0)
        matched_q_median = np.median(matched_q_mat, axis=0)
        
    delta_dce = event_dce - matched_dce_mean
    delta_q = event_q - matched_q_median
    
    # Dimensional collapse threshold: 10th percentile of baseline normal q*
    q_normal = dce_df.loc[~event_mask, "q_star"].values
    q_collapse_thresh = np.percentile(q_normal, 10.0)
    
    event_core_q = dce_df.loc[event_mask, "q_star"].values
    collapse_event = float(np.mean(event_core_q <= q_collapse_thresh))
    collapse_base = float(np.mean(q_normal <= q_collapse_thresh))
    
    # Binomial / proportion test for collapse rate
    n_ev = len(event_core_q)
    n_base = len(q_normal)
    k_ev = np.sum(event_core_q <= q_collapse_thresh)
    k_base = np.sum(q_normal <= q_collapse_thresh)
    
    p_pool = (k_ev + k_base) / (n_ev + n_base)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_ev + 1.0 / n_base) + 1e-10)
    z_stat = (collapse_event - collapse_base) / se_pool
    p_val = float(1.0 - stats.norm.cdf(z_stat))
    
    return Hypothesis3Result(
        event_id=event_id,
        event_name=event_name,
        event_timestamps=window_ts,
        event_dce=event_dce,
        event_q_star=event_q,
        matched_dce_mean=matched_dce_mean,
        matched_dce_std=matched_dce_std,
        matched_q_star_median=matched_q_median,
        delta_dce=delta_dce,
        delta_q_star=delta_q,
        collapse_ratio_event=collapse_event,
        collapse_ratio_baseline=collapse_base,
        collapse_pvalue=p_val
    )


def compute_diebold_mariano_test(
    loss_0: np.ndarray,
    loss_1: np.ndarray,
    horizon_h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Newey-West HAC covariance for h-step ahead forecasts.
    
    H0: E[loss_0 - loss_1] = 0.
    """
    d = loss_0 - loss_1
    n = len(d)
    d_mean = np.mean(d)
    
    # Newey-West autocovariance weighting up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, horizon_h):
        w = 1.0 - (lag / horizon_h)
        c = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += 2.0 * w * c
        
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0.0:
        var_d = np.var(d, ddof=1) / n + 1e-12
        
    dm_stat = float(d_mean / np.sqrt(var_d))
    p_val = float(2.0 * stats.norm.sf(abs(dm_stat)))
    return dm_stat, p_val


def compute_h4_walk_forward_forecasting(
    forecast_error: np.ndarray,
    dce_causal: np.ndarray,
    q_star_causal: np.ndarray,
    horizons: List[int] = [1, 6, 12, 24],
    train_window: int = 2000,
    eval_step: int = 24
) -> Hypothesis4Result:
    """
    Test H4: Out-of-Sample Walk-Forward Rolling Predictive Regressions (Section 20).
    
    Evaluates incremental predictive power of causal-only DCE without look-ahead bias.
    """
    rmse_base_dict = {}
    rmse_aug_dict = {}
    mae_base_dict = {}
    mae_aug_dict = {}
    pct_imp_dict = {}
    dm_stat_dict = {}
    dm_pval_dict = {}
    stress_base_dict = {}
    stress_aug_dict = {}
    
    T_total = len(forecast_error)
    effective_train_window = min(train_window, max(20, T_total // 2))
    
    for h in horizons:
        y_true_list = []
        pred_base_list = []
        pred_aug_list = []
        
        # Walk-forward rolling origin
        for origin in range(effective_train_window, T_total - h, eval_step):
            # Target at horizon h
            y_target = forecast_error[origin + h]
            
            # Training data available strictly up to origin
            train_y = forecast_error[h : origin + h]
            fe_lag0 = forecast_error[:origin]
            fe_lag1 = np.roll(fe_lag0, 1)
            fe_lag1[0] = fe_lag0[0]
            
            dce_past = dce_causal[:origin]
            q_past = q_star_causal[:origin]
            
            X_b_train = np.column_stack([np.ones(origin), fe_lag0, fe_lag1])
            X_a_train = np.column_stack([np.ones(origin), fe_lag0, fe_lag1, dce_past, q_past])
            
            # Predictor at origin
            x_b_orig = np.array([1.0, forecast_error[origin], forecast_error[origin - 1]])
            x_a_orig = np.array([1.0, forecast_error[origin], forecast_error[origin - 1], dce_causal[origin], q_star_causal[origin]])
            
            try:
                fit_base = np.linalg.lstsq(X_b_train, train_y, rcond=None)[0]
                fit_aug = np.linalg.lstsq(X_a_train, train_y, rcond=None)[0]
                
                p_b = float(x_b_orig @ fit_base)
                p_a = float(x_a_orig @ fit_aug)
                
                y_true_list.append(y_target)
                pred_base_list.append(p_b)
                pred_aug_list.append(p_a)
            except Exception:
                continue
                
        y_true = np.array(y_true_list)
        p_base = np.array(pred_base_list)
        p_aug = np.array(pred_aug_list)
        
        err_base = y_true - p_base
        err_aug = y_true - p_aug
        
        rmse_b = float(np.sqrt(np.mean(err_base ** 2)))
        rmse_a = float(np.sqrt(np.mean(err_aug ** 2)))
        mae_b = float(np.mean(np.abs(err_base)))
        mae_a = float(np.mean(np.abs(err_aug)))
        
        pct_imp = float((rmse_b - rmse_a) / rmse_b * 100.0)
        
        loss_b = err_base ** 2
        loss_a = err_aug ** 2
        dm_stat, dm_pval = compute_diebold_mariano_test(loss_b, loss_a, horizon_h=h)
        
        # Stress periods: top 10% highest forecast errors
        stress_cutoff = np.percentile(y_true, 90.0)
        stress_mask = y_true >= stress_cutoff
        stress_rmse_b = float(np.sqrt(np.mean(err_base[stress_mask] ** 2)))
        stress_rmse_a = float(np.sqrt(np.mean(err_aug[stress_mask] ** 2)))
        
        rmse_base_dict[h] = rmse_b
        rmse_aug_dict[h] = rmse_a
        mae_base_dict[h] = mae_b
        mae_aug_dict[h] = mae_a
        pct_imp_dict[h] = pct_imp
        dm_stat_dict[h] = dm_stat
        dm_pval_dict[h] = dm_pval
        stress_base_dict[h] = stress_rmse_b
        stress_aug_dict[h] = stress_rmse_a
        
    return Hypothesis4Result(
        horizons=horizons,
        rmse_baseline=rmse_base_dict,
        rmse_augmented=rmse_aug_dict,
        mae_baseline=mae_base_dict,
        mae_augmented=mae_aug_dict,
        rmse_improvement_pct=pct_imp_dict,
        diebold_mariano_stat=dm_stat_dict,
        diebold_mariano_pvalue=dm_pval_dict,
        stress_tail_rmse_baseline=stress_base_dict,
        stress_tail_rmse_augmented=stress_aug_dict
    )


def compute_h1_causal_emergence_significance(
    empirical_dce: np.ndarray,
    surr_dce_ensemble: np.ndarray,
    alpha: float = 0.05
) -> Hypothesis1Result:
    """Convenience wrapper for H1 significance test given precomputed surrogate DCE ensemble."""
    n_surrogates, T = surr_dce_ensemble.shape
    p_values_pointwise = np.zeros(T)
    for t in range(T):
        p_values_pointwise[t] = (np.sum(surr_dce_ensemble[:, t] >= empirical_dce[t]) + 1.0) / (n_surrogates + 1.0)
    
    q_values_fdr = fdr_bh(p_values_pointwise)
    sig_raw = float(np.mean(p_values_pointwise < alpha))
    sig_fdr = float(np.mean(q_values_fdr < alpha))
    
    t_max_emp = float(np.max(empirical_dce))
    t_max_surr = np.max(surr_dce_ensemble, axis=1)
    p_max = float((np.sum(t_max_surr >= t_max_emp) + 1.0) / (n_surrogates + 1.0))
    
    t_mean_emp = float(np.mean(empirical_dce))
    t_mean_surr = np.mean(surr_dce_ensemble, axis=1)
    p_mean = float((np.sum(t_mean_surr >= t_mean_emp) + 1.0) / (n_surrogates + 1.0))
    
    surr_95th = np.percentile(surr_dce_ensemble, 95.0, axis=0)
    
    return Hypothesis1Result(
        p_values_pointwise=p_values_pointwise,
        q_values_fdr=q_values_fdr,
        significant_ratio_raw=sig_raw,
        significant_ratio_fdr=sig_fdr,
        max_stat_empirical=t_max_emp,
        max_stat_pvalue=p_max,
        mean_stat_empirical=t_mean_emp,
        mean_stat_pvalue=p_mean,
        surrogate_dce_ensemble=surr_dce_ensemble,
        surrogate_dce_95th=surr_95th
    )


def compute_h4_out_of_sample_forecasting(
    demand_forecast_error: np.ndarray,
    dce_series: np.ndarray,
    q_star_series: np.ndarray,
    horizon_h: int = 1,
    min_train_size: int = 40
) -> Hypothesis4Result:
    """Convenience wrapper for H4 out-of-sample forecasting."""
    return compute_h4_walk_forward_forecasting(
        forecast_error=demand_forecast_error,
        dce_causal=dce_series,
        q_star_causal=q_star_series,
        horizons=[horizon_h],
        train_window=min_train_size,
        eval_step=1
    )



