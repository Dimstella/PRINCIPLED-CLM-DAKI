"""
matching/psm.py
---------------
Propensity-Score Matching (Algorithm 3, Dimitsaki et al. 2026) and all
alternative matching methods used in the benchmark.

Public API
----------
psm_caliper_k1(X, T, Y, propensity_scores, caliper_coef, k) -> (X_m, T_m, Y_m)
compute_smd(X, T)                    -> pd.Series
evaluate_balance(X_matched, T_matched, col_names) -> BalanceResult
apply_all_matching_methods(X, T, Y)  -> (summary_df, smd_df)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMD_THRESHOLD = 0.2   # covariate balance threshold (Austin, 2009)
PSM_CALIPER   = 0.2   # caliper coefficient (× SD of logit PS)
PSM_K         = 4     # treated : control matching ratio  (k:1 = 4:1)


# ---------------------------------------------------------------------------
# Data class for balance results
# ---------------------------------------------------------------------------

@dataclass
class BalanceResult:
    mean_smd: float
    n_exceeding_threshold: int
    n_treated: int
    n_control: int
    smd_per_covariate: pd.Series


# ---------------------------------------------------------------------------
# SMD calculation  (Equation 2 in the paper)
# ---------------------------------------------------------------------------

def compute_smd(X: pd.DataFrame, T: pd.Series) -> pd.Series:
    """
    Standardised Mean Difference for each covariate.

    SMD = (mean_T1 - mean_T0) / sqrt((var_T1 + var_T0) / 2)

    Parameters
    ----------
    X : pd.DataFrame  — covariate matrix
    T : pd.Series     — treatment indicator (1 = treated, 0 = control)

    Returns
    -------
    pd.Series  — SMD per covariate (signed; caller takes abs() if needed)
    """
    T = pd.Series(T, index=X.index) if not isinstance(T, pd.Series) else T
    m1 = X[T == 1].mean()
    m0 = X[T == 0].mean()
    s1 = X[T == 1].std()
    s0 = X[T == 0].std()
    pooled_sd = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
    return (m1 - m0) / pooled_sd


def evaluate_balance(
    X_matched: np.ndarray | pd.DataFrame,
    T_matched: np.ndarray | pd.Series,
    col_names: pd.Index | list[str],
) -> BalanceResult:
    """
    Evaluate covariate balance after matching.

    Parameters
    ----------
    X_matched  : matched covariate matrix (numpy array or DataFrame)
    T_matched  : matched treatment vector
    col_names  : column names for X_matched (used when X_matched is ndarray)

    Returns
    -------
    BalanceResult
    """
    X_df = pd.DataFrame(X_matched, columns=col_names)
    T_s  = pd.Series(np.asarray(T_matched).squeeze())

    smd = compute_smd(X_df, T_s)
    abs_smd = smd.abs()

    mean_smd = float(np.nanmean(abs_smd))
    n_exceed = int((abs_smd > SMD_THRESHOLD).sum())

    # Count unique treated/control rows for matched-cohort size reporting
    try:
        n_treated = X_df[T_s == 1].drop_duplicates().shape[0]
        n_control = X_df[T_s == 0].drop_duplicates().shape[0]
    except Exception:
        n_treated = int((T_s == 1).sum())
        n_control = int((T_s == 0).sum())

    return BalanceResult(
        mean_smd=mean_smd,
        n_exceeding_threshold=n_exceed,
        n_treated=n_treated,
        n_control=n_control,
        smd_per_covariate=smd,
    )


def _ratio_score(n_treated: int, n_control: int, n_exceed: int, mean_smd: float) -> float:
    """
    Composite score used for ranking matching configurations:
    higher is better  (more patients, fewer imbalanced covariates).
    """
    total = n_treated + n_control
    return round(total / (n_exceed + mean_smd + 1), 0)


# ---------------------------------------------------------------------------
# Algorithm 3 — k:1 PSM with caliper  (Dimitsaki et al. 2026)
# ---------------------------------------------------------------------------

def psm_caliper_k1(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    propensity_scores: np.ndarray,
    caliper_coef: float = PSM_CALIPER,
    k: int = PSM_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[int, int, int]:
    """
    k:1 nearest-neighbour PSM with caliper (Algorithm 3).

    Steps
    -----
    1. Compute logit of PS to stabilise distances.
    2. Set caliper = caliper_coef × SD(logit PS).
    3. For each treated unit find the k nearest controls (no replacement).
    4. Accept match only if |logit_ps_treated − logit_ps_control| ≤ caliper.

    Returns
    -------
    (matched_X, matched_T, matched_Y)  or  (0, 0, 0) if no matches found.
    """
    eps = 1e-6
    logit_ps = np.log(propensity_scores / (1 - propensity_scores + eps))
    caliper  = caliper_coef * np.std(logit_ps)

    treated = np.where(T == 1)[0]
    control = np.where(T == 0)[0]

    treated_ps = propensity_scores[treated].reshape(-1, 1)
    control_ps = propensity_scores[control].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=k).fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)

    matched_pairs: list[tuple[int, int]] = []
    used_controls: set[int] = set()

    for i, treated_idx in enumerate(treated):
        for j in range(k):
            ctrl_candidate = control[indices[i][j]]
            if ctrl_candidate in used_controls:
                continue
            if abs(logit_ps[treated_idx] - logit_ps[ctrl_candidate]) <= caliper:
                matched_pairs.append((treated_idx, ctrl_candidate))
                used_controls.add(ctrl_candidate)
                break  # take the first valid match; move on to next treated

    if not matched_pairs:
        return 0, 0, 0

    matched_treated = np.array([p[0] for p in matched_pairs])
    matched_control = np.array([p[1] for p in matched_pairs])

    matched_X = np.vstack([X[matched_treated], X[matched_control]])
    matched_T = np.concatenate([
        np.ones(len(matched_treated)),
        np.zeros(len(matched_control)),
    ])
    matched_Y = np.concatenate([Y[matched_treated], Y[matched_control]])

    return matched_X, matched_T, matched_Y


# ---------------------------------------------------------------------------
# Propensity-score classifiers used in the benchmark
# ---------------------------------------------------------------------------

def _get_ps_classifiers() -> dict:
    return {
        "Logistic Regression":       LogisticRegression(max_iter=1000),
        "Random Forest":             RandomForestClassifier(),
        "Gradient Boosting":         GradientBoostingClassifier(),
        "Support Vector Classifier": SVC(probability=True),
        "Multilayer Perceptron":     MLPClassifier(max_iter=500),
        "AdaBoost":                  AdaBoostClassifier(),
        "Decision Tree":             DecisionTreeClassifier(),
        "XGBoost":                   XGBClassifier(eval_metric="logloss", verbosity=0),
    }


# ---------------------------------------------------------------------------
# Full matching benchmark: PSM × classifiers + alternative methods
# ---------------------------------------------------------------------------

def apply_all_matching_methods(
    X: pd.DataFrame,
    T: pd.Series,
    Y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every matching strategy and evaluate covariate balance.

    Strategies
    ----------
    * PSM with each classifier in _get_ps_classifiers()
    * Nearest-Neighbour Matching (on raw covariates)
    * Mahalanobis Distance Matching
    * Exact Matching

    Parameters
    ----------
    X : pd.DataFrame  — covariate matrix (already preprocessed / normalised)
    T : pd.Series     — treatment indicator (1 = treated, 0 = control)
    Y : np.ndarray    — outcome vector

    Returns
    -------
    summary_df : pd.DataFrame — one row per strategy, balance metrics
    smd_df     : pd.DataFrame — per-covariate SMD series for each strategy
    """
    X_arr = np.array(X).astype("float32")
    T_arr = np.array(T).astype("float32").squeeze()
    Y_arr = np.array(Y).astype("float32").squeeze()

    # Pre-compute initial (unmatched) mean SMD for reference
    smd_before   = compute_smd(X, T).abs()
    mean_smd_bf  = round(float(np.mean(smd_before)), 2)

    records: list[dict]         = []
    smd_records: list[pd.Series] = []

    # ------------------------------------------------------------------ PSM
    for clf_name, clf in _get_ps_classifiers().items():
        try:
            clf.fit(X, T)
            ps = clf.predict_proba(X)[:, 1]
            mx, mt, my = psm_caliper_k1(X_arr, T_arr, Y_arr, ps)
            if not isinstance(mx, np.ndarray):
                continue
            bal = evaluate_balance(mx, mt, X.columns)
            ratio = _ratio_score(bal.n_treated, bal.n_control,
                                 bal.n_exceeding_threshold, bal.mean_smd)
            records.append({
                "Methodology":        f"PSM {clf_name}",
                "Initial SMD":        mean_smd_bf,
                "Mean SMD":           round(bal.mean_smd, 2),
                "SMD > 0.2":          bal.n_exceeding_threshold,
                "Matched patients 1": bal.n_treated,
                "Matched patients 0": bal.n_control,
                "Total patients":     bal.n_treated + bal.n_control,
                "Ratio":              ratio,
            })
            smd_records.append(bal.smd_per_covariate.rename(f"PSM {clf_name}"))
        except Exception as exc:
            logger.warning("PSM %s failed: %s", clf_name, exc)

    # ------------------------------------------------ Nearest-Neighbour (NN)
    try:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X[T == 0])
        _, idx = nn.kneighbors(X[T == 1])
        ctrl_idx = X.index[T == 0][idx.flatten()]
        X_nn = pd.concat([X.loc[T == 1], X.loc[ctrl_idx]])
        T_nn = pd.concat([T.loc[T == 1], T.loc[ctrl_idx]])
        bal  = evaluate_balance(X_nn, T_nn, X.columns)
        ratio = _ratio_score(bal.n_treated, bal.n_control,
                             bal.n_exceeding_threshold, bal.mean_smd)
        records.append({
            "Methodology":        "Nearest Neighbour Matching",
            "Initial SMD":        mean_smd_bf,
            "Mean SMD":           round(bal.mean_smd, 2),
            "SMD > 0.2":          bal.n_exceeding_threshold,
            "Matched patients 1": bal.n_treated,
            "Matched patients 0": bal.n_control,
            "Total patients":     bal.n_treated + bal.n_control,
            "Ratio":              ratio,
        })
        smd_records.append(bal.smd_per_covariate.rename("Nearest Neighbour Matching"))
    except Exception as exc:
        logger.warning("Nearest Neighbour Matching failed: %s", exc)

    # ----------------------------------------- Mahalanobis Distance Matching
    try:
        treated_arr = X[T == 1].to_numpy()
        control_arr = X[T == 0].to_numpy()
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)
        dists = cdist(treated_arr, control_arr, metric="mahalanobis", VI=inv_cov)
        nearest = np.argmin(dists, axis=1)
        ctrl_matched = control_arr[nearest]
        X_mah = pd.DataFrame(
            np.vstack([treated_arr, ctrl_matched]), columns=X.columns
        )
        T_mah = pd.Series(
            np.concatenate([np.ones(len(treated_arr)),
                            np.zeros(len(ctrl_matched))])
        )
        bal   = evaluate_balance(X_mah, T_mah, X.columns)
        ratio = _ratio_score(bal.n_treated, bal.n_control,
                             bal.n_exceeding_threshold, bal.mean_smd)
        records.append({
            "Methodology":        "Mahalanobis Distance Matching",
            "Initial SMD":        mean_smd_bf,
            "Mean SMD":           round(bal.mean_smd, 2),
            "SMD > 0.2":          bal.n_exceeding_threshold,
            "Matched patients 1": bal.n_treated,
            "Matched patients 0": bal.n_control,
            "Total patients":     bal.n_treated + bal.n_control,
            "Ratio":              ratio,
        })
        smd_records.append(bal.smd_per_covariate.rename("Mahalanobis Distance Matching"))
    except Exception as exc:
        logger.warning("Mahalanobis Distance Matching failed: %s", exc)

    # --------------------------------------------------------- Exact Matching
    try:
        tmp = X.copy()
        tmp["_T"] = T.values
        matched_exact = tmp.groupby(list(X.columns)).filter(
            lambda g: g["_T"].nunique() > 1
        )
        T_ex = matched_exact["_T"]
        X_ex = matched_exact.drop(columns=["_T"])
        bal  = evaluate_balance(X_ex, T_ex, X.columns)
        ratio = _ratio_score(bal.n_treated, bal.n_control,
                             bal.n_exceeding_threshold, bal.mean_smd)
        records.append({
            "Methodology":        "Exact Matching",
            "Initial SMD":        mean_smd_bf,
            "Mean SMD":           round(bal.mean_smd, 2),
            "SMD > 0.2":          bal.n_exceeding_threshold,
            "Matched patients 1": bal.n_treated,
            "Matched patients 0": bal.n_control,
            "Total patients":     bal.n_treated + bal.n_control,
            "Ratio":              ratio,
        })
        smd_records.append(bal.smd_per_covariate.rename("Exact Matching"))
    except Exception as exc:
        logger.warning("Exact Matching failed: %s", exc)

    # ---------------------------------------------------------------- Assemble
    summary_cols = [
        "Methodology", "Initial SMD", "Mean SMD", "SMD > 0.2",
        "Matched patients 1", "Matched patients 0", "Total patients", "Ratio",
    ]
    summary_df = pd.DataFrame(records)[summary_cols] if records else pd.DataFrame(columns=summary_cols)
    smd_df     = pd.concat(smd_records, axis=1) if smd_records else pd.DataFrame()

    return summary_df, smd_df


# ---------------------------------------------------------------------------
# Love Plot (covariate balance visualisation)
# ---------------------------------------------------------------------------

def plot_love_plot(
    X: pd.DataFrame,
    T: pd.Series,
    X_matched: np.ndarray | pd.DataFrame,
    T_matched: np.ndarray | pd.Series,
    title: str = "Covariate Balance (Love Plot)",
    ax=None,
) -> None:
    """
    Plot SMD before and after matching for all covariates.

    Parameters
    ----------
    X, T           : unmatched covariate matrix and treatment vector
    X_matched, T_matched : matched equivalents
    title          : plot title
    ax             : optional matplotlib Axes; creates a new figure if None
    """
    import matplotlib.pyplot as plt

    X_m  = pd.DataFrame(X_matched, columns=X.columns) if not isinstance(X_matched, pd.DataFrame) else X_matched
    T_m  = pd.Series(np.asarray(T_matched).squeeze())

    smd_before = compute_smd(X, T).abs()
    smd_after  = compute_smd(X_m, T_m).abs()
    covariates = X.columns
    y_pos      = np.arange(len(covariates))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(6, len(covariates) * 0.35)))

    ax.axvline(0.1, color="gray",  linestyle="--", linewidth=0.8, label="SMD = 0.1")
    ax.axvline(0.2, color="red",   linestyle="--", linewidth=0.8, label="SMD = 0.2")
    ax.scatter(smd_before, y_pos, color="steelblue", label="Before Matching", zorder=3)
    ax.scatter(smd_after,  y_pos, color="seagreen",  label="After Matching",  zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontsize=8)
    ax.set_xlabel("Absolute Standardised Mean Difference (SMD)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
