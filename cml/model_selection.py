"""
cml/model_selection.py
----------------------
Selects the best ML model for each CML architecture (S/T/X/DML learners)
by evaluating all classifier/regressor combinations with cross-validation.

Replaces Select_best_ML_models_for_CML_Simethicone.ipynb.

Public API
----------
evaluate_s_learner(X, y, t, n_splits)   -> pd.DataFrame
evaluate_t_learner(X, y, t)             -> pd.DataFrame
evaluate_x_learner(X, y, t, n_jobs)     -> pd.DataFrame
evaluate_dml_learner(X, y, t, n_jobs)   -> pd.DataFrame
run_model_selection(X, y, t, ...)       -> dict[str, pd.DataFrame]
save_model_selection_results(...)       -> None
"""

from __future__ import annotations

import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import KFold

from config import (
    CHARLSON_COLS, CI_WIDTH_THRESHOLD, N_BOOTSTRAP,
    composite_regression_score,
    get_classifiers, get_regressors,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Return (accuracy, precision, recall, f1) for binary predictions."""
    bt = (np.asarray(y_true) > 0.5).astype(int)
    bp = (np.asarray(y_pred) > 0.5).astype(int)
    return (
        accuracy_score(bt, bp),
        precision_score(bt, bp, average="macro", zero_division=0),
        recall_score(bt, bp, average="macro", zero_division=0),
        f1_score(bt, bp, average="macro", zero_division=0),
    )


def _avg_metrics(*metric_tuples) -> tuple[float, float, float, float]:
    """Average (acc, prec, rec, f1) across multiple (acc, prec, rec, f1) tuples."""
    arrs = np.array(metric_tuples)  # shape (n, 4)
    return tuple(arrs.mean(axis=0))


# ===========================================================================
# S-Learner model selection  (5-fold CV)
# ===========================================================================

def evaluate_s_learner(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Evaluate every classifier as the S-Learner overall model.

    The S-Learner augments X with the treatment indicator and trains a
    single model.  Performance is measured by average classification
    metrics over k-fold CV.

    Returns a ranked DataFrame (by F1 score, descending).
    """
    classifiers = get_classifiers()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for name, clf in classifiers.items():
        fold_accs, fold_precs, fold_recs, fold_f1s, fold_rrs = [], [], [], [], []

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            t_tr, t_te = t[train_idx], t[test_idx]

            X_aug_tr = np.hstack([X_tr, t_tr.reshape(-1, 1)])
            try:
                clf_copy = clone(clf)
                clf_copy.fit(X_aug_tr, y_tr)

                pred_t1 = clf_copy.predict(np.hstack([X_te, np.ones((len(X_te), 1))]))
                pred_t0 = clf_copy.predict(np.hstack([X_te, np.zeros((len(X_te), 1))]))

                m1 = _binary_metrics(y_te, pred_t1)
                m0 = _binary_metrics(y_te, pred_t0)
                avg = _avg_metrics(m1, m0)

                fold_accs.append(avg[0])
                fold_precs.append(avg[1])
                fold_recs.append(avg[2])
                fold_f1s.append(avg[3])

                rr = float(np.mean(pred_t1)) / float(np.mean(pred_t0) + 1e-9)
                fold_rrs.append(rr)
            except Exception as exc:
                logger.warning("S-Learner CV fold failed for %s: %s", name, exc)

        if fold_f1s:
            records.append({
                "Model": name,
                "Accuracy":  round(float(np.mean(fold_accs)),  3),
                "Precision": round(float(np.mean(fold_precs)), 3),
                "Recall":    round(float(np.mean(fold_recs)),  3),
                "F1":        round(float(np.mean(fold_f1s)),   3),
                "Causal RR": round(float(np.mean(fold_rrs)),   3),
            })

    df = pd.DataFrame(records)
    return df.sort_values("F1", ascending=False).reset_index(drop=True)


# ===========================================================================
# T-Learner model selection
# ===========================================================================

def evaluate_t_learner(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all pairs of classifiers as T-Learner (model_treated, model_control).

    Returns a ranked DataFrame (by average F1, descending).
    """
    classifiers = get_classifiers()
    X_tr1, y_tr1 = X[t == 1], y[t == 1]
    X_tr0, y_tr0 = X[t == 0], y[t == 0]
    records = []

    for name1, clf1 in classifiers.items():
        for name2, clf2 in classifiers.items():
            try:
                c1 = clone(clf1); c2 = clone(clf2)
                c1.fit(X_tr1, y_tr1)
                c2.fit(X_tr0, y_tr0)

                pred_t1 = c1.predict(X)
                pred_t0 = c2.predict(X)

                m1 = _binary_metrics(y, pred_t1)
                m0 = _binary_metrics(y, pred_t0)
                avg = _avg_metrics(m1, m0)

                # Which sub-model performed worse? (flag for selection)
                score1 = float(np.mean(m1))
                score0 = float(np.mean(m0))
                weaker = name1 if score1 < score0 else name2

                records.append({
                    "Model":        f"{name1} vs {name2}",
                    "Accuracy":     round(avg[0], 3),
                    "Precision":    round(avg[1], 3),
                    "Recall":       round(avg[2], 3),
                    "F1":           round(avg[3], 3),
                    "Weaker model": weaker,
                    "Causal RR":    round(float(np.mean(pred_t1)) /
                                         float(np.mean(pred_t0) + 1e-9), 3),
                })
            except Exception as exc:
                logger.warning("T-Learner failed (%s vs %s): %s", name1, name2, exc)

    df = pd.DataFrame(records)
    return df.sort_values("F1", ascending=False).reset_index(drop=True)


# ===========================================================================
# X-Learner model selection  (parallelised)
# ===========================================================================

def _evaluate_x_pair(
    clf_name1: str, clf1,
    clf_name2: str, clf2,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> list[dict]:
    """Evaluate one (clf1, clf2) first-stage pair across all regressor pairs."""
    regressors = get_regressors()
    X_tr1, y_tr1 = X[t == 1], y[t == 1]
    X_tr0, y_tr0 = X[t == 0], y[t == 0]

    c1 = clone(clf1); c2 = clone(clf2)
    c1.fit(X_tr1, y_tr1)
    c2.fit(X_tr0, y_tr0)

    pred_t1 = c1.predict(X)
    pred_t0 = c2.predict(X)

    m1 = _binary_metrics(y, pred_t1)
    m0 = _binary_metrics(y, pred_t0)
    acc  = (m1[0] + m0[0]) / 2
    prec = (m1[1] + m0[1]) / 2
    rec  = (m1[2] + m0[2]) / 2
    f1   = (m1[3] + m0[3]) / 2

    d1 = y_tr1 - c2.predict(X_tr1)    # imputed treatment effect (treated group)
    d0 = c1.predict(X_tr0) - y_tr0    # imputed treatment effect (control group)

    results = []
    for reg_name3, reg3 in regressors.items():
        for reg_name4, reg4 in regressors.items():
            try:
                r3 = clone(reg3); r4 = clone(reg4)
                r3.fit(X_tr0, d0)
                r4.fit(X_tr1, d1)

                p3 = r3.predict(X)
                p4 = r4.predict(X)

                mse3 = mean_squared_error(y, p3)
                mse4 = mean_squared_error(y, p4)
                r2_3 = r2_score(y, p3)
                r2_4 = r2_score(y, p4)

                rmse3 = float(np.sqrt(mse3))
                rmse4 = float(np.sqrt(mse4))

                s3 = composite_regression_score(mse3, rmse3, r2_3)
                s4 = composite_regression_score(mse4, rmse4, r2_4)
                overall_score = round((acc + prec + rec + f1 + (r2_3 + r2_4) / 2) / 5, 3)

                results.append({
                    "Classifiers":  f"{clf_name1} vs {clf_name2}",
                    "Regressors":   f"{reg_name3} vs {reg_name4}",
                    "Accuracy":     round(acc,  3),
                    "Precision":    round(prec, 3),
                    "Recall":       round(rec,  3),
                    "F1":           round(f1,   3),
                    "MSE":          round((mse3 + mse4) / 2, 4),
                    "RMSE":         round((rmse3 + rmse4) / 2, 4),
                    "R2":           round((r2_3 + r2_4) / 2, 4),
                    "S_score":      round((s3 + s4) / 2, 4),
                    "Overall":      overall_score,
                })
            except Exception:
                pass
    return results


def evaluate_x_learner(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Evaluate all combinations of (classifier_1, classifier_2, regressor_1, regressor_2)
    for the X-Learner architecture.  Parallelised with joblib.

    Returns a ranked DataFrame (by Overall score, descending).
    """
    classifiers = get_classifiers()
    clf_pairs = list(itertools.product(classifiers.items(), classifiers.items()))

    all_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_evaluate_x_pair)(n1, c1, n2, c2, X, y, t)
        for (n1, c1), (n2, c2) in clf_pairs
    )
    flat = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(flat)
    return df.sort_values("Overall", ascending=False).reset_index(drop=True)


# ===========================================================================
# DML model selection  (parallelised)
# ===========================================================================

def _evaluate_dml_single(
    model_y_name: str, model_y,
    model_t_name: str, model_t,
    model_final_name: str, model_final,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
) -> dict:
    """
    Evaluate one (model_y, model_t, model_final) triplet via DML cross-fitting.
    Returns a metrics dict; sets Error key on failure.
    """
    from sklearn.base import clone as sk_clone

    n = X.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_resid = np.zeros(n)
    t_resid = np.zeros(n)
    y_true_all, y_pred_all = [], []
    t_true_all, t_pred_all = [], []

    try:
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            T_tr, T_te = T[tr_idx], T[te_idx]

            my = sk_clone(model_y).fit(X_tr, Y_tr)
            mt = sk_clone(model_t).fit(X_tr, T_tr)

            yp = my.predict(X_te)
            tp = mt.predict(X_te)

            y_resid[te_idx] = Y_te - yp
            t_resid[te_idx] = T_te - tp

            y_true_all.extend(Y_te);  y_pred_all.extend(yp)
            t_true_all.extend(T_te);  t_pred_all.extend(tp)

        X_final = np.column_stack([t_resid, X])
        mf = sk_clone(model_final).fit(X_final, y_resid)
        yf = mf.predict(X_final)

        mse_y  = mean_squared_error(y_true_all, y_pred_all)
        rmse_y = float(np.sqrt(mse_y))
        r2_y   = r2_score(y_true_all, y_pred_all)

        mse_t  = mean_squared_error(t_true_all, t_pred_all)
        rmse_t = float(np.sqrt(mse_t))
        r2_t   = r2_score(t_true_all, t_pred_all)

        mse_f  = mean_squared_error(y_resid, yf)
        rmse_f = float(np.sqrt(mse_f))
        r2_f   = r2_score(y_resid, yf)

        return {
            "Outcome Model":   model_y_name,
            "Treatment Model": model_t_name,
            "Final Model":     model_final_name,
            "Y_MSE": round(mse_y, 4),  "Y_RMSE": round(rmse_y, 4),  "Y_R2": round(r2_y, 4),
            "T_MSE": round(mse_t, 4),  "T_RMSE": round(rmse_t, 4),  "T_R2": round(r2_t, 4),
            "F_MSE": round(mse_f, 4),  "F_RMSE": round(rmse_f, 4),  "F_R2": round(r2_f, 4),
            "S_Y": round(composite_regression_score(mse_y, rmse_y, r2_y), 4),
            "S_T": round(composite_regression_score(mse_t, rmse_t, r2_t), 4),
            "S_F": round(composite_regression_score(mse_f, rmse_f, r2_f), 4),
            "Error": "",
        }
    except Exception as exc:
        return {
            "Outcome Model":   model_y_name,
            "Treatment Model": model_t_name,
            "Final Model":     model_final_name,
            "Error": str(exc),
        }


def evaluate_dml_learner(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_jobs: int = -1,
    n_splits: int = 2,
) -> pd.DataFrame:
    """
    Evaluate all triplets of (model_y, model_t, model_final) for the DML learner.

    Returns a DataFrame ranked by composite score S_F (Final Model).
    """
    regressors = get_regressors()
    combos = list(itertools.product(regressors.items(), repeat=3))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_evaluate_dml_single)(ny, ry, nt, rt, nf, rf, X, t, y, n_splits)
        for (ny, ry), (nt, rt), (nf, rf) in combos
    )

    df = pd.DataFrame(results)
    if "S_F" in df.columns:
        df = df.sort_values("S_F", ascending=False)
    return df.reset_index(drop=True)


# ===========================================================================
# Master runner
# ===========================================================================

def run_model_selection(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_splits_s: int = 5,
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """
    Run model selection for all four CML architectures and return results.

    Returns
    -------
    dict with keys: "S-Learner", "T-Learner", "X-Learner", "DML-Learner"
    """
    logger.info("Running S-Learner model selection …")
    s_df = evaluate_s_learner(X, y, t, n_splits=n_splits_s)

    logger.info("Running T-Learner model selection …")
    t_df = evaluate_t_learner(X, y, t)

    logger.info("Running X-Learner model selection …")
    x_df = evaluate_x_learner(X, y, t, n_jobs=n_jobs)

    logger.info("Running DML model selection …")
    dml_df = evaluate_dml_learner(X, y, t, n_jobs=n_jobs)

    return {
        "S-Learner":   s_df,
        "T-Learner":   t_df,
        "X-Learner":   x_df,
        "DML-Learner": dml_df,
    }


def save_model_selection_results(
    results: dict[str, pd.DataFrame],
    out_path: str | Path,
) -> None:
    """Save all model-selection results to one Excel workbook, one sheet per architecture."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Model-selection results saved to %s", out_path)
