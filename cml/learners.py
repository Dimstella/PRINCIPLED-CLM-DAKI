"""
cml/learners.py
---------------
Thin wrappers around EconML metalearners and DML, providing a uniform
interface that returns (ATE, CI_ATE) for all four CML architectures.

All functions use bootstrap inference (n=100) as specified in the paper.

Public API
----------
s_learner(clf, X, y, t)                          -> (ate, ci_ate)
t_learner(clf_treat, clf_control, X, y, t)       -> (ate, ci_ate)
x_learner(clf_treat, clf_ctrl, X, y, t,
          clf_prop, reg_treat, reg_ctrl)          -> (ate, ci_ate)
dml_learner(model_y, model_t, model_final,
            X, y, t)                              -> (ate, ci_ate)
"""

from __future__ import annotations

import warnings
import numpy as np

from econml.metalearners import SLearner, TLearner, XLearner
from econml.dml import DML

from config import N_BOOTSTRAP

warnings.filterwarnings("ignore")


def s_learner(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, list]:
    """
    S-Learner: single model trained on (X, T) augmented features.

    Parameters
    ----------
    clf        : sklearn-compatible classifier
    X          : covariate matrix
    y          : binary outcome
    t          : treatment indicator
    n_bootstrap: number of bootstrap samples for CI

    Returns
    -------
    (ate, [ci_lower, ci_upper])
    """
    learner = SLearner(overall_model=clf)
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    ate = float(learner.ate(X))
    ci_lo, ci_hi = learner.ate_interval(X)
    return ate, [float(ci_lo), float(ci_hi)]


def t_learner(
    clf_treat,
    clf_control,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, list]:
    """
    T-Learner: separate model per treatment arm.

    Parameters
    ----------
    clf_treat   : classifier for the treated group
    clf_control : classifier for the control group

    Returns
    -------
    (ate, [ci_lower, ci_upper])
    """
    learner = TLearner(models=[clf_treat, clf_control])
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    ate = float(learner.ate(X))
    ci_lo, ci_hi = learner.ate_interval(X)
    return ate, [float(ci_lo), float(ci_hi)]


def x_learner(
    clf_treat,
    clf_control,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    clf_propensity,
    reg_treat,
    reg_control,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, list]:
    """
    X-Learner: two-stage learner with propensity score weighting.

    Parameters
    ----------
    clf_treat     : first-stage classifier for treated group
    clf_control   : first-stage classifier for control group
    clf_propensity: propensity score model
    reg_treat     : second-stage regressor for treated group CATE
    reg_control   : second-stage regressor for control group CATE

    Returns
    -------
    (ate, [ci_lower, ci_upper])
    """
    learner = XLearner(
        models=[clf_treat, clf_control],
        propensity_model=clf_propensity,
        cate_models=[reg_treat, reg_control],
    )
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    ate = float(learner.ate(X))
    ci_lo, ci_hi = learner.ate_interval(X)
    return ate, [float(ci_lo), float(ci_hi)]


def dml_learner(
    model_y,
    model_t,
    model_final,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, list]:
    """
    Double Machine Learning learner.

    Parameters
    ----------
    model_y    : nuisance model for the outcome
    model_t    : nuisance model for the treatment
    model_final: final stage regression model

    Returns
    -------
    (ate, [ci_lower, ci_upper])
    """
    learner = DML(
        model_t=model_t,
        model_y=model_y,
        model_final=model_final,
        discrete_outcome=True,
        discrete_treatment=True,
    )
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    ate = float(learner.ate(X))
    ci_lo, ci_hi = learner.ate_interval(X)
    return ate, [float(ci_lo), float(ci_hi)]


def run_meta_learners(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    s_models: list | None = None,
    t_models: list | None = None,
    x_models: list | None = None,
) -> "pd.DataFrame":
    """
    Run S-, T-, and X-Learner with specified models and return a results table.

    Parameters
    ----------
    s_models : [overall_clf]
    t_models : [clf_treated, clf_control]
    x_models : [clf_treat, clf_control, clf_propensity, reg_treat, reg_control]

    Returns
    -------
    pd.DataFrame with columns: Meta-Learner | Model | ATE | CI ATE
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC, SVR

    if s_models is None:
        s_models = [RandomForestClassifier()]
    if t_models is None:
        t_models = [RandomForestClassifier(), RandomForestClassifier()]
    if x_models is None:
        x_models = [
            RandomForestClassifier(), RandomForestClassifier(),
            LogisticRegression(max_iter=1000), SVR(), SVR(),
        ]

    results = []

    # S-Learner
    ate, ci = s_learner(s_models[0], X, y, t)
    results.append({
        "Meta-Learner": "S-Learner",
        "Model": s_models[0].__class__.__name__,
        "ATE": round(ate, 5),
        "CI ATE": ci,
    })

    # T-Learner
    ate, ci = t_learner(t_models[0], t_models[1], X, y, t)
    results.append({
        "Meta-Learner": "T-Learner",
        "Model": f"{t_models[0].__class__.__name__} vs {t_models[1].__class__.__name__}",
        "ATE": round(ate, 5),
        "CI ATE": ci,
    })

    # X-Learner
    ate, ci = x_learner(
        x_models[0], x_models[1], X, y, t,
        x_models[2], x_models[3], x_models[4],
    )
    results.append({
        "Meta-Learner": "X-Learner",
        "Model": (
            f"{x_models[0].__class__.__name__} vs "
            f"{x_models[1].__class__.__name__} vs "
            f"{x_models[2].__class__.__name__} vs "
            f"{x_models[3].__class__.__name__}"
        ),
        "ATE": round(ate, 5),
        "CI ATE": ci,
    })

    return pd.DataFrame(results)


def run_dml_learner(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    dml_models: list | None = None,
) -> "pd.DataFrame":
    """
    Run the DML learner with specified models.

    Parameters
    ----------
    dml_models : [model_y, model_t, model_final]

    Returns
    -------
    pd.DataFrame with columns: Meta-Learner | Model | ATE | CI ATE
    """
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    if dml_models is None:
        dml_models = [
            GradientBoostingRegressor(),
            RandomForestRegressor(),
            DecisionTreeRegressor(),
        ]

    ate, ci = dml_learner(dml_models[0], dml_models[1], dml_models[2], X, y, t)
    return pd.DataFrame([{
        "Meta-Learner": "DML-Learner",
        "Model": (
            f"{dml_models[0].__class__.__name__} vs "
            f"{dml_models[1].__class__.__name__} vs "
            f"{dml_models[2].__class__.__name__}"
        ),
        "ATE": round(ate, 5),
        "CI ATE": ci,
    }])
