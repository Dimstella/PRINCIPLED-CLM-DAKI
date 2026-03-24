"""
cml/config.py
-------------
Central registry for the CML pipeline.

Contains:
  - Classifier / regressor registries used in model selection
  - Per-drug best-model configurations (from Tables 12-15 in the paper)
  - Shared feature-column groups
  - Composite score formula (Equation 1) for regression model ranking
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sklearn / XGBoost imports — kept here so every other module only needs
# `from config import get_classifiers, get_regressors, ...`
# ---------------------------------------------------------------------------
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    SGDClassifier, SGDRegressor, Ridge,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Model factories (callable so each call returns a fresh instance)
# ---------------------------------------------------------------------------

def get_classifiers() -> dict:
    """Return the classifier registry used across S/T/X learners."""
    return {
        "Logistic Regression":       LogisticRegression(max_iter=1000),
        "Linear Regression":         LinearRegression(),
        "SGD Classifier":            SGDClassifier(loss="log_loss"),
        "Support Vector Classifier": SVC(probability=True),
        "Decision Tree":             DecisionTreeClassifier(),
        "Random Forest":             RandomForestClassifier(),
        "XGBoost":                   XGBClassifier(n_jobs=1, verbosity=0, eval_metric="logloss"),
        "MLP Classifier":            MLPClassifier(max_iter=500),
    }


def get_regressors() -> dict:
    """Return the regressor registry used in X-Learner stage 2 and DML."""
    return {
        "Ridge Regression":       Ridge(alpha=1.0),
        "SGD Regressor":          SGDRegressor(),
        "K-Nearest Neighbors":    KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
        "Decision Tree":          DecisionTreeRegressor(),
        "Random Forest":          RandomForestRegressor(),
        "Gradient Boosting":      GradientBoostingRegressor(),
        "AdaBoost":               AdaBoostRegressor(),
        "Extra Trees":            ExtraTreesRegressor(),
        "MLP Regressor":          MLPRegressor(max_iter=1000),
        "XGBoost":                XGBRegressor(n_jobs=1, verbosity=0),
    }


# ---------------------------------------------------------------------------
# Composite regression score  (Equation 1 in the paper)
#
#   S = 1 / (log(10 + MSE) + log(10 + RMSE) − 1) + R²
#
# Higher is better.
# ---------------------------------------------------------------------------
import numpy as np

def composite_regression_score(mse: float, rmse: float, r2: float) -> float:
    return 1.0 / (np.log(10 + mse) + np.log(10 + rmse) - 1) + r2


# ---------------------------------------------------------------------------
# Shared feature-column groups
# ---------------------------------------------------------------------------

LAB_COLS_MEAN = [
    "Big_bp", "Small_bp", "glucose", "sodium", "creatinine",
    "potassium", "bun", "bicarbonate", "chloride", "aniongap",
]

LAB_SUFFIXES_T0T1 = [
    "aniongap_first", "aniongap_last",
    "bicarbonate_first", "bicarbonate_last",
    "bun_first", "bun_last",
    "chloride_first", "chloride_last",
    "creatinine_first", "creatinine_last",
    "glucose_first", "glucose_last",
    "potassium_first", "potassium_last",
    "sodium_first", "sodium_last",
]

LAB_COLS_T0T1 = ["Big_bp", "Small_bp"] + LAB_SUFFIXES_T0T1

DEM_COLS = ["Weight (Lbs)", "gender", "anchor_age"]

CHARLSON_COLS = [
    "congestive_heart_failure", "renal_disease", "diabetes_without_cc",
    "chronic_pulmonary_disease", "myocardial_infarct", "diabetes_with_cc",
    "peripheral_vascular_disease", "cerebrovascular_disease",
    "malignant_cancer", "mild_liver_disease", "metastatic_solid_tumor",
    "rheumatic_disease", "peptic_ulcer_disease", "dementia",
    "paraplegia", "severe_liver_disease", "aids",
]

SCALE_COLS_MEAN = [
    "glucose", "Big_bp", "bicarbonate", "bun", "BMI (kg/m2)",
    "Weight (Lbs)", "chloride", "sodium", "potassium",
    "Small_bp", "creatinine", "aniongap",
]

SCALE_COLS_T0T1 = [
    "Big_bp", "BMI (kg/m2)", "Weight (Lbs)", "Small_bp",
] + LAB_SUFFIXES_T0T1

NO_BINARISE_COLS = set(SCALE_COLS_MEAN + SCALE_COLS_T0T1 + ["anchor_age"])
DROP_ALWAYS = ["charlson_comorbidity_index", "BMI (kg/m2)"]
ID_COLS = ["y", "t", "subject_id"]

CI_WIDTH_THRESHOLD = 0.85   # CATEs with CI width >= this are non-robust (paper §2.5.3)
N_BOOTSTRAP = 100            # bootstrap samples for CI estimation

# ---------------------------------------------------------------------------
# Per-drug confounder DAG configuration (shared with matching/config.py)
# Repeated here for standalone use of the CML module.
# ---------------------------------------------------------------------------

DRUG_DAG_CONFIG = {
    "ibuprofen": {
        "drop_own_atc": [],
        "dags": {
            "Literature": {
                "drug_families": [
                    "ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS",
                    "ANTIHYPERTENSIVES", "CONTRAST MEDIA",
                    "DRUGS USED IN DIABETES", "DIURETICS",
                    "CORTICOSTEROIDS FOR SYSTEMIC USE",
                    "TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS",
                    "CONTRAST MEDIA", "ANTIHYPERTENSIVES", "DIURETICS",
                    "CORTICOSTEROIDS FOR SYSTEMIC USE",
                    "TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "ketorolac": {
        "drop_own_atc": [],
        "dags": {
            "Literature": {
                "drug_families": ["CONTRAST MEDIA", "ANTIHYPERTENSIVES",
                                  "DIURETICS", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "glucose", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["CONTRAST MEDIA", "ANTIHYPERTENSIVES",
                                  "DIURETICS", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "sodium", "pre_AKI"],
            },
        },
    },
    "vancomycin": {
        "drop_own_atc": [],
        "dags": {
            "Literature": {
                "drug_families": ["CONTRAST MEDIA", "ANTIHYPERTENSIVES", "DIURETICS",
                                  "VACCINES", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["glucose", "Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["CONTRAST MEDIA", "ANTIHYPERTENSIVES", "DIURETICS",
                                  "VACCINES", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["glucose", "Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "lisinopril": {
        "drop_own_atc": [],
        "dags": {
            "Literature": {
                "drug_families": ["DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                                  "ANTIEPILEPTICS", "DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "furosemide": {
        "drop_own_atc": ["C03CA01"],
        "dags": {
            "Literature": {
                "drug_families": ["DIURETICS", "DRUGS FOR ACID RELATED DISORDERS",
                                  "CONTRAST MEDIA", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["DIURETICS", "DRUGS FOR ACID RELATED DISORDERS",
                                  "CONTRAST MEDIA", "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "pantoprazole": {
        "drop_own_atc": ["A02BC02"],
        "dags": {
            "Literature": {
                "drug_families": ["DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "omeprazole": {
        "drop_own_atc": ["A02BC01"],
        "dags": {
            "Literature": {
                "drug_families": ["DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["DIURETICS", "CONTRAST MEDIA",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    "allopurinol": {
        "drop_own_atc": [],
        "dags": {
            "Literature": {
                "drug_families": ["DIURETICS",
                                  "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE", "CONTRAST MEDIA"],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": ["AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                                  "DRUGS FOR ACID RELATED DISORDERS", "IMMUNOSUPPRESSANTS",
                                  "ANTIBACTERIALS FOR SYSTEMIC USE", "PSYCHOLEPTICS",
                                  "CONTRAST MEDIA"],
                "extra_confounders": [
                    "Big_bp", "Small_bp", "sodium", "creatinine", "potassium",
                    "bun", "chloride", "bicarbonate",
                    "Weight (Lbs)", "gender", "anchor_age", "pre_AKI",
                ],
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Best models per drug — derived from Tables 12-15 in the paper.
# Each entry holds the specific sklearn/xgboost instances to pass to
# run_meta_learners() and run_dml_learner().
# Format:
#   "<drug>": {
#       "s_models":   [overall_model]
#       "t_models":   [model_treated, model_control]
#       "x_models":   [clf_treat, clf_control, clf_propensity,
#                      cate_regressor_treat, cate_regressor_control]
#       "dml_models": [model_y, model_t, model_final]
#   }
# ---------------------------------------------------------------------------

BEST_MODELS = {
    "ibuprofen": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [XGBClassifier(n_jobs=1, verbosity=0),
                       XGBClassifier(n_jobs=1, verbosity=0)],
        "x_models":   [MLPClassifier(max_iter=500), SVC(probability=True),
                       LogisticRegression(max_iter=1000),
                       MLPRegressor(max_iter=1000), MLPRegressor(max_iter=1000)],
        "dml_models": [XGBRegressor(n_jobs=1, verbosity=0),
                       RandomForestRegressor(),
                       DecisionTreeRegressor()],
    },
    "ketorolac": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [RandomForestClassifier(), RandomForestClassifier()],
        "x_models":   [SVC(probability=True), SGDClassifier(loss="log_loss"),
                       LogisticRegression(max_iter=1000),
                       SVR(), SVR()],
        "dml_models": [GradientBoostingRegressor(),
                       XGBRegressor(n_jobs=1, verbosity=0),
                       DecisionTreeRegressor()],
    },
    "vancomycin": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [RandomForestClassifier(), SGDClassifier(loss="log_loss")],
        "x_models":   [XGBClassifier(n_jobs=1, verbosity=0),
                       SGDClassifier(loss="log_loss"),
                       SVC(probability=True),
                       RandomForestRegressor(), RandomForestRegressor()],
        "dml_models": [GradientBoostingRegressor(),
                       GradientBoostingRegressor(),
                       DecisionTreeRegressor()],
    },
    "lisinopril": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [XGBClassifier(n_jobs=1, verbosity=0), RandomForestClassifier()],
        "x_models":   [RandomForestClassifier(), SVC(probability=True),
                       GradientBoostingClassifier(),
                       RandomForestRegressor(), RandomForestRegressor()],
        "dml_models": [GradientBoostingRegressor(),
                       GradientBoostingRegressor(),
                       DecisionTreeRegressor()],
    },
    "furosemide": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [DecisionTreeClassifier(), MLPClassifier(max_iter=500)],
        "x_models":   [RandomForestClassifier(), SVC(probability=True),
                       SVC(probability=True),
                       RandomForestRegressor(), RandomForestRegressor()],
        "dml_models": [GradientBoostingRegressor(),
                       GradientBoostingRegressor(),
                       DecisionTreeRegressor()],
    },
    "pantoprazole": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [DecisionTreeClassifier(), SGDClassifier(loss="log_loss")],
        "x_models":   [DecisionTreeClassifier(), SVC(probability=True),
                       SVC(probability=True),
                       DecisionTreeRegressor(), DecisionTreeRegressor()],
        "dml_models": [GradientBoostingRegressor(),
                       DecisionTreeRegressor(),
                       DecisionTreeRegressor()],
    },
    "omeprazole": {
        "s_models":   [XGBClassifier(n_jobs=1, verbosity=0)],
        "t_models":   [DecisionTreeClassifier(), RandomForestClassifier()],
        "x_models":   [DecisionTreeClassifier(), SVC(probability=True),
                       GradientBoostingClassifier(),
                       XGBRegressor(n_jobs=1, verbosity=0),
                       XGBRegressor(n_jobs=1, verbosity=0)],
        "dml_models": [GradientBoostingRegressor(),
                       XGBRegressor(n_jobs=1, verbosity=0),
                       DecisionTreeRegressor()],
    },
    "allopurinol": {
        "s_models":   [RandomForestClassifier()],
        "t_models":   [XGBClassifier(n_jobs=1, verbosity=0),
                       XGBClassifier(n_jobs=1, verbosity=0)],
        "x_models":   [RandomForestClassifier(), SVC(probability=True),
                       SVC(probability=True),
                       DecisionTreeRegressor(), DecisionTreeRegressor()],
        "dml_models": [GradientBoostingRegressor(),
                       GradientBoostingRegressor(),
                       DecisionTreeRegressor()],
    },
}

POSITIVE_DRUGS  = list(DRUG_DAG_CONFIG.keys())
NEGATIVE_DRUGS  = ["simethicone", "prochlorperazine", "lactulose"]
LAB_REPRS       = ["mean", "t0t1"]
