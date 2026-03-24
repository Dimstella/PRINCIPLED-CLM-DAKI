"""
analysis/hte_estimation.py
--------------------------
Heterogeneous Treatment Effect (HTE) estimation for all 8 positive drugs.

Replaces ITE_clustering.ipynb (228 cells).

For each drug the notebook used the *single* best-performing (drug, control, lab_repr)
configuration and the matching best CML architecture to produce per-patient ITE
estimates.  This script encodes those choices as data (HTE_CONFIG), runs the
learner, writes the ITE column back to the dataset, *restores raw (unscaled) lab
values* from the original parquet so that downstream clinical plots are
interpretable, then saves a `<drug>_<control>_ite.parquet.gzip` artefact.

HTE configuration per drug (from the notebook):
┌─────────────┬────────────────────┬───────────┬────────────┬───────────────────────────────────────────────────┐
│ Drug        │ Control            │ Lab repr  │ Learner    │ Models                                            │
├─────────────┼────────────────────┼───────────┼────────────┼───────────────────────────────────────────────────┤
│ ibuprofen   │ prochlorperazine   │ mean      │ DML        │ SVR, SVR, SVR                                     │
│ ketorolac   │ lactulose          │ mean      │ X-Learner  │ SVC, SGDClf, LR, SVR, SVR                         │
│ vancomycin  │ prochlorperazine   │ t0t1      │ S-Learner  │ RandomForest                                      │
│ lisinopril  │ prochlorperazine   │ t0t1      │ X-Learner  │ RF, SVC, GBC, RF, RF                              │
│ furosemide  │ lactulose          │ t0t1      │ X-Learner  │ DTC, SVC(prob), SVC(prob), RF, RF                 │
│ pantoprazole│ simethicone        │ t0t1      │ S-Learner  │ RandomForest                                      │
│ omeprazole  │ prochlorperazine   │ mean      │ S-Learner  │ RandomForest                                      │
│ allopurinol │ prochlorperazine   │ mean      │ S-Learner  │ RandomForest                                      │
└─────────────┴────────────────────┴───────────┴────────────┴───────────────────────────────────────────────────┘

Usage
-----
    # All 8 drugs
    python hte_estimation.py --work-dir /data/final_datasets --out-dir /results/hte

    # Single drug
    python hte_estimation.py --work-dir /data --out-dir /results/hte --drug ibuprofen
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

# EconML learners
from econml.metalearners import SLearner, XLearner
from econml.dml import DML

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cml"))
from config import (
    CHARLSON_COLS,
    DROP_ALWAYS,
    DRUG_DAG_CONFIG,
    ID_COLS,
    LAB_COLS_MEAN,
    LAB_COLS_T0T1,
    LAB_SUFFIXES_T0T1,
    NO_BINARISE_COLS,
    POSITIVE_DRUGS,
    SCALE_COLS_MEAN,
    SCALE_COLS_T0T1,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw lab columns that must be restored after scaling (for interpretable plots)
# ---------------------------------------------------------------------------

RAW_RESTORE_MEAN = [
    "glucose", "sodium", "creatinine", "potassium", "bun",
    "bicarbonate", "chloride", "aniongap",
    "Big_bp", "Small_bp", "Weight (Lbs)",
]

RAW_RESTORE_T0T1 = [
    "glucose_first", "glucose_last",
    "sodium_first", "sodium_last",
    "creatinine_first", "creatinine_last",
    "potassium_first", "potassium_last",
    "bun_first", "bun_last",
    "bicarbonate_first", "bicarbonate_last",
    "chloride_first", "chloride_last",
    "aniongap_first", "aniongap_last",
    "Big_bp", "Small_bp", "Weight (Lbs)",
]

# ---------------------------------------------------------------------------
# HTE configuration: one entry per drug
# ---------------------------------------------------------------------------
# Each entry has:
#   control   : negative control drug used for the HTE dataset
#   lab_repr  : "mean" or "t0t1"
#   learner   : "s_learner" | "x_learner" | "dml"
#   models    : model instances passed to the respective learner
#   dag       : DAG source ("Literature" or "Clinician")
# ---------------------------------------------------------------------------

HTE_CONFIG: dict[str, dict] = {
    "ibuprofen": {
        "control":   "prochlorperazine",
        "lab_repr":  "mean",
        "learner":   "dml",
        "models":    [SVR(), SVR(), SVR()],          # [model_y, model_t, model_final]
        "dag":       "Clinician",
    },
    "ketorolac": {
        "control":   "lactulose",
        "lab_repr":  "mean",
        "learner":   "x_learner",
        "models":    [SVC(probability=True), SGDClassifier(loss="log_loss"),
                      LogisticRegression(random_state=42, max_iter=1000),
                      SVR(), SVR()],
        "dag":       "Literature",
    },
    "vancomycin": {
        "control":   "prochlorperazine",
        "lab_repr":  "t0t1",
        "learner":   "s_learner",
        "models":    [RandomForestClassifier()],
        "dag":       "Literature",
    },
    "lisinopril": {
        "control":   "prochlorperazine",
        "lab_repr":  "t0t1",
        "learner":   "x_learner",
        "models":    [RandomForestClassifier(), SVC(probability=True),
                      GradientBoostingClassifier(),
                      RandomForestRegressor(), RandomForestRegressor()],
        "dag":       "Literature",
    },
    "furosemide": {
        "control":   "lactulose",
        "lab_repr":  "t0t1",
        "learner":   "x_learner",
        "models":    [DecisionTreeClassifier(), SVC(probability=True),
                      SVC(probability=True),
                      RandomForestRegressor(), RandomForestRegressor()],
        "dag":       "Literature",
    },
    "pantoprazole": {
        "control":   "simethicone",
        "lab_repr":  "t0t1",
        "learner":   "s_learner",
        "models":    [RandomForestClassifier()],
        "dag":       "Literature",
    },
    "omeprazole": {
        "control":   "prochlorperazine",
        "lab_repr":  "mean",
        "learner":   "s_learner",
        "models":    [RandomForestClassifier()],
        "dag":       "Literature",
    },
    "allopurinol": {
        "control":   "prochlorperazine",
        "lab_repr":  "mean",
        "learner":   "s_learner",
        "models":    [RandomForestClassifier()],
        "dag":       "Clinician",
    },
}


# ===========================================================================
# Dataset helpers
# ===========================================================================

def _resolve_path(work_dir: Path, drug: str, control: str, lab_repr: str) -> Path:
    """Try common filename patterns for the paired dataset."""
    suffix = "" if lab_repr == "mean" else "_t0t1"
    candidates = [
        f"{drug}_{control}{suffix}.parquet.gzip",
        f"{drug}_{control}_AKI{suffix}.parquet.gzip",
        f"{drug}_{control}_minmax.parquet.gzip",    # notebook naming convention
        f"{drug}_vs_{control}_{lab_repr}.parquet.gzip",
    ]
    for name in candidates:
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No paired dataset for {drug} vs {control} ({lab_repr}) in {work_dir}.\n"
        f"Tried: {candidates}"
    )


def _load_and_preprocess(
    work_dir: Path,
    drug: str,
    control: str,
    lab_repr: str,
    drop_own_atc: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw paired dataset, encode gender, scale continuous columns,
    binarise ATC duration columns.

    Returns
    -------
    df_raw    : original dataset (BEFORE any scaling — used for raw-value restoration)
    df_scaled : dataset with scaled + binarised features (used for learner input)
    """
    path = _resolve_path(work_dir, drug, control, lab_repr)
    df_raw = pd.read_parquet(path)
    df = df_raw.copy()

    df["gender"] = df["gender"].replace({"F": 1, "M": 0})

    # Drop timestamp / identifier columns
    drop_cols = (
        [c for c in ID_COLS if c in df.columns]
        + [c for c in drop_own_atc if c in df.columns]
        + [c for c in [
            "starttime_lisinopril", "stoptime_lisinopril",
            "starttime_kidney_disease", "stoptime_kidney_disease",
            "kidney_disease_flag", "charttime_first", "charttime_last",
        ] if c in df.columns]
    )
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Scale continuous columns in-place on df (df_raw stays unscaled)
    scale_cols = SCALE_COLS_T0T1 if lab_repr == "t0t1" else SCALE_COLS_MEAN
    present_scale = [c for c in scale_cols if c in df.columns]
    if present_scale:
        sc = StandardScaler()
        df[present_scale] = sc.fit_transform(df[present_scale])

    # Binarise ATC duration columns (values > 1 → 1)
    for col in df.columns:
        if col not in NO_BINARISE_COLS and col not in DROP_ALWAYS and col not in {"y", "t"}:
            df.loc[df[col] > 1, col] = 1

    # Drop non-feature columns
    df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns],
            errors="ignore", inplace=True)

    return df_raw, df


def _build_confounder_list(
    drug: str,
    dag_source: str,
    atc_families_df: pd.DataFrame,
    df: pd.DataFrame,
) -> list[str]:
    """Return the confounder list for a given drug / DAG."""
    dag = DRUG_DAG_CONFIG[drug]["dags"][dag_source]
    families = [f.upper() for f in dag["drug_families"]]
    extra = dag["extra_confounders"]

    atc_prefixes = list(
        atc_families_df[atc_families_df["Name"].str.upper().isin(families)]["ATC.code"]
    )

    # Identify available ATC medication columns
    lab_cols = LAB_COLS_T0T1 if "aniongap_first" in df.columns else LAB_COLS_MEAN
    non_feature = set(
        lab_cols + ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI",
                    "y", "t", "subject_id"] + CHARLSON_COLS
    )
    available_med_cols = [c for c in df.columns if c not in non_feature]
    med_confounders = [
        c for c in available_med_cols
        if any(c.startswith(p) for p in atc_prefixes)
    ]

    # Remove pre_AKI from extra if already handled
    extra_clean = [e for e in extra if e != "pre_AKI"]

    return list(dict.fromkeys(extra_clean + med_confounders + CHARLSON_COLS))


# ===========================================================================
# Per-learner HTE runners  (return per-patient hte_pred array)
# ===========================================================================

def _run_s_learner(models: list, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    learner = SLearner(overall_model=models[0])
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    return learner.effect(X)


def _run_x_learner(models: list, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    # models = [clf_treat, clf_control, clf_propensity, reg_treat, reg_control]
    learner = XLearner(
        models=[models[0], models[1]],
        propensity_model=models[2],
        cate_models=[models[3], models[4]],
    )
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    return learner.effect(X)


def _run_dml(models: list, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    # models = [model_y, model_t, model_final]
    learner = DML(
        model_y=models[0],
        model_t=models[1],
        model_final=models[2],
        discrete_outcome=True,
        discrete_treatment=True,
    )
    learner.fit(Y=y, T=t, X=X, inference="bootstrap")
    return learner.effect(X)


_LEARNER_FNS = {
    "s_learner": _run_s_learner,
    "x_learner": _run_x_learner,
    "dml":        _run_dml,
}


# ===========================================================================
# Per-drug HTE pipeline
# ===========================================================================

def compute_hte(
    work_dir: Path,
    out_dir: Path,
    drug: str,
    atc_families_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the HTE pipeline for one drug.

    Steps
    -----
    1. Load + preprocess the paired dataset.
    2. Build confounder list from the drug's DAG.
    3. Filter pre_AKI == 1 patients.
    4. Run the best CML learner to get per-patient ITE estimates.
    5. Write ITE column to the scaled dataframe.
    6. Restore raw (unscaled) lab/vital values from the original parquet.
    7. Save to `<out_dir>/<drug>_<control>_ite.parquet.gzip`.

    Returns
    -------
    df_ite : DataFrame with ITE column and raw lab values
    """
    cfg = HTE_CONFIG[drug]
    control    = cfg["control"]
    lab_repr   = cfg["lab_repr"]
    learner_fn = _LEARNER_FNS[cfg["learner"]]
    models     = cfg["models"]
    dag_source = cfg["dag"]

    logger.info("  %s  vs  %s  [%s] — %s", drug, control, lab_repr, cfg["learner"].upper())

    drop_own_atc = DRUG_DAG_CONFIG[drug].get("drop_own_atc", [])
    df_raw, df = _load_and_preprocess(work_dir, drug, control, lab_repr, drop_own_atc)

    confounders = _build_confounder_list(drug, dag_source, atc_families_df, df)

    # Filter pre_AKI == 1 patients
    mask_aki = pd.Series(True, index=df.index)
    if "pre_AKI" in df.columns:
        mask_aki = df["pre_AKI"] == 0
        df = df[mask_aki].drop(columns=["pre_AKI"], errors="ignore")

    y = np.array(df["y"]).astype("float32").squeeze()
    t = np.array(df["t"]).astype("float32").squeeze()

    present_conf = [c for c in confounders if c in df.columns]
    X = np.array(df[present_conf]).astype("float32")

    # Run learner
    hte_pred = learner_fn(models, X, y, t)
    df["ite"] = hte_pred

    # Restore raw lab values from the original unscaled parquet
    raw_cols = RAW_RESTORE_T0T1 if lab_repr == "t0t1" else RAW_RESTORE_MEAN
    df_raw_filtered = df_raw[mask_aki].copy() if "pre_AKI" in df_raw.columns else df_raw.copy()
    for col in raw_cols:
        if col in df_raw_filtered.columns:
            df[col] = df_raw_filtered[col].values

    # Save
    out_path = out_dir / f"{drug}_{control}_ite.parquet.gzip"
    df.to_parquet(out_path, compression="gzip")
    logger.info("    Saved: %s  (n=%d,  mean_ITE=%.4f)", out_path.name, len(df), float(np.mean(hte_pred)))

    return df


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_hte_pipeline(
    work_dir: Path,
    out_dir: Path,
    drugs: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run HTE estimation for all (or a subset of) positive drugs.

    Returns
    -------
    dict mapping drug name → ITE DataFrame
    """
    drugs = drugs or POSITIVE_DRUGS
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ATC families reference
    atc_csv = work_dir / "atc_codes.csv"
    atc_df  = pd.read_csv(atc_csv)
    atc_df  = atc_df.loc[:, ~atc_df.columns.str.startswith("Unnamed")]
    atc_families = atc_df[atc_df["ATC.code"].str.len() == 3].copy()
    atc_families["Name"] = atc_families["Name"].str.upper()

    results: dict[str, pd.DataFrame] = {}

    for drug in drugs:
        if drug not in HTE_CONFIG:
            logger.warning("No HTE config for %s — skipping", drug)
            continue
        logger.info("[%d/%d]  %s", list(drugs).index(drug) + 1, len(drugs), drug)
        try:
            df_ite = compute_hte(work_dir, out_dir, drug, atc_families)
            results[drug] = df_ite
        except FileNotFoundError as exc:
            logger.warning("  SKIP — %s", exc)
        except Exception as exc:
            logger.error("  ERROR — %s", exc, exc_info=True)

    logger.info("HTE pipeline complete. ITE files written to %s", out_dir)
    return results


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HTE estimation for all positive drugs.")
    p.add_argument("--work-dir", required=True, type=Path,
                   help="Directory containing paired parquet datasets and atc_codes.csv.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory where _ite.parquet.gzip files will be saved.")
    p.add_argument("--drug", default=None,
                   help="Run only for this drug (default: all 8 positive drugs).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_hte_pipeline(
        work_dir = args.work_dir.resolve(),
        out_dir  = args.out_dir.resolve(),
        drugs    = [args.drug] if args.drug else None,
    )


if __name__ == "__main__":
    main()
