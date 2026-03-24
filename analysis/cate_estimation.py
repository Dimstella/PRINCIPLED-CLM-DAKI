"""
analysis/cate_estimation.py
----------------------------
Full CATE estimation pipeline for all 24 drug-pair datasets.

Replaces the 8 per-drug Jupyter notebooks (Ibuprofen.ipynb, Ketorolac.ipynb, …).

For each combination of:
  • positive drug   (8)
  • control drug    (3)
  • lab representation  (mean / t0t1)
  • DAG source      (Literature / Clinician)

the pipeline:
  1. Loads and preprocesses the paired dataset.
  2. Builds the confounder list from the DAG config.
  3. Runs the three preprocessing scenarios (Original / Binary / Normalised).
  4. Executes S-, T-, X-Learner and DML with the best pre-selected models.
  5. Tags results with CI-width validity (< 0.85 = valid).
  6. Saves one Excel workbook per positive drug.

Usage
-----
    python cate_estimation.py --work-dir /data/final_datasets --out-dir /results/cate

    # Single drug
    python cate_estimation.py --work-dir /data --out-dir /results --drug ibuprofen

    # Single pair
    python cate_estimation.py --work-dir /data --out-dir /results \\
        --drug ibuprofen --control simethicone
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add repo root to path so cml/ modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cml"))

from config import (
    BEST_MODELS,
    CHARLSON_COLS,
    CI_WIDTH_THRESHOLD,
    DROP_ALWAYS,
    DRUG_DAG_CONFIG,
    ID_COLS,
    LAB_COLS_MEAN,
    LAB_COLS_T0T1,
    LAB_SUFFIXES_T0T1,
    NO_BINARISE_COLS,
    NEGATIVE_DRUGS,
    POSITIVE_DRUGS,
    SCALE_COLS_MEAN,
    SCALE_COLS_T0T1,
)
from learners import run_dml_learner, run_meta_learners

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# I/O helpers
# ===========================================================================

def _resolve_path(work_dir: Path, positive_drug: str, control_drug: str,
                  lab_repr: str) -> Path:
    """Resolve the paired dataset path, trying common naming conventions."""
    # lab_repr: "mean" or "t0t1"
    suffix = "" if lab_repr == "mean" else "_t0t1"
    candidates = [
        f"{positive_drug}_{control_drug}{suffix}.parquet.gzip",
        f"{positive_drug}_vs_{control_drug}_{lab_repr}.parquet.gzip",
        f"{positive_drug}_{control_drug}_AKI{suffix}.parquet.gzip",
    ]
    for name in candidates:
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No file found for {positive_drug} vs {control_drug} ({lab_repr}) "
        f"in {work_dir}.\nTried: {candidates}"
    )


def _load_atc_families(work_dir: Path) -> pd.DataFrame:
    """Load 3-character ATC family codes from atc_codes.csv."""
    csv_path = work_dir / "atc_codes.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df[df["ATC.code"].str.len() == 3].copy()


# ===========================================================================
# Preprocessing
# ===========================================================================

def _preprocess(
    df: pd.DataFrame,
    lab_repr: str,
    drop_own_atc: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode gender, drop identifier/timestamp columns, z-score continuous
    columns, binarise ATC duration columns.

    Returns
    -------
    df_clean   : full DataFrame (with y, t columns preserved for extraction)
    column_dt  : preprocessed feature-only DataFrame (no y/t/subject_id)
    """
    df = df.copy()
    df["gender"] = df["gender"].replace({"F": 1, "M": 0})

    # Columns to completely remove
    drop_cols = (
        [c for c in ID_COLS if c in df.columns]
        + [c for c in drop_own_atc if c in df.columns]
        + [c for c in ["starttime_lisinopril", "stoptime_lisinopril",
                        "starttime_kidney_disease", "stoptime_kidney_disease",
                        "kidney_disease_flag"] if c in df.columns]
    )

    column_dt = df.drop(columns=drop_cols, errors="ignore").copy()

    # Z-score continuous columns
    scale_cols = SCALE_COLS_T0T1 if lab_repr == "t0t1" else SCALE_COLS_MEAN
    present_scale = [c for c in scale_cols if c in column_dt.columns]
    if present_scale:
        sc = StandardScaler()
        column_dt[present_scale] = sc.fit_transform(column_dt[present_scale])

    # Binarise ATC duration columns (values > 1 → 1)
    for col in column_dt.columns:
        if col not in NO_BINARISE_COLS and col not in DROP_ALWAYS:
            column_dt.loc[column_dt[col] > 1, col] = 1

    # Drop non-feature columns
    column_dt.drop(
        columns=[c for c in DROP_ALWAYS if c in column_dt.columns],
        inplace=True,
        errors="ignore",
    )

    # Clean the raw df too (drop charlson_comorbidity_index / BMI)
    df.drop(
        columns=[c for c in DROP_ALWAYS if c in df.columns],
        inplace=True,
        errors="ignore",
    )

    return df, column_dt


# ===========================================================================
# Confounder list construction
# ===========================================================================

def _build_confounders(
    positive_drug: str,
    dag_source: str,
    atc_families_df: pd.DataFrame,
    available_med_cols: list[str],
) -> tuple[list[str], list[str]]:
    """
    Return (full_confounders, extra_confounders) for a given drug/DAG combo.

    full_confounders = extra_confounders + ATC-family cols + Charlson cols
    """
    dag = DRUG_DAG_CONFIG[positive_drug]["dags"][dag_source]
    families = [f.upper() for f in dag["drug_families"]]
    extra = dag["extra_confounders"]

    atc_prefixes = list(
        atc_families_df[atc_families_df["Name"].str.upper().isin(families)]["ATC.code"]
    )
    med_confounders = [
        c for c in available_med_cols
        if any(c.startswith(p) for p in atc_prefixes)
    ]

    full = list(dict.fromkeys(extra + med_confounders + CHARLSON_COLS))
    return full, extra


# ===========================================================================
# Three-scenario CATE runner
# ===========================================================================

def _run_three_scenarios(
    df: pd.DataFrame,
    column_dt: pd.DataFrame,
    confounders: list[str],
    extra_confounders: list[str],
    positive_drug: str,
    run_dml: bool = True,
) -> pd.DataFrame:
    """
    Run S/T/X-Learner (+DML) under three preprocessing scenarios:
      Original   — raw confounder columns
      Binary     — normalised ATC + raw extra cols
      Normalised — normalised ATC only

    Filters out rows where pre_AKI == 1 (paper §2.2.1).
    """
    # Filter pre-AKI patients
    if "pre_AKI" in df.columns:
        df = df[df["pre_AKI"] == 0].drop(columns=["pre_AKI"], errors="ignore")
    if "pre_AKI" in column_dt.columns:
        column_dt = column_dt[column_dt["pre_AKI"] == 0].drop(
            columns=["pre_AKI"], errors="ignore"
        )

    y = np.array(df["y"]).astype("float32").squeeze()
    t = df["t"].to_numpy().astype("float32")

    bm = BEST_MODELS[positive_drug]
    all_parts: list[pd.DataFrame] = []

    for scenario, X in _scenario_matrices(df, column_dt, confounders, extra_confounders):
        parts = []
        # S/T/X
        res_stx = run_meta_learners(
            X, y, t,
            s_models=bm["s_models"],
            t_models=bm["t_models"],
            x_models=bm["x_models"],
        )
        res_stx["Process"] = scenario
        parts.append(res_stx)

        # DML
        if run_dml:
            res_dml = run_dml_learner(X, y, t, dml_models=bm["dml_models"])
            res_dml["Process"] = scenario
            parts.append(res_dml)

        all_parts.extend(parts)

    return pd.concat(all_parts, ignore_index=True, sort=False)


def _scenario_matrices(
    df: pd.DataFrame,
    column_dt: pd.DataFrame,
    confounders: list[str],
    extra_confounders: list[str],
):
    """
    Yield (scenario_name, X) for the three preprocessing scenarios.
    Only yields a scenario if the required confounder columns are present.
    """
    # Original
    present_orig = [c for c in confounders if c in df.columns]
    if present_orig:
        yield "Original", df[present_orig].copy().astype(float).to_numpy()

    # Binary (normalised ATC + raw extra cols)
    present_norm = [c for c in confounders if c in column_dt.columns]
    if present_norm:
        X2 = column_dt[present_norm].copy()
        for ec in extra_confounders:
            if ec in df.columns and ec not in X2.columns:
                X2[ec] = df[ec].values
        yield "Binary", X2.astype(float).to_numpy()

        # Normalised (normalised ATC only)
        X3 = column_dt[present_norm].copy()
        yield "Normalised", X3.astype(float).to_numpy()


# ===========================================================================
# CI validity tagging
# ===========================================================================

def _tag_ci_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns:
      CI_width  : upper - lower bound of CI
      Valid     : True if CI_width < CI_WIDTH_THRESHOLD
      Significant: True if CI does not contain 0
    """
    df = df.copy()

    def _width(ci):
        try:
            lo, hi = ci if isinstance(ci, (list, tuple)) else eval(str(ci))
            return float(hi) - float(lo)
        except Exception:
            return np.nan

    def _significant(ci):
        try:
            lo, hi = ci if isinstance(ci, (list, tuple)) else eval(str(ci))
            return not (float(lo) <= 0 <= float(hi))
        except Exception:
            return False

    df["CI_width"]    = df["CI ATE"].apply(_width)
    df["Valid"]       = df["CI_width"] < CI_WIDTH_THRESHOLD
    df["Significant"] = df["CI ATE"].apply(_significant)
    return df


# ===========================================================================
# Per-pair orchestration
# ===========================================================================

def run_pair(
    work_dir: Path,
    positive_drug: str,
    control_drug: str,
    lab_repr: str,
    atc_families_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full CATE pipeline for one (positive_drug, control_drug, lab_repr) triple.
    Runs both Literature and Clinician DAGs and combines results.

    Returns a tagged DataFrame with columns:
        Meta-Learner | Model | ATE | CI ATE | Process | DAG |
        Lab | Drug | Control | CI_width | Valid | Significant
    """
    logger.info("    %s vs %s  [%s]", positive_drug, control_drug, lab_repr)
    cfg = DRUG_DAG_CONFIG[positive_drug]

    path = _resolve_path(work_dir, positive_drug, control_drug, lab_repr)
    df_raw = pd.read_parquet(path)

    df_clean, column_dt = _preprocess(df_raw, lab_repr, cfg["drop_own_atc"])

    # Identify available ATC medication columns
    lab_cols = LAB_COLS_T0T1 if lab_repr == "t0t1" else LAB_COLS_MEAN
    non_feature = set(lab_cols + ["Weight (Lbs)", "gender", "anchor_age",
                                   "pre_AKI", "y", "t", "subject_id"]
                      + CHARLSON_COLS)
    available_med_cols = [c for c in column_dt.columns if c not in non_feature]

    all_dag_parts: list[pd.DataFrame] = []

    for dag_source in cfg["dags"]:
        confounders, extra = _build_confounders(
            positive_drug, dag_source, atc_families_df, available_med_cols
        )
        res = _run_three_scenarios(df_clean, column_dt, confounders, extra, positive_drug)
        res["DAG"]     = dag_source
        res["Lab"]     = "Mean" if lab_repr == "mean" else "t0-t1"
        res["Drug"]    = positive_drug.capitalize()
        res["Control"] = control_drug.capitalize()
        all_dag_parts.append(res)

    combined = pd.concat(all_dag_parts, ignore_index=True, sort=False)
    return _tag_ci_validity(combined)


# ===========================================================================
# Full pipeline orchestrator
# ===========================================================================

def run_cate_pipeline(
    work_dir: Path,
    out_dir: Path,
    positive_drugs: list[str] | None = None,
    control_drugs: list[str] | None = None,
    lab_reprs: list[str] | None = None,
) -> None:
    """
    Run the CATE pipeline for all requested (drug, control, lab_repr) triples
    and save one Excel workbook per positive drug.
    """
    positives   = positive_drugs or POSITIVE_DRUGS
    controls    = control_drugs  or NEGATIVE_DRUGS
    lab_reprs_  = lab_reprs      or ["mean", "t0t1"]

    out_dir.mkdir(parents=True, exist_ok=True)
    atc_df = _load_atc_families(work_dir)

    # Collect results per positive drug
    drug_results: dict[str, list[pd.DataFrame]] = {d: [] for d in positives}

    total = len(positives) * len(controls) * len(lab_reprs_)
    done  = 0

    for pos, ctrl, lab in product(positives, controls, lab_reprs_):
        done += 1
        logger.info("[%d/%d]  %s  vs  %s  (%s)", done, total, pos, ctrl, lab)
        try:
            pair_df = run_pair(work_dir, pos, ctrl, lab, atc_df)
            drug_results[pos].append(pair_df)
        except FileNotFoundError as exc:
            logger.warning("  SKIP — %s", exc)
        except Exception as exc:
            logger.error("  ERROR — %s", exc, exc_info=True)

    # Write workbooks
    for drug in positives:
        frames = drug_results[drug]
        if not frames:
            logger.warning("No results for %s", drug)
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        out_path = out_dir / f"{drug}_cate_results.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for ctrl in controls:
                sheet_df = combined[combined["Control"] == ctrl.capitalize()]
                if not sheet_df.empty:
                    sheet_df.to_excel(writer, sheet_name=ctrl.capitalize(), index=False)
        logger.info("Saved: %s  (%d rows)", out_path.name, len(combined))

    logger.info("CATE pipeline complete. Results in %s", out_dir)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CATE estimation for all drug pairs."
    )
    p.add_argument("--work-dir", required=True, type=Path,
                   help="Directory with paired parquet datasets and atc_codes.csv.")
    p.add_argument("--out-dir",  required=True, type=Path,
                   help="Directory where Excel results will be saved.")
    p.add_argument("--drug",    default=None,
                   help="Run only for this positive drug (default: all 8).")
    p.add_argument("--control", default=None,
                   help="Run only for this control drug (default: all 3).")
    p.add_argument("--lab-repr", default=None, choices=["mean", "t0t1"],
                   help="Lab representation (default: both).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_cate_pipeline(
        work_dir      = args.work_dir.resolve(),
        out_dir       = args.out_dir.resolve(),
        positive_drugs = [args.drug]      if args.drug     else None,
        control_drugs  = [args.control]   if args.control  else None,
        lab_reprs      = [args.lab_repr]  if args.lab_repr else None,
    )


if __name__ == "__main__":
    main()
