"""
matching/run_benchmark.py
-------------------------
Orchestrates the full matching benchmark across all 24 drug-pair datasets
(8 positive drugs × 3 negative controls), both DAG sources (Literature /
Clinician), and three data-preprocessing scenarios (Original / Binary /
Normalised).

This script replaces the 180-cell MatchingBenchmark.ipynb notebook.

Usage
-----
    # Run everything and save to sml_mean.xlsx
    python run_benchmark.py --work-dir /path/to/final_datasets --out sml_mean.xlsx

    # Run a single pair
    python run_benchmark.py --work-dir /path/to/data \\
        --drug vancomycin --control lactulose --out vac_lac.xlsx

    # Run a single drug, all controls
    python run_benchmark.py --work-dir /path/to/data --drug ibuprofen

Output
------
One Excel workbook with one sheet per positive drug.  Each sheet contains
the full benchmark results (SMD metrics, patient counts, ratio score) for
every combination of:
  • control drug  (Simethicone / Prochlorperazine / Lactulose)
  • DAG source    (Literature / Clinician)
  • preprocessing (Original / Binary / Normalised)
  • matching method (PSM-LR, PSM-RF, …, NN, Mahalanobis, Exact)
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    CHARLSON_COLS,
    DRUG_CONFIG,
    DROP_FROM_FEATURES,
    ID_COLS,
    LAB_COLS,
    DEM_COLS,
    NO_BINARISE_COLS,
    NEGATIVE_DRUGS,
    POSITIVE_DRUGS,
    SCALE_COLS,
    TIMESTAMP_COLS,
)
from psm import apply_all_matching_methods

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Step 1 – Dataset loading and preprocessing
# ===========================================================================

def _resolve_path(work_dir: Path, positive_drug: str, control_drug: str) -> Path:
    """Try multiple common filename patterns for the paired dataset."""
    candidates = [
        f"{positive_drug}_{control_drug}.parquet.gzip",
        f"{positive_drug}_vs_{control_drug}_mean.parquet.gzip",
        f"{positive_drug}_{control_drug}_AKI.parquet.gzip",
    ]
    for name in candidates:
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No paired dataset found for {positive_drug} vs {control_drug} "
        f"in {work_dir}.\nTried: {candidates}"
    )


def load_and_preprocess(
    work_dir: Path,
    positive_drug: str,
    control_drug: str,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the paired dataset, encode gender, drop non-feature columns, scale
    continuous lab/vital columns, and binarise ATC duration columns.

    Returns
    -------
    df         : the full loaded DataFrame (with y, t, subject_id)
    column_dt  : the preprocessed feature-only DataFrame (no y/t/subject_id)
                 Used for the Normalised and Binary preprocessing scenarios.
    """
    path = _resolve_path(work_dir, positive_drug, control_drug)
    df = pd.read_parquet(path)

    # Encode gender
    df["gender"] = df["gender"].replace({"F": 1, "M": 0})

    # Columns to drop when building the feature matrix
    cols_to_drop = (
        [c for c in ID_COLS if c in df.columns]
        + [c for c in cfg["extra_drop"] if c in df.columns]
        + [c for c in TIMESTAMP_COLS if c in df.columns]
        + [c for c in cfg["drop_own_atc"] if c in df.columns]
    )
    column_dt = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # Scale continuous columns
    sc = StandardScaler()
    scale_present = [c for c in SCALE_COLS if c in column_dt.columns]
    column_dt[scale_present] = sc.fit_transform(column_dt[scale_present])

    # Binarise ATC duration columns (values > 1 → 1)
    # Only apply to columns not in the no-binarise list
    for col in column_dt.columns:
        if col not in NO_BINARISE_COLS and col not in DROP_FROM_FEATURES:
            column_dt.loc[column_dt[col] > 1, col] = 1

    # Drop columns that should not be features at all
    column_dt.drop(
        columns=[c for c in DROP_FROM_FEATURES if c in column_dt.columns],
        inplace=True,
        errors="ignore",
    )

    return df, column_dt


# ===========================================================================
# Step 2 – Confounder selection
# ===========================================================================

def build_confounder_list(
    positive_drug: str,
    dag_source: str,
    atc_codes_df: pd.DataFrame,
    available_med_cols: list[str],
) -> list[str]:
    """
    Build the full confounder list for one (drug, DAG) combination:
      extra_confounders + matching ATC-family columns + Charlson columns

    Parameters
    ----------
    positive_drug     : e.g. 'ibuprofen'
    dag_source        : 'Literature' or 'Clinician'
    atc_codes_df      : DataFrame with columns ['ATC.code', 'Name'] (3-char families)
    available_med_cols: ATC columns actually present in this pair's dataset

    Returns
    -------
    list[str]  confounder column names
    """
    dag_cfg   = DRUG_CONFIG[positive_drug]["dags"][dag_source]
    families  = dag_cfg["drug_families"]
    extra     = dag_cfg["extra_confounders"]

    # Get 3-char ATC prefixes for selected drug families
    fam_upper = [f.upper() for f in families]
    atc_prefixes = list(
        atc_codes_df[atc_codes_df["Name"].str.upper().isin(fam_upper)]["ATC.code"]
    )

    # Find medication columns that start with any of those prefixes
    med_confounders = [
        col for col in available_med_cols
        if any(col.startswith(prefix) for prefix in atc_prefixes)
    ]

    # Combine: extra demographics/labs + medication ATC cols + Charlson
    confounders = list(dict.fromkeys(extra + med_confounders + CHARLSON_COLS))
    return confounders


# ===========================================================================
# Step 3 – Three-scenario matching runner
# ===========================================================================

def run_three_scenarios(
    df: pd.DataFrame,
    column_dt: pd.DataFrame,
    confounders: list[str],
    extra_confounders: list[str],
) -> pd.DataFrame:
    """
    Run the matching benchmark under three preprocessing scenarios:

    Original   : raw confounder columns from the paired dataset
    Binary     : normalised ATC + raw extra confounders appended
    Normalised : normalised ATC columns only (extra cols excluded)

    Parameters
    ----------
    df              : full paired DataFrame (contains y and t)
    column_dt       : preprocessed / normalised feature DataFrame
    confounders     : full confounder list (ATC families + demographics + Charlson)
    extra_confounders: the short list of demographic/lab confounders (no ATC)

    Returns
    -------
    pd.DataFrame with a 'Process' column tagging each scenario
    """
    y = np.array(df["y"]).astype("float32").squeeze()
    t = df["t"]

    all_parts: list[pd.DataFrame] = []

    # -- Scenario 1: Original (raw ATC duration values) --
    present_conf = [c for c in confounders if c in df.columns]
    X1 = df[present_conf].copy().astype(float)
    res1, _ = apply_all_matching_methods(X1, t, y)
    res1["Process"] = "Original"
    all_parts.append(res1)

    # -- Scenario 2: Binary (normalised ATC + extra cols appended) --
    norm_present = [c for c in confounders if c in column_dt.columns]
    X2 = column_dt[norm_present].copy()
    extra_present = [c for c in extra_confounders if c in df.columns]
    X2[extra_present] = df[extra_present].values
    res2, _ = apply_all_matching_methods(X2.astype(float), t, y)
    res2["Process"] = "Binary"
    all_parts.append(res2)

    # -- Scenario 3: Normalised (normalised ATC only) --
    X3 = column_dt[norm_present].copy()
    res3, _ = apply_all_matching_methods(X3.astype(float), t, y)
    res3["Process"] = "Normalised"
    all_parts.append(res3)

    return pd.concat(all_parts, ignore_index=True, sort=False)


# ===========================================================================
# Step 4 – Per-pair orchestration
# ===========================================================================

def run_pair(
    work_dir: Path,
    positive_drug: str,
    control_drug: str,
    atc_codes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full benchmark for one (positive_drug, control_drug) pair.

    Returns a DataFrame with columns:
        Methodology | Initial SMD | Mean SMD | SMD > 0.2 |
        Matched patients 1 | Matched patients 0 | Total patients | Ratio |
        Process | DAG | Drug | Control
    """
    logger.info("  pair: %s  vs  %s", positive_drug, control_drug)
    cfg = DRUG_CONFIG[positive_drug]

    df, column_dt = load_and_preprocess(work_dir, positive_drug, control_drug, cfg)

    # Identify available ATC medication columns in this pair
    lab_dem_com = set(LAB_COLS + DEM_COLS + CHARLSON_COLS + ["pre_AKI", "anchor_age"])
    available_med_cols = [
        c for c in column_dt.columns if c not in lab_dem_com
    ]

    all_dag_results: list[pd.DataFrame] = []

    for dag_source in cfg["dags"]:
        confounders     = build_confounder_list(
            positive_drug, dag_source, atc_codes_df, available_med_cols
        )
        extra_conf = cfg["dags"][dag_source]["extra_confounders"]

        results = run_three_scenarios(df, column_dt, confounders, extra_conf)
        results["DAG"]     = dag_source
        results["Drug"]    = positive_drug.capitalize()
        results["Control"] = control_drug.capitalize()
        all_dag_results.append(results)

    return pd.concat(all_dag_results, ignore_index=True, sort=False)


# ===========================================================================
# Step 5 – Full-run orchestrator
# ===========================================================================

def run_benchmark(
    work_dir: Path,
    atc_csv_path: Path,
    out_path: Path,
    positive_drugs: list[str] | None = None,
    control_drugs: list[str] | None = None,
) -> None:
    """
    Run the matching benchmark for all (or a subset of) drug pairs and
    save results to an Excel workbook with one sheet per positive drug.
    """
    positives = positive_drugs or POSITIVE_DRUGS
    controls  = control_drugs  or NEGATIVE_DRUGS

    # Load ATC code reference (3-char family level)
    atc_df_raw = pd.read_csv(atc_csv_path)
    atc_df_raw = atc_df_raw.loc[:, ~atc_df_raw.columns.str.startswith("Unnamed")]
    atc_families = atc_df_raw[atc_df_raw["ATC.code"].str.len() == 3].copy()
    atc_families["Name"] = atc_families["Name"].str.upper()

    # Collect results per positive drug
    drug_results: dict[str, pd.DataFrame] = {d: [] for d in positives}

    total = len(positives) * len(controls)
    done  = 0
    for pos, ctrl in product(positives, controls):
        logger.info("[%d/%d]  %s  vs  %s", done + 1, total, pos, ctrl)
        try:
            pair_df = run_pair(work_dir, pos, ctrl, atc_families)
            drug_results[pos].append(pair_df)
        except FileNotFoundError as exc:
            logger.warning("  SKIP — %s", exc)
        except Exception as exc:
            logger.error("  ERROR in %s vs %s: %s", pos, ctrl, exc, exc_info=True)
        done += 1

    # Write Excel workbook
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for drug in positives:
            frames = drug_results[drug]
            if not frames:
                logger.warning("No results for %s — skipping sheet", drug)
                continue
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined.to_excel(writer, sheet_name=drug.capitalize(), index=False)
            logger.info("  sheet '%s': %d rows", drug.capitalize(), len(combined))

    logger.info("Benchmark complete. Results saved to %s", out_path)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full matching benchmark for all drug pairs."
    )
    p.add_argument(
        "--work-dir", required=True, type=Path,
        help="Directory containing the 24 paired parquet datasets.",
    )
    p.add_argument(
        "--atc-csv", default=None, type=Path,
        help="Path to atc_codes.csv.  Defaults to <work-dir>/atc_codes.csv.",
    )
    p.add_argument(
        "--out", default="sml_mean.xlsx", type=Path,
        help="Output Excel file path (default: sml_mean.xlsx).",
    )
    p.add_argument(
        "--drug", default=None,
        help="Run only for this positive drug (default: all 8).",
    )
    p.add_argument(
        "--control", default=None,
        help="Run only for this control drug (default: all 3).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    work_dir  = args.work_dir.resolve()
    atc_csv   = (args.atc_csv.resolve() if args.atc_csv
                 else work_dir / "atc_codes.csv")
    out_path  = args.out if args.out.is_absolute() else Path.cwd() / args.out

    run_benchmark(
        work_dir      = work_dir,
        atc_csv_path  = atc_csv,
        out_path      = out_path,
        positive_drugs = [args.drug]    if args.drug    else None,
        control_drugs  = [args.control] if args.control else None,
    )


if __name__ == "__main__":
    main()
