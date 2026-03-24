"""
03_merge_datasets.py
--------------------
Merge AKI and no-AKI cohorts for every suspect × control drug pair to produce
the **24 final analysis datasets** used in the CML pipeline.

For each of the 8 positive drugs × 3 negative control drugs:
  1. Load <positive_drug>_aki_*.parquet.gzip
  2. Load <control_drug>_noaki_*.parquet.gzip
  3. Remove patients who took BOTH the positive and control drug simultaneously.
  4. Align columns (keep only the intersection of concomitant-drug ATC columns).
  5. Add an ``outcome`` column: 1 = AKI patient, 0 = no-AKI patient.
  6. Concatenate and save to:
       <out_dir>/<positive_drug>_vs_<control_drug>_mean.parquet.gzip
       <out_dir>/<positive_drug>_vs_<control_drug>_t0t1.parquet.gzip

Usage
-----
    python 03_merge_datasets.py \\
        --work-dir /path/to/processed \\
        --out-dir  /path/to/final_datasets

    # To run only one pair:
    python 03_merge_datasets.py \\
        --work-dir /path/to/processed \\
        --out-dir  /path/to/final_datasets \\
        --positive vancomycin \\
        --control  lactulose
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import pandas as pd

from config import NEGATIVE_DRUGS, POSITIVE_DRUGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="gzip")
    logger.info("  saved  %s  (%d rows × %d cols)", path.name, len(df), df.shape[1])


def _remove_dual_exposed(
    aki_df: pd.DataFrame,
    noaki_df: pd.DataFrame,
    positive_drug: str,
    control_drug: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop patients who appear in BOTH cohorts (i.e. received both the suspect
    and the control drug during the same hospitalisation).
    """
    shared_ids = set(aki_df["subject_id"]) & set(noaki_df["subject_id"])
    if shared_ids:
        logger.info(
            "  removing %d patients present in both %s-AKI and %s-noAKI cohorts",
            len(shared_ids), positive_drug, control_drug,
        )
        aki_df = aki_df[~aki_df["subject_id"].isin(shared_ids)].copy()
        noaki_df = noaki_df[~noaki_df["subject_id"].isin(shared_ids)].copy()
    return aki_df, noaki_df


def _align_columns(
    aki_df: pd.DataFrame,
    noaki_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only the columns shared between both DataFrames (except subject_id
    and outcome which are handled separately).
    """
    common = set(aki_df.columns) & set(noaki_df.columns)
    # Always include subject_id
    common.add("subject_id")
    aki_df = aki_df[[c for c in aki_df.columns if c in common]].copy()
    noaki_df = noaki_df[[c for c in noaki_df.columns if c in common]].copy()
    return aki_df, noaki_df


def merge_pair(
    work_dir: Path,
    out_dir: Path,
    positive_drug: str,
    control_drug: str,
    lab_repr: str,   # "mean" or "t0t1"
) -> None:
    """
    Merge one (positive_drug, control_drug, lab_repr) triple.
    """
    aki_path = work_dir / f"{positive_drug}_aki_{lab_repr}.parquet.gzip"
    noaki_path = work_dir / f"{control_drug}_noaki_{lab_repr}.parquet.gzip"

    if not aki_path.exists():
        logger.warning("  SKIP – missing AKI file: %s", aki_path.name)
        return
    if not noaki_path.exists():
        logger.warning("  SKIP – missing noAKI file: %s", noaki_path.name)
        return

    aki_df = _read(aki_path)
    noaki_df = _read(noaki_path)

    # Step 1: remove patients in both cohorts
    aki_df, noaki_df = _remove_dual_exposed(
        aki_df, noaki_df, positive_drug, control_drug
    )

    # Step 2: align columns (keep intersection of ATC features)
    aki_df, noaki_df = _align_columns(aki_df, noaki_df)

    # Step 3: add outcome label
    aki_df = aki_df.copy()
    noaki_df = noaki_df.copy()
    aki_df["outcome"] = 1
    noaki_df["outcome"] = 0

    # Step 4: add treatment indicator (1 = suspect drug, 0 = control)
    aki_df["treatment"] = 1
    noaki_df["treatment"] = 0

    # Step 5: concatenate
    final = pd.concat([aki_df, noaki_df], ignore_index=True)

    # Step 6: drop rows with missing values in core feature columns
    core_cols = [c for c in final.columns if c not in ("subject_id", "outcome", "treatment")]
    before = len(final)
    final = final.dropna(subset=core_cols).reset_index(drop=True)
    after = len(final)
    if before != after:
        logger.info("  dropped %d rows with missing values", before - after)

    out_name = f"{positive_drug}_vs_{control_drug}_{lab_repr}.parquet.gzip"
    _write(final, out_dir / out_name)
    logger.info(
        "  %s vs %s (%s): %d AKI + %d noAKI = %d total patients",
        positive_drug, control_drug, lab_repr,
        aki_df["outcome"].sum(), (noaki_df["outcome"] == 0).sum(),
        len(final),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_merge_pipeline(
    work_dir: Path,
    out_dir: Path,
    positive_drugs: list[str] | None = None,
    control_drugs: list[str] | None = None,
) -> None:
    """
    Merge all (or a subset of) suspect × control drug pairs for both lab
    representations.
    """
    positives = positive_drugs or list(POSITIVE_DRUGS.keys())
    controls = control_drugs or list(NEGATIVE_DRUGS.keys())

    total = len(positives) * len(controls) * 2
    done = 0
    for pos, ctrl, repr_ in product(positives, controls, ["mean", "t0t1"]):
        logger.info(
            "[%d/%d]  %s  vs  %s  (%s)",
            done + 1, total, pos, ctrl, repr_,
        )
        merge_pair(work_dir, out_dir, pos, ctrl, repr_)
        done += 1

    logger.info("All %d dataset pairs produced in %s", done, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge AKI and no-AKI cohorts into the 24 final paired datasets."
        )
    )
    parser.add_argument(
        "--work-dir", required=True, type=Path,
        help="Directory containing <drug>_aki_*.parquet.gzip and "
             "<drug>_noaki_*.parquet.gzip files (output of steps 01 & 02).",
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="Directory where the 24 paired parquet files will be written.",
    )
    parser.add_argument(
        "--positive", default=None,
        help="Run only for this positive drug (default: all 8).",
    )
    parser.add_argument(
        "--control", default=None,
        help="Run only for this control drug (default: all 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_merge_pipeline(
        work_dir=args.work_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        positive_drugs=[args.positive] if args.positive else None,
        control_drugs=[args.control] if args.control else None,
    )


if __name__ == "__main__":
    main()
