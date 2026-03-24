"""
02_build_noaki_cohort.py
------------------------
Build the **no-AKI cohort** for a given suspect or negative-control drug.

The no-AKI cohort consists of patients who received the drug but did NOT
develop AKI.  The key difference from the AKI cohort (01_build_aki_cohort.py)
is the absence of a KDIGO AKI-worsening filter: patients are selected based
on the DURING-treatment flag alone, without any AKI event.

Produces:
  <work_dir>/<drug>_noaki_mean.parquet.gzip   (mean lab representation)
  <work_dir>/<drug>_noaki_t0t1.parquet.gzip   (first/last lab representation)

Usage
-----
    python 02_build_noaki_cohort.py --drug vancomycin --work-dir /path/to/data
    python 02_build_noaki_cohort.py --drug all        --work-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from atc_mapping import (
    apply_name_unification,
    build_atc_lookup,
    map_drugs_to_atc,
    apply_atc_from_cache,
)
from config import (
    ALL_DRUGS,
    ATC_SIMILARITY_THRESHOLD,
    CHARLSON_COMORBIDITY_COLS,
    CHARLSON_SELECT_COLS,
    CHEM_DROP_COLS,
    CONCOMITANT_ATC_CODES,
    LAB_COLS,
    MED_DROP_COLS,
    OMAR_RESULT_NAMES,
    PATIENTS_DROP_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers  (identical to 01_build_aki_cohort.py – shared in utils.py
# in a real package; duplicated here for standalone clarity)
# ---------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")
    if path.suffix in (".parquet", ".gzip"):
        return pd.read_parquet(path)
    if path.name.endswith(".csv.zip"):
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="gzip")
    logger.info("  saved  %s  (%d rows, %d cols)", path.name, len(df), df.shape[1])


def _resolve_input(work_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of these input files found in {work_dir}:\n  " + "\n  ".join(candidates)
    )


# ===========================================================================
# Step 1 – Charlson (no KDIGO step for no-AKI cohort)
# ===========================================================================

def process_charlson_noaki(
    work_dir: Path, drug: str, cfg: dict,
) -> tuple[pd.DataFrame, list]:
    """
    Load the no-AKI Charlson table, filter to the DURING-treatment window,
    binarise comorbidities, and return (cls_df, cohort_subject_ids).
    """
    logger.info("[Charlson-noAKI] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()
    during_label = f"DURING {drug_label}"

    charlson_path = _resolve_input(
        work_dir,
        [
            f"charlson_{slug}_noAKI.parquet.gzip",
            f"charlson_{slug}_control.parquet.gzip",
            f"charlson_{slug}_noAKI.csv",
            f"charlson_{slug}.csv",
        ],
    )
    charlson = _read(charlson_path)

    # Keep only DURING-treatment rows
    if flag_col in charlson.columns:
        charlson = charlson[charlson[flag_col] == during_label].copy()

    cols = [c for c in CHARLSON_SELECT_COLS if c in charlson.columns]
    ch = charlson[cols].drop_duplicates()

    comorbidity_cols = [c for c in CHARLSON_COMORBIDITY_COLS if c in ch.columns]
    agg_cols = [c for c in comorbidity_cols if c != "charlson_comorbidity_index"]
    cci_in_cols = "charlson_comorbidity_index" in ch.columns

    group_cols = agg_cols + (["charlson_comorbidity_index"] if cci_in_cols else [])
    cls = (
        ch.groupby("subject_id")[group_cols]
        .sum()
        .reset_index()
    )

    counts = (
        ch.groupby("subject_id")["subject_id"]
        .count()
        .rename("_count")
        .reset_index()
    )
    cls = cls.merge(counts, on="subject_id")
    if cci_in_cols:
        cls["charlson_comorbidity_index"] = (
            cls["charlson_comorbidity_index"] / cls["_count"]
        ).round()
    cls.drop(columns=["_count"], inplace=True)
    cls[agg_cols] = cls[agg_cols].clip(upper=1)

    if "age_score" in ch.columns:
        age_df = ch.groupby("subject_id")["age_score"].first().reset_index()
        cls = cls.merge(age_df, on="subject_id", how="left")

    subject_ids = cls["subject_id"].unique().tolist()
    logger.info("  %d patients in Charlson-noAKI", len(subject_ids))
    return cls, subject_ids


# ===========================================================================
# Step 2 – Chemistry
# ===========================================================================

def process_chemistry_noaki(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce mean and first/last chemistry DataFrames for the no-AKI cohort.
    Returns (cm_mean, cm_t0t1).
    """
    logger.info("[Chemistry-noAKI] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()
    during_label = f"DURING {drug_label}"

    chem_path = _resolve_input(
        work_dir,
        [
            f"chemistry_t0_{slug}_noAKI.csv",
            f"chem_{slug}_t0_noAKI.parquet.gzip",
            f"chemistry_{slug}_noAKI.csv",
            f"chem_vancomycin_t0_noAKI.parquet.gzip",   # legacy name pattern
        ],
    )
    chem = _read(chem_path)

    # Drop unused columns
    chem.drop(
        columns=[c for c in CHEM_DROP_COLS if c in chem.columns],
        inplace=True,
        errors="ignore",
    )

    # Filter to Charlson-derived cohort
    chem = chem[chem["subject_id"].isin(subject_ids)].copy()

    # DURING-treatment window
    if flag_col in chem.columns:
        cm_during = chem[chem[flag_col] == during_label].copy()
        cm_during.drop(columns=[flag_col], inplace=True, errors="ignore")
    else:
        cm_during = chem.copy()

    available_lab_cols = [c for c in LAB_COLS if c in cm_during.columns]

    # ---- Mean ----
    cm_mean = (
        cm_during
        .dropna(subset=["glucose"])
        .dropna(subset=["aniongap"] if "aniongap" in cm_during.columns else [])
        .groupby("subject_id")[available_lab_cols]
        .mean()
        .reset_index()
    )

    # ---- First / last ----
    sort_col = "charttime" if "charttime" in cm_during.columns else "chartdatetime"
    if sort_col in cm_during.columns:
        cm_sorted = cm_during.sort_values(["subject_id", sort_col])
    else:
        cm_sorted = cm_during.copy()

    agg_dict = {}
    for col in available_lab_cols:
        agg_dict[f"{col}_first"] = (col, "first")
        agg_dict[f"{col}_last"] = (col, "last")

    cm_t0t1 = cm_sorted.groupby("subject_id").agg(**agg_dict).reset_index()
    cm_t0t1 = cm_t0t1.dropna(subset=["glucose_first"])
    if "bicarbonate_first" in cm_t0t1.columns:
        cm_t0t1 = cm_t0t1.dropna(subset=["bicarbonate_first"])

    logger.info(
        "  cm_mean: %d | cm_t0t1: %d patients",
        cm_mean["subject_id"].nunique(),
        cm_t0t1["subject_id"].nunique(),
    )
    return cm_mean, cm_t0t1


# ===========================================================================
# Step 3 – Vital signs (OMAR)
# ===========================================================================

def process_omar_noaki(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list,
) -> pd.DataFrame:
    """
    Load vitals for the no-AKI cohort (DURING window only, no KDIGO filter).
    """
    logger.info("[OMAR-noAKI] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()
    during_label = f"DURING {drug_label}"

    omar_path = _resolve_input(
        work_dir,
        [
            f"omar_{slug}_noAKI.parquet.gzip",
            f"omar_{slug}_control.parquet.gzip",
            f"omar_{slug}.csv",
        ],
    )
    omar = _read(omar_path)
    omar = omar[omar["subject_id"].isin(subject_ids)].copy()

    # DURING window
    if flag_col in omar.columns:
        omar_during = omar[omar[flag_col] == during_label].copy()
        omar_during.drop(columns=[flag_col], inplace=True, errors="ignore")
    else:
        omar_during = omar.copy()

    omar_during = omar_during[
        omar_during["result_name"].isin(OMAR_RESULT_NAMES)
    ][["subject_id", "result_name", "result_value"]].copy()

    # BMI / Weight
    omar_bmi = omar_during[
        omar_during["result_name"].isin(["BMI (kg/m2)", "Weight (Lbs)"])
    ].copy()
    omar_bmi["result_value"] = pd.to_numeric(omar_bmi["result_value"], errors="coerce")
    omar_bmi_wide = pd.pivot_table(
        omar_bmi,
        index="subject_id",
        columns="result_name",
        values="result_value",
        aggfunc="mean",
        fill_value=0,
    ).round()

    # Blood Pressure
    omar_bp = omar_during[omar_during["result_name"] == "Blood Pressure"].copy()
    omar_bp[["Small_bp", "Big_bp"]] = omar_bp["result_value"].str.split("/", expand=True)
    for col in ["Small_bp", "Big_bp"]:
        omar_bp[col] = pd.to_numeric(omar_bp[col], errors="coerce")
    omar_bp_wide = (
        omar_bp.groupby("subject_id")[["Big_bp", "Small_bp"]]
        .mean()
        .round()
        .reset_index()
    )

    omar_out = omar_bmi_wide.merge(omar_bp_wide, on="subject_id", how="outer")
    logger.info("  %d patients in OMAR-noAKI", omar_out["subject_id"].nunique())
    return omar_out


# ===========================================================================
# Step 4 – Medications
# ===========================================================================

def process_medications_noaki(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list,
    atc_csv_path: Path,
    mapping_cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Same as the AKI version but reads from the noAKI medication file.
    """
    logger.info("[Medications-noAKI] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()
    during_label = f"DURING {drug_label}"

    med_path = _resolve_input(
        work_dir,
        [
            f"medication_{slug}_noAKI.parquet.gzip",
            f"medication_{slug}_control.parquet.gzip",
            f"medication_{slug}_noAKI.csv.zip",
        ],
    )
    med = _read(med_path)

    med.drop(
        columns=[c for c in MED_DROP_COLS if c in med.columns],
        inplace=True,
        errors="ignore",
    )

    med = med[med["subject_id"].isin(subject_ids)].copy()

    if flag_col in med.columns:
        med = med[med[flag_col] == during_label].copy()

    med = apply_name_unification(med)

    if mapping_cache_path is not None and mapping_cache_path.exists():
        med = apply_atc_from_cache(med, mapping_cache_path)
    else:
        atc_lookup = build_atc_lookup(atc_csv_path)
        med = map_drugs_to_atc(
            med, atc_lookup,
            threshold=ATC_SIMILARITY_THRESHOLD,
            mapping_cache_path=mapping_cache_path,
        )

    med = med[med["ATC"] != "N/A"].copy()

    if "drug_stoptime" in med.columns and "drug_starttime" in med.columns:
        med["drug_duration_days"] = (
            pd.to_datetime(med["drug_stoptime"]) - pd.to_datetime(med["drug_starttime"])
        ).dt.days
    elif "drug_duration_days" not in med.columns:
        med["drug_duration_days"] = 1

    if "dose_val_rx" in med.columns:
        med = med[~med["dose_val_rx"].isnull()]

    med = med.explode("ATC")
    med = med[med["ATC"] != "N/A"].copy()

    med_atc = med[["subject_id", "ATC", "drug_duration_days"]]
    med_pivot = pd.pivot_table(
        med_atc,
        index="subject_id",
        columns="ATC",
        values="drug_duration_days",
        aggfunc="sum",
        fill_value=0,
    )

    col_ls = []
    for atc_prefix in CONCOMITANT_ATC_CODES:
        for col in med_pivot.columns:
            if col.startswith(atc_prefix) or col == atc_prefix:
                col_ls.append(col)

    if col_ls:
        med_pivot = med_pivot[col_ls]
        med_pivot = med_pivot.loc[:, ~med_pivot.columns.duplicated()].copy()

    med_out = med_pivot.reset_index()
    logger.info(
        "  %d patients, %d ATC features",
        med_out["subject_id"].nunique(),
        med_out.shape[1] - 1,
    )
    return med_out


# ===========================================================================
# Step 5 – Patients
# ===========================================================================

def process_patients_noaki(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list,
) -> pd.DataFrame:
    slug = cfg["slug"]
    patients_path = _resolve_input(
        work_dir,
        [
            f"patients_{slug}_noAKI.csv",
            f"patients_{slug}.csv",
            "patients.csv",
        ],
    )
    patients = _read(patients_path)
    patients = patients[patients["subject_id"].isin(subject_ids)].copy()
    patients.drop(
        columns=[c for c in PATIENTS_DROP_COLS if c in patients.columns],
        inplace=True,
        errors="ignore",
    )
    logger.info("  %d patients in patients table", patients["subject_id"].nunique())
    return patients


# ===========================================================================
# Step 6 – Final merge
# ===========================================================================

def build_final_noaki_dataset(
    omar_during: pd.DataFrame,
    cm: pd.DataFrame,
    cls: pd.DataFrame,
    med_atc: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    # Intersect
    sid = set(omar_during["subject_id"].unique())
    for df in [cm, cls, med_atc, patients]:
        sid &= set(df["subject_id"].unique())
    sid = list(sid)
    logger.info("  intersection: %d patients with all features present", len(sid))

    dfs = [omar_during, cm, cls, med_atc, patients]
    dfs = [d[d["subject_id"].isin(sid)] for d in dfs]

    final = dfs[0]
    for d in dfs[1:]:
        overlap = [c for c in d.columns if c in final.columns and c != "subject_id"]
        d = d.drop(columns=overlap)
        final = final.merge(d, on="subject_id", how="inner")

    return final


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_noaki_pipeline(
    drug: str,
    work_dir: Path,
    atc_csv_path: Path,
    mapping_cache_path: Path | None = None,
) -> None:
    if drug not in ALL_DRUGS:
        raise ValueError(f"Unknown drug '{drug}'. Choices: {list(ALL_DRUGS)}")

    cfg = ALL_DRUGS[drug]
    logger.info("=" * 60)
    logger.info("no-AKI pipeline  |  drug = %s", drug)
    logger.info("=" * 60)

    # 1. Charlson (defines the cohort for no-AKI)
    cls, subject_ids = process_charlson_noaki(work_dir, drug, cfg)

    # 2. Chemistry
    cm_mean, cm_t0t1 = process_chemistry_noaki(work_dir, drug, cfg, subject_ids)

    # Update subject_ids from chemistry intersection
    chem_ids = set(cm_mean["subject_id"])
    subject_ids = list(set(subject_ids) & chem_ids)

    # 3. OMAR
    omar_during = process_omar_noaki(work_dir, drug, cfg, subject_ids)

    # Update from OMAR
    subject_ids = list(set(subject_ids) & set(omar_during["subject_id"]))

    # 4. Medications
    med_atc = process_medications_noaki(
        work_dir, drug, cfg, subject_ids,
        atc_csv_path=atc_csv_path,
        mapping_cache_path=mapping_cache_path,
    )

    # 5. Patients
    patients = process_patients_noaki(work_dir, drug, cfg, subject_ids)

    # 6. Final – mean labs
    final_mean = build_final_noaki_dataset(omar_during, cm_mean, cls, med_atc, patients)
    _write(final_mean, work_dir / f"{drug}_noaki_mean.parquet.gzip")

    # 6b. Final – t0-t1 labs
    final_t0t1 = build_final_noaki_dataset(omar_during, cm_t0t1, cls, med_atc, patients)
    _write(final_t0t1, work_dir / f"{drug}_noaki_t0t1.parquet.gzip")

    logger.info("Done: %s  (%d / %d patients)", drug, len(final_mean), len(final_t0t1))


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the no-AKI cohort parquet files for one or all drugs."
    )
    parser.add_argument("--drug", required=True,
                        help="Drug name or 'all'.")
    parser.add_argument("--work-dir", required=True, type=Path,
                        help="Directory containing raw MIMIC-IV files.")
    parser.add_argument("--atc-csv", default=None, type=Path,
                        help="Path to atc_codes.csv.")
    parser.add_argument("--mapping-cache", default=None, type=Path,
                        help="Path to validated Drug->ATC Excel file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work_dir: Path = args.work_dir.resolve()
    atc_csv: Path = (
        args.atc_csv.resolve() if args.atc_csv else work_dir / "atc_codes.csv"
    )
    drugs = list(ALL_DRUGS.keys()) if args.drug == "all" else [args.drug]
    for d in drugs:
        cache = (
            args.mapping_cache
            if args.mapping_cache
            else work_dir / f"atc_med_map_85_{d}_noAKI.xlsx"
        )
        run_noaki_pipeline(drug=d, work_dir=work_dir,
                           atc_csv_path=atc_csv, mapping_cache_path=cache)


if __name__ == "__main__":
    main()
