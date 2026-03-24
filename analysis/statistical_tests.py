"""
analysis/statistical_tests.py
-------------------------------
Statistical characterisation of Heterogeneous Treatment Effects (HTE).

Replaces the "Interpretability" and "Summarization" sections that were
copy-pasted verbatim for each of the 8 positive drugs in ITE_clustering.ipynb.

Pipeline per drug
-----------------
1. Load the `_ite.parquet.gzip` produced by hte_estimation.py.
2. Classify patients: ITE < -0.1 → Protective, -0.1–0.1 → No effect, > 0.1 → Adverse.
3. Kruskal-Wallis test across ITE classes for:
   - Vitals     (Big_bp, Small_bp)                → 1×2 subplot PDF
   - Demographics (anchor_age, Weight (Lbs))       → 1×2 subplot PDF
   - Labs        (glucose, sodium, …, aniongap)     → 2×4 subplot PDF
4. Apply clinical bins (ACC/AHA guidelines) to all continuous variables.
5. Summarise ITE by category: count, mean_ITE, std_ITE.
6. Save one Excel workbook per drug with one sheet per feature group.

Usage
-----
    python statistical_tests.py --ite-dir /results/hte --out-dir /results/stats

    # Single drug
    python statistical_tests.py --ite-dir /results/hte --out-dir /results/stats \\
        --drug ibuprofen
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import kruskal, ttest_ind

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — ITE classification threshold
# ---------------------------------------------------------------------------

ITE_LOW  = -0.1    # below → Protective effect
ITE_HIGH =  0.1    # above → Adverse effect
ITE_CLASSES = ["Protective effect", "No effect", "Adverse effect"]

# ---------------------------------------------------------------------------
# Clinical bin definitions (ACC/AHA 2017 guidelines + standard EHR ranges)
# ---------------------------------------------------------------------------

CLINICAL_BINS: dict[str, tuple[list, list]] = {
    # (bins, labels)
    "age":        ([0, 17, 39, 59, 79, 150],         ["<18", "18–39", "40–59", "60–79", "≥80"]),
    "weight":     ([0, 60, 80, 100, float("inf")],   ["<60 kg", "60–79 kg", "80–99 kg", "≥100 kg"]),
    "glucose":    ([0, 100, 126, float("inf")],       ["<100", "100–125", "≥126"]),
    "sodium":     ([0, 135, 145, float("inf")],       ["<135", "135–145", ">145"]),
    "creatinine": ([0, 1.2, 2.0, float("inf")],       ["<1.2", "1.2–1.9", "≥2.0"]),
    "potassium":  ([0, 3.5, 5.0, float("inf")],       ["<3.5", "3.5–5.0", ">5.0"]),
    "bun":        ([0, 7, 20, float("inf")],           ["<7", "7–20", ">20"]),
    "bicarb":     ([0, 22, 29, float("inf")],          ["<22", "22–29", ">29"]),
    "chloride":   ([0, 98, 106, float("inf")],         ["<98", "98–106", ">106"]),
    "aniongap":   ([0, 8, 16, float("inf")],           ["<8", "8–16", ">16"]),
    "sbp":        ([0, 120, 130, 140, float("inf")],  ["<120", "120–129", "130–139", "≥140"]),
    "dbp":        ([0, 80, 90, float("inf")],          ["<80", "80–89", "≥90"]),
}

# Mapping: clinical bin name → DataFrame column(s)
# For t0t1 lab_repr we use the "_last" suffix; for mean repr we use the plain name.
COLUMN_MAP_MEAN: dict[str, str] = {
    "age":        "anchor_age",
    "weight":     "Weight (Lbs)",
    "glucose":    "glucose",
    "sodium":     "sodium",
    "creatinine": "creatinine",
    "potassium":  "potassium",
    "bun":        "bun",
    "bicarb":     "bicarbonate",
    "chloride":   "chloride",
    "aniongap":   "aniongap",
    "sbp":        "Big_bp",
    "dbp":        "Small_bp",
}

COLUMN_MAP_T0T1: dict[str, str] = {
    "age":        "anchor_age",
    "weight":     "Weight (Lbs)",
    "glucose":    "glucose_last",
    "sodium":     "sodium_last",
    "creatinine": "creatinine_last",
    "potassium":  "potassium_last",
    "bun":        "bun_last",
    "bicarb":     "bicarbonate_last",
    "chloride":   "chloride_last",
    "aniongap":   "aniongap_last",
    "sbp":        "Big_bp",
    "dbp":        "Small_bp",
}

# Feature groups for boxplots
VITAL_FEATURES_MEAN = {
    "Diastolic blood pressure": "Big_bp",
    "Systolic blood pressure":  "Small_bp",
}

DEM_FEATURES = {
    "Age":           "anchor_age",
    "Weight (Lbs)":  "Weight (Lbs)",
}

LAB_FEATURES_MEAN = {
    "glucose (mg/dL)":      "glucose",
    "sodium (mEq/L)":       "sodium",
    "creatinine (mg/dL)":   "creatinine",
    "potassium (mEq/L)":    "potassium",
    "bun (mg/dL)":          "bun",
    "bicarbonate (mEq/L)":  "bicarbonate",
    "chloride (mEq/L)":     "chloride",
    "aniongap (mEq/L)":     "aniongap",
}

LAB_FEATURES_T0T1 = {
    "glucose (mg/dL)":      "glucose_last",
    "sodium (mEq/L)":       "sodium_last",
    "creatinine (mg/dL)":   "creatinine_last",
    "potassium (mEq/L)":    "potassium_last",
    "bun (mg/dL)":          "bun_last",
    "bicarbonate (mEq/L)":  "bicarbonate_last",
    "chloride (mEq/L)":     "chloride_last",
    "aniongap (mEq/L)":     "aniongap_last",
}


# ===========================================================================
# Step 1: ITE classification
# ===========================================================================

def classify_ite(
    df: pd.DataFrame,
    ite_col: str = "ite",
    low: float = ITE_LOW,
    high: float = ITE_HIGH,
) -> pd.DataFrame:
    """
    Add a 'Class' column: Protective effect / No effect / Adverse effect.

    Parameters
    ----------
    df      : DataFrame with an ITE column
    ite_col : name of the ITE column
    low     : lower threshold (ITE < low → Protective)
    high    : upper threshold (ITE > high → Adverse)

    Returns
    -------
    df with a new 'Class' column (does NOT modify df in-place)
    """
    df = df.copy()
    df["Class"] = pd.cut(
        df[ite_col],
        bins=[-np.inf, low, high, np.inf],
        labels=ITE_CLASSES,
    )
    return df


# ===========================================================================
# Step 2: Kruskal-Wallis tests
# ===========================================================================

def kruskal_wallis_tests(
    df: pd.DataFrame,
    feature_col_map: dict[str, str],
    class_col: str = "Class",
) -> dict[str, float]:
    """
    Run Kruskal-Wallis (≥3 groups) or Welch's t-test (2 groups) per feature.

    Parameters
    ----------
    df              : DataFrame with Class column
    feature_col_map : {display_label: df_column_name}

    Returns
    -------
    {display_label: p_value}
    """
    p_values: dict[str, float] = {}
    classes = [c for c in ITE_CLASSES if c in df[class_col].values]

    for label, col in feature_col_map.items():
        if col not in df.columns:
            p_values[label] = np.nan
            continue
        groups = [df[df[class_col] == cls][col].dropna() for cls in classes]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            p_values[label] = np.nan
        elif len(groups) == 2:
            _, p = ttest_ind(groups[0], groups[1], equal_var=False)
            p_values[label] = float(p)
        else:
            _, p = kruskal(*groups)
            p_values[label] = float(p)
    return p_values


def _format_p(p: float) -> str:
    """Format p-value for subplot title."""
    if np.isnan(p):
        return "p = N/A"
    if p < 1e-5:
        return f"p = {p:.1e}"
    return f"p = {p:.5f}"


# ===========================================================================
# Step 3: Plotly boxplot panels
# ===========================================================================

def plot_boxplots(
    df: pd.DataFrame,
    feature_col_map: dict[str, str],
    drug_name: str,
    plot_type: str,
    out_dir: Path,
    n_rows: int = 1,
    n_cols: int | None = None,
    fig_height: int = 600,
    fig_width: int = 1200,
    class_col: str = "Class",
) -> None:
    """
    Create Plotly boxplot subplots for a feature group, with Kruskal-Wallis
    p-values in subplot titles.  Saves as PDF via kaleido.

    Parameters
    ----------
    df              : DataFrame with Class and feature columns
    feature_col_map : {display_label: df_column_name}
    drug_name       : used in the output filename
    plot_type       : short label for the filename (e.g. "vital", "dem", "labs")
    out_dir         : directory to write the PDF
    n_rows, n_cols  : subplot grid dimensions (auto-computed if n_cols is None)
    fig_height/width: figure dimensions in pixels
    """
    features = list(feature_col_map.keys())
    n = len(features)
    if n_cols is None:
        n_cols = n if n_rows == 1 else 4

    p_values = kruskal_wallis_tests(df, feature_col_map, class_col)

    titles = [f"{f}<br>{_format_p(p_values[f])}" for f in features]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

    classes_present = [c for c in ITE_CLASSES if c in df[class_col].values]

    for i, (label, col) in enumerate(feature_col_map.items()):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        if col not in df.columns:
            continue
        for cls in classes_present:
            fig.add_trace(
                go.Box(
                    y=df[df[class_col] == cls][col],
                    name=str(cls),
                    boxmean=True,
                    showlegend=(i == 0),
                ),
                row=row, col=col_pos,
            )

    fig.update_layout(height=fig_height, width=fig_width, showlegend=False)

    out_path = out_dir / f"{drug_name}_ite_{plot_type}.pdf"
    try:
        fig.write_image(str(out_path))
        logger.info("    Plot saved: %s", out_path.name)
    except Exception as exc:
        logger.warning("    Could not write PDF (%s) — kaleido may not be installed: %s", out_path.name, exc)


# ===========================================================================
# Step 4: Clinical binning
# ===========================================================================

def apply_clinical_bins(
    df: pd.DataFrame,
    lab_repr: str = "mean",
) -> pd.DataFrame:
    """
    Add clinical category columns (_cat suffix) for all continuous variables.

    Parameters
    ----------
    df       : DataFrame with raw (unscaled) lab/vital/demographic values
    lab_repr : "mean" or "t0t1" (determines which column names to use)

    Returns
    -------
    df with new category columns (does NOT modify df in-place)
    """
    df = df.copy()
    col_map = COLUMN_MAP_T0T1 if lab_repr == "t0t1" else COLUMN_MAP_MEAN

    for feature, (bins, labels) in CLINICAL_BINS.items():
        raw_col = col_map.get(feature)
        if raw_col is None or raw_col not in df.columns:
            continue
        df[f"{feature}_cat"] = pd.cut(
            df[raw_col],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )

    # Gender
    if "gender" in df.columns:
        df["gender_cat"] = df["gender"].astype("category")

    return df


# ===========================================================================
# Step 5: ITE summary tables
# ===========================================================================

def summarize_ite_by_category(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    ite_col: str = "ite",
) -> dict[str, pd.DataFrame]:
    """
    Create ITE summary tables (count / mean_ITE / std_ITE) per category column.

    Parameters
    ----------
    df              : DataFrame with ITE column and *_cat columns
    categorical_cols: list of category columns to summarise (default: all *_cat columns)
    ite_col         : ITE column name

    Returns
    -------
    dict of {col → summary DataFrame}
    """
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c.endswith("_cat")]

    tables: dict[str, pd.DataFrame] = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        summary = (
            df.groupby(col, observed=True)[ite_col]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "n_patients", "mean": "mean_ITE", "std": "std_ITE"})
        )
        tables[col] = summary
    return tables


def save_summary_tables(
    tables: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Save all summary tables to one Excel workbook (one sheet per feature)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for col, tbl in tables.items():
            sheet = col[:31]  # Excel limit
            tbl.to_excel(writer, sheet_name=sheet, index=False)
    logger.info("    Summary tables: %s", out_path.name)


# ===========================================================================
# Step 6: Per-drug full analysis pipeline
# ===========================================================================

def run_drug_analysis(
    drug: str,
    df_ite: pd.DataFrame,
    lab_repr: str,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Run the complete statistical analysis for one drug.

    Steps: classify → boxplots (vitals, demographics, labs) →
           clinical bins → ITE summaries → save Excel.

    Parameters
    ----------
    drug     : positive drug name
    df_ite   : DataFrame with 'ite' column and raw lab/vital values
    lab_repr : "mean" or "t0t1"
    out_dir  : directory to write plots and Excel tables

    Returns
    -------
    dict of summary tables
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Analysing %s …", drug)

    # 1. Classify
    df = classify_ite(df_ite)

    # 2. Boxplots — Vitals (1 × 2)
    plot_boxplots(
        df, VITAL_FEATURES_MEAN, drug, "vital",
        out_dir, n_rows=1, n_cols=2, fig_height=600,
    )

    # 3. Boxplots — Demographics (1 × 2)
    plot_boxplots(
        df, DEM_FEATURES, drug, "dem",
        out_dir, n_rows=1, n_cols=2, fig_height=600,
    )

    # 4. Boxplots — Labs (2 × 4)
    lab_features = LAB_FEATURES_T0T1 if lab_repr == "t0t1" else LAB_FEATURES_MEAN
    plot_boxplots(
        df, lab_features, drug, "labs",
        out_dir, n_rows=2, n_cols=4, fig_height=800,
    )

    # 5. Clinical binning
    df_binned = apply_clinical_bins(df, lab_repr=lab_repr)

    # 6. ITE summary tables
    cat_cols = [
        "age_cat", "gender_cat", "weight_cat", "glucose_cat", "sodium_cat",
        "creatinine_cat", "potassium_cat", "bun_cat", "bicarb_cat",
        "chloride_cat", "aniongap_cat", "sbp_cat", "dbp_cat",
    ]
    tables = summarize_ite_by_category(df_binned, cat_cols)

    # 7. Save Excel
    save_summary_tables(tables, out_dir / f"{drug}_ite_summary.xlsx")

    return tables


# ===========================================================================
# All-drugs orchestrator
# ===========================================================================

def run_all_analyses(
    ite_dir: Path,
    out_dir: Path,
    drugs: list[str] | None = None,
) -> None:
    """
    Run statistical analysis for all drugs that have an _ite.parquet.gzip file.

    Parameters
    ----------
    ite_dir : directory containing `<drug>_<control>_ite.parquet.gzip` files
    out_dir : directory to write PDF plots and Excel summaries
    drugs   : optional list of drugs to process (default: all found files)
    """
    from hte_estimation import HTE_CONFIG

    drugs_to_run = drugs or list(HTE_CONFIG.keys())
    out_dir.mkdir(parents=True, exist_ok=True)

    for drug in drugs_to_run:
        if drug not in HTE_CONFIG:
            logger.warning("No HTE config for %s — skipping", drug)
            continue

        cfg = HTE_CONFIG[drug]
        ite_path = ite_dir / f"{drug}_{cfg['control']}_ite.parquet.gzip"
        if not ite_path.exists():
            logger.warning("ITE file not found: %s — run hte_estimation.py first", ite_path.name)
            continue

        logger.info("[drug: %s]  loading %s", drug, ite_path.name)
        df_ite = pd.read_parquet(ite_path)
        run_drug_analysis(drug, df_ite, cfg["lab_repr"], out_dir / drug)

    logger.info("Statistical analysis complete. Results in %s", out_dir)


# ===========================================================================
# Utility: print a concise ITE class distribution table
# ===========================================================================

def ite_class_distribution(df: pd.DataFrame, ite_col: str = "ite") -> pd.DataFrame:
    """Return a small DataFrame showing count and % per ITE class."""
    df = classify_ite(df, ite_col)
    dist = df["Class"].value_counts().rename("n").reset_index()
    dist.columns = ["Class", "n"]
    dist["pct"] = (dist["n"] / len(df) * 100).round(1)
    return dist.sort_values("Class")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run statistical analysis of HTE results for all drugs."
    )
    p.add_argument("--ite-dir", required=True, type=Path,
                   help="Directory containing _ite.parquet.gzip files.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write PDF plots and Excel summaries.")
    p.add_argument("--drug", default=None,
                   help="Run only for this drug (default: all).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_all_analyses(
        ite_dir = args.ite_dir.resolve(),
        out_dir = args.out_dir.resolve(),
        drugs   = [args.drug] if args.drug else None,
    )


if __name__ == "__main__":
    main()
