# `matching/`

Propensity-score matching benchmark for all 24 drug-pair datasets used
in *Dimitsaki et al. (2026) — CML Framework for Pharmacovigilance*.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Per-drug confounder families (Literature & Clinician DAGs), drop lists, feature column groups |
| `psm.py` | Algorithm 3 (k:1 PSM with caliper), SMD evaluation, all alternative matching methods, Love Plot |
| `run_benchmark.py` | Full orchestrator — replaces the notebook; CLI entry point |

---

## Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost openpyxl matplotlib
```

---

## Input

The benchmark expects the 24 paired datasets produced by `data/preprocessing/03_merge_datasets.py`:

```
<work-dir>/
  ibuprofen_simethicone.parquet.gzip
  ibuprofen_prochlorperazine.parquet.gzip
  ibuprofen_lactulose.parquet.gzip
  ... (24 files total)
  atc_codes.csv
```

Each file must contain columns `y`, `t`, `subject_id`, `gender`, `anchor_age`,
`pre_AKI`, the Charlson comorbidity flags, the lab/vital columns, and the
ATC-encoded concomitant medication columns.

---

## Running the benchmark

### Full run (all 8 drugs × 3 controls)

```bash
python run_benchmark.py \
    --work-dir /data/final_datasets \
    --out      results/sml_mean.xlsx
```

### Single drug

```bash
python run_benchmark.py \
    --work-dir /data/final_datasets \
    --drug vancomycin \
    --out  results/vancomycin.xlsx
```

### Single pair

```bash
python run_benchmark.py \
    --work-dir /data/final_datasets \
    --drug    ibuprofen \
    --control lactulose \
    --out     results/ibu_lac.xlsx
```

---

## Output

An Excel workbook (`sml_mean.xlsx`) with one sheet per positive drug.
Each sheet has the following columns:

| Column | Description |
|---|---|
| `Methodology` | Matching strategy (e.g. `PSM Random Forest`) |
| `Initial SMD` | Mean absolute SMD before matching |
| `Mean SMD` | Mean absolute SMD after matching |
| `SMD > 0.2` | Number of covariates with SMD > 0.2 after matching |
| `Matched patients 1` | Number of unique treated patients in matched sample |
| `Matched patients 0` | Number of unique control patients in matched sample |
| `Total patients` | Total matched sample size |
| `Ratio` | Composite score: higher = better balance + more patients |
| `Process` | Preprocessing scenario (`Original` / `Binary` / `Normalised`) |
| `DAG` | Confounder selection source (`Literature` / `Clinician`) |
| `Drug` | Positive drug name |
| `Control` | Negative control drug name |

---

## Design decisions

### Why three separate files?

The original notebook mixed algorithm code, per-drug boilerplate (180 cells,
largely copy-pasted with drug names changed), and the final export step.
Splitting into `config.py`, `psm.py`, and `run_benchmark.py` means:

- Adding a new drug = one block in `config.py`
- Changing a matching algorithm = one edit in `psm.py`
- Changing the output format = one edit in `run_benchmark.py`

### Three preprocessing scenarios (Original / Binary / Normalised)

Directly mirror the paper's robustness evaluation (Table 10/11):

| Scenario | ATC columns | Extra columns (labs/demographics) |
|---|---|---|
| `Original` | Raw duration in days | Raw values |
| `Binary` | Z-score normalised, then used as-is | Raw values appended |
| `Normalised` | Z-score normalised | Not appended |

### The `Ratio` score

Used to rank matching configurations when selecting the best PSM model
for the X-Learner architecture:

```
Ratio = (n_treated + n_control) / (SMD_exceeding_0.2 + mean_SMD + 1)
```

Higher is better: it rewards larger matched cohorts and penalises
residual imbalance.

### Confounder selection per DAG

Each drug has a Literature-based and a Clinician-based DAG (see paper
Appendix F).  `config.py` maps each DAG to a list of ATC drug-family
names.  The pipeline then selects all ATC columns in the dataset whose
3-character prefix matches one of those families, ensuring the confounder
set adapts automatically to whichever ATC codes are present in a given
pair's dataset.
