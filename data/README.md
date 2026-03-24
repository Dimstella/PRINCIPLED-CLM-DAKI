# `data/preprocessing/`

End-to-end preprocessing pipeline that converts raw MIMIC-IV query outputs
into the 24 paired AKI / no-AKI analysis datasets used in
*Dimitsaki et al. (2026) — CML Framework for Pharmacovigilance*.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Central registry of all 11 drugs, feature columns, and shared constants |
| `atc_mapping.py` | Drug-name unification (Algorithm 1) and ATC mapping (Algorithm 2) |
| `01_build_aki_cohort.py` | Builds `<drug>_aki_mean/t0t1.parquet.gzip` for all 11 drugs |
| `02_build_noaki_cohort.py` | Builds `<drug>_noaki_mean/t0t1.parquet.gzip` for all 11 drugs |
| `03_merge_datasets.py` | Merges pairs into 24 × 2 final datasets |

---

## Prerequisites

```bash
pip install pandas numpy fuzzywuzzy python-levenshtein openpyxl pyarrow
```

MIMIC-IV access is required. You cannot download the data from this repository.
Request access at https://physionet.org/content/mimiciv/.

---

## Input files expected in `--work-dir`

Each drug requires the following raw files extracted from MIMIC-IV BigQuery.
File names follow the pattern `<table>_<drug_slug>[_AKI|_noAKI].csv` or `.parquet.gzip`.

| Table | Suffix convention | Description |
|---|---|---|
| `kdigo_<slug>.csv` | AKI only | KDIGO AKI stages with timestamps |
| `charlson_<slug>[_AKI\|_noAKI].csv` | both | Charlson comorbidity index |
| `chemistry_t0_<slug>[_noAKI].csv` | both | Lab test results |
| `omar_<slug>[_AKI].csv` | both | Vital signs (weight, BMI, BP) |
| `medication_<slug>[_AKI\|_noAKI].csv.zip` | both | Prescription records |
| `patients_<slug>.csv` | both | Patient demographics |
| `atc_codes.csv` | reference | WHO ATC code list |

---

## Running the pipeline

### Step 1 — Build AKI cohorts (all 11 drugs)

```bash
python 01_build_aki_cohort.py \
    --drug all \
    --work-dir /data/mimic_raw \
    --atc-csv  /data/mimic_raw/atc_codes.csv
```

For a single drug:

```bash
python 01_build_aki_cohort.py --drug vancomycin --work-dir /data/mimic_raw
```

On the **first run** the fuzzy ATC mapping saves an Excel file  
`atc_med_map_85_<drug>_AKI.xlsx` in `--work-dir` for manual validation.  
On subsequent runs the validated file is used directly (no re-matching).

### Step 2 — Build no-AKI cohorts

```bash
python 02_build_noaki_cohort.py --drug all --work-dir /data/mimic_raw
```

### Step 3 — Merge into 24 paired datasets

```bash
python 03_merge_datasets.py \
    --work-dir /data/mimic_raw \
    --out-dir  /data/final_datasets
```

This produces 48 files (24 pairs × 2 lab representations):

```
vancomycin_vs_simethicone_mean.parquet.gzip
vancomycin_vs_simethicone_t0t1.parquet.gzip
vancomycin_vs_prochlorperazine_mean.parquet.gzip
...
allopurinol_vs_lactulose_t0t1.parquet.gzip
```

---

## Design decisions

### Why three scripts instead of one notebook?

The original analysis used two Jupyter notebooks (`ETL_vancomycin_AKI.ipynb`
and `ETL_vancomycin_noAKI.ipynb`) that mixed exploratory and production code,
contained hardcoded drug names (`lisinopril_flag`, `stoptime_lisinopril` etc.),
and relied on manually updated cells between runs.

The refactored pipeline:
- Parameterises every drug-specific string through `config.py`
- Separates concerns (one script per logical stage)
- Is idempotent: re-running any script overwrites only its own outputs
- Saves ATC-mapping Excel files for manual review without blocking the run

### The `--mapping-cache` flag

Fuzzy matching is non-deterministic at borderline similarity scores.
Saving the mapping to Excel after the first run and loading it on
subsequent runs guarantees exact reproducibility of the drug–ATC assignments
reported in the paper.

### Column alignment in step 3

Concomitant-drug ATC columns vary per drug because each cohort is built
from a different patient population.  Step 3 takes the **intersection** of
ATC columns between the AKI and no-AKI tables so the final dataset has a
consistent feature set for causal modelling.
