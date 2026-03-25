# DAKI Causal ML

**A causal machine learning framework for pharmacovigilance signal detection in electronic health records: Drug-induced acute kidney injury**

*Dimitsaki et al. (2026) — Proceedings of the Conference on Causal Learning and Reasoning (CLeaR)*

---

## Overview

This repository contains the full, reproducible pipeline that accompanies the paper. It converts 20+ exploratory Jupyter notebooks into a clean, parameterised codebase covering every stage of the study:

| Stage | What it does |
|---|---|
| **ETL** | Build AKI and no-AKI cohorts from MIMIC-IV; map drugs to ATC codes; create 48 paired datasets |
| **Matching benchmark** | Evaluate 12+ matching strategies (PSM × 8 classifiers, NN, Mahalanobis, Exact) across all drug pairs and DAG sources |
| **CML model selection** | Benchmark all classifier/regressor combinations for S-, T-, X-, and DML-Learner architectures |
| **CATE estimation** | Estimate ATE + bootstrap CIs for all 48 drug-pair × lab-repr combinations across two DAGs and three preprocessing scenarios |
| **HTE estimation** | Compute per-patient Individual Treatment Effects (ITE) for each of the 8 positive drugs using the paper's best-performing CML architecture |
| **Statistical analysis** | Kruskal-Wallis tests, clinical binning (ACC/AHA guidelines), ITE summary tables, Plotly PDF boxplots |

---

## Study drugs

**8 positive drugs (suspected DAKI signals)**

| Drug | ATC Class |
|---|---|
| Ibuprofen | NSAIDs |
| Ketorolac | NSAIDs |
| Vancomycin | Antibacterials |
| Lisinopril | ACE Inhibitors |
| Furosemide | Diuretics |
| Pantoprazole | Proton Pump Inhibitors |
| Omeprazole | Proton Pump Inhibitors |
| Allopurinol | Gout / Uric Acid |

**3 negative control drugs (no known AKI association)**

Simethicone · Prochlorperazine · Lactulose

This produces **24 drug-pair datasets** (8 × 3), each available in two laboratory representations (**mean** and **t0t1** / first–last), giving **48 paired datasets** in total.

---

## Repository structure

```
daki-causal-ml/
│
├── requirements.txt
├── README.md  ← you are here
│
├── data/
│   └── preprocessing/
│       ├── config.py              Drug registry, ATC thresholds, feature column groups
│       ├── atc_mapping.py         Algorithm 1 (drug name unification) + Algorithm 2 (fuzzy ATC matching)
│       ├── 01_build_aki_cohort.py KDIGO filter → Charlson → labs → medications → merge (AKI)
│       ├── 02_build_noaki_cohort.py Same pipeline without the KDIGO step (no-AKI)
│       ├── 03_merge_datasets.py   Produces 48 paired parquet files
│       └── README.md
│
├── matching/
│   ├── config.py                  Per-drug confounder DAG configs (Literature & Clinician)
│   ├── psm.py                     Algorithm 3 (k:1 PSM with caliper), SMD evaluation, Love Plot
│   ├── run_benchmark.py           Full orchestrator → sml_mean.xlsx
│   └── README.md
│
├── cml/
│   ├── config.py                  Classifier/regressor registries, per-drug best-model configs,
│   │                              Equation 1 (composite regression score), DAG configs
│   ├── learners.py                s_learner(), t_learner(), x_learner(), dml_learner() wrappers
│   ├── model_selection.py         Benchmark all model combos for S/T/X/DML architectures
│   └── README.md
│
└── analysis/
    ├── cate_estimation.py         Full ATE pipeline (48 datasets × 2 DAGs × 3 scenarios)
    ├── hte_estimation.py          Per-patient ITE for all 8 drugs (best CML per drug)
    ├── statistical_tests.py       Kruskal-Wallis, clinical bins, PDF boxplots, Excel summaries
    └── README.md
```

**Total: ~5,900 lines of Python** replacing 20+ Jupyter notebooks with ~1,500+ cells.

---

## Installation

### Requirements

- Python 3.10 or 3.11
- Access to the MIMIC-IV clinical database (PhysioNet credentialed access)

### Setup

```bash
git clone https://github.com/your-org/daki-causal-ml.git
cd daki-causal-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on kaleido:** version `0.2.1` is pinned because later versions have
> known rendering issues on Linux. If PDF export fails, try
> `pip install kaleido==0.2.1 --force-reinstall`.

---

## Data

This study uses [MIMIC-IV](https://physionet.org/content/mimiciv/) (Medical Information Mart for Intensive Care).
Access requires completion of the CITI Data or Specimens Only Research course and signing of the PhysioNet Credentialed Health Data Use Agreement.

The pipeline also requires an **ATC code reference file** (`atc_codes.csv`) downloadable from the [WHO Collaborating Centre for Drug Statistics Methodology](https://www.whocc.no/atc_ddd_index/).

The expected file layout for `--work-dir`:

```
<work-dir>/
  atc_codes.csv
  ibuprofen_simethicone.parquet.gzip
  ibuprofen_simethicone_t0t1.parquet.gzip
  ibuprofen_prochlorperazine.parquet.gzip
  ...  (48 paired files total)
```

---

## Execution order

Run stages in sequence. Each stage reads the outputs of the previous one.

### Stage 1 — ETL

```bash
# Build AKI cohort for all 8 positive drugs
python data/preprocessing/01_build_aki_cohort.py \
    --drug all \
    --work-dir /data/mimic_iv

# Build no-AKI cohort for all drugs
python data/preprocessing/02_build_noaki_cohort.py \
    --drug all \
    --work-dir /data/mimic_iv

# Merge into 48 paired datasets
python data/preprocessing/03_merge_datasets.py \
    --work-dir /data/mimic_iv \
    --out-dir  /data/final_datasets
```

### Stage 2 — Matching benchmark

```bash
python matching/run_benchmark.py \
    --work-dir /data/final_datasets \
    --out      results/sml_mean.xlsx
```

This produces one Excel workbook with one sheet per positive drug comparing 12+ matching methods across Original / Binary / Normalised preprocessing scenarios.

To run a single drug pair:

```bash
python matching/run_benchmark.py \
    --work-dir /data/final_datasets \
    --drug     vancomycin \
    --control  lactulose \
    --out      results/vac_lac.xlsx
```

### Stage 3 — CML model selection

```python
# Example: select best model for a single drug-pair dataset
import sys; sys.path.insert(0, "cml/")
import pandas as pd, numpy as np
from model_selection import run_model_selection, save_model_selection_results

df = pd.read_parquet("data/final_datasets/ibuprofen_simethicone.parquet.gzip")
X = df.drop(columns=["y","t","subject_id"]).to_numpy()
y = df["y"].to_numpy()
t = df["t"].to_numpy()

results = run_model_selection(X, y, t)
save_model_selection_results(results, "results/ibuprofen_sim_model_selection.xlsx")
```

The best models discovered during this stage are already pre-encoded in
`cml/config.py → BEST_MODELS` for immediate use in Stages 4 and 5.

### Stage 4 — CATE estimation

```bash
python analysis/cate_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  results/cate

# Single drug, both lab representations
python analysis/cate_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  results/cate \
    --drug     ibuprofen

# Single pair, single lab representation
python analysis/cate_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  results/cate \
    --drug     ketorolac \
    --control  lactulose \
    --lab-repr mean
```

### Stage 5 — HTE estimation

```bash
python analysis/hte_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  results/hte

# Single drug
python analysis/hte_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  results/hte \
    --drug     lisinopril
```

### Stage 6 — Statistical analysis

```bash
python analysis/statistical_tests.py \
    --ite-dir results/hte \
    --out-dir results/stats
```

Outputs per drug:
- `<drug>_ite_vital.pdf` — blood pressure boxplots by ITE class
- `<drug>_ite_dem.pdf`   — age / weight boxplots by ITE class
- `<drug>_ite_labs.pdf`  — 8-lab panel boxplots by ITE class
- `<drug>_ite_summary.xlsx` — ITE summary by clinical category

---

## Key methodology

### Causal graph (DAG) sources

Each drug has two confounder sets derived from:
- **Literature** — confounders identified from published pharmacology and epidemiology literature
- **Clinician** — confounders selected by clinical expert review

Both sets are stored in `matching/config.py` and `cml/config.py` and used throughout the pipeline.

### Three preprocessing scenarios

Every analysis runs under three data representations:

| Scenario | ATC columns | Extra columns |
|---|---|---|
| **Original** | Raw exposure duration (days) | Raw lab/demographic values |
| **Binary** | Z-score normalised, clipped to [0,1] | Raw values appended |
| **Normalised** | Z-score normalised only | Not appended |

### CML architectures

| Learner | EconML class | Key parameter |
|---|---|---|
| S-Learner | `SLearner` | `overall_model` |
| T-Learner | `TLearner` | `models=[clf_treated, clf_control]` |
| X-Learner | `XLearner` | `models`, `propensity_model`, `cate_models` |
| DML | `DML` | `model_y`, `model_t`, `model_final` |

All CIs use `inference='bootstrap'` with n = 100 samples. A CI is flagged **Valid** if its width is < 0.85 (paper §2.5.3), and **Significant** if it excludes zero.

### Propensity score matching (Algorithm 3)

k:1 nearest-neighbour PSM with caliper = 0.2 × SD(logit PS), k = 4.
See `matching/psm.py → psm_caliper_k1()`.

### ITE classification thresholds

| ITE range | Class |
|---|---|
| ITE < −0.1 | Protective effect |
| −0.1 ≤ ITE ≤ 0.1 | No effect |
| ITE > 0.1 | Adverse effect |

### ATC fuzzy matching (Algorithm 2)

Drug names in MIMIC-IV prescriptions are matched to ATC codes using FuzzyWuzzy token-sort ratio with an 85% similarity threshold. A hand-curated override dictionary (~150 entries) handles common trade names. The mapping is cached to Excel after the first run for auditability and reproducibility.

---

## Results structure

```
results/
├── sml_mean.xlsx              Matching benchmark (one sheet per drug)
├── cate/
│   ├── ibuprofen_cate_results.xlsx
│   ├── ketorolac_cate_results.xlsx
│   └── ...
├── hte/
│   ├── ibuprofen_prochlorperazine_ite.parquet.gzip
│   ├── ketorolac_lactulose_ite.parquet.gzip
│   └── ...
└── stats/
    ├── ibuprofen/
    │   ├── ibuprofen_ite_vital.pdf
    │   ├── ibuprofen_ite_dem.pdf
    │   ├── ibuprofen_ite_labs.pdf
    │   └── ibuprofen_ite_summary.xlsx
    ├── ketorolac/
    └── ...
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{dimitsaki2026daki,
  title     = {A causal machine learning framework for pharmacovigilance signal
               detection in electronic health records:
               Drug-induced acute kidney injury},
  author    = {Dimitsaki, Stella and others},
  booktitle = {Proceedings of the Conference on Causal Learning and Reasoning (CLeaR)},
  year      = {2026}
}
```

---

## Licence

This project is released under the **MIT Licence**. See `LICENSE` for details.

Data derived from MIMIC-IV is subject to the [PhysioNet Credentialed Health Data Licence 1.5.0](https://physionet.org/content/mimiciv/view-license/2.2/).
