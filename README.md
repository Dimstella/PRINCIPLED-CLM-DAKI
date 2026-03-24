# Causal machine learning framework for pharmacovigilance signal detection in electronic health records: Drug-induced acute kidney injury

In this work, we implemented a CML framework using the MIMIC-IV database to detect acute drug-induced kidney injury, demonstrating that CML can provide interpretable, personalized, and clinically relevant information. The results position causal AI as a promising avenue for improving the accuracy, transparency, and regulatory acceptance of pharmacovigilance systems.

![figure](https://github.com/Dimstella/PRINCIPLED-CLM-DAKI/blob/main/principled_methodology.png)


## Project Overview
This repository implements the methodology described in the paper: *"A causal machine learning framework for pharmacovigilance signal detection in electronic health records: Drug-induced acute kidney injury."*

The project addresses the limitations of traditional correlation-based PV by using **Target Trial Emulation** and **Causal AI** to estimate the actual treatment effect of suspect drugs on renal function using data from the MIMIC-IV database.

## Key Features
* **PRINCIPLED Workflow**: A structured 5-step process (Protocol, Emulation, Precision, Robustness, Inference) for causal analysis in healthcare.
* **Advanced CML Estimators**: Implementation of various meta-learners (**S-learner, T-learner, X-learner**) and **Double Machine Learning (DML)** via the `EconML` library.
* **Clinically-Validated DAGs**: Causal discovery supported by Directed Acyclic Graphs (DAGs) verified by literature and nephrologists.
* **Subgroup Analysis**: Estimation of **Heterogeneous Treatment Effects (HTE)** to identify which patient phenotypes are at higher risk.

## Methodology

### 1. Protocol & Emulation
- **Database**: MIMIC-IV (Medical Information Mart for Intensive Care).
- **Interventions**: Includes drugs like Ibuprofen, Vancomycin, and Lisinopril.
- **Outcome**: Acute Kidney Injury (AKI) defined by KDIGO criteria (serum creatinine changes).
- **Matching**: Propensity Score Matching (PSM) to ensure balanced cohorts.

### 2. Causal Estimation
The framework utilizes several CML techniques to calculate the **Conditional Average Treatment Effect (CATE)**:
- **S-Learner**: Treatment as a feature in a single model.
- **T-Learner**: Separate models for treated and control groups.
- **X-Learner**: Specifically designed for unbalanced treatment assignments.
- **DML**: Combining machine learning with linear modeling for orthogonalization.

## Repository Structure
* `/data/` - Scripts for cohort extraction and preprocessing.
* `/cml/` - Causal Machine Learning pipeline for DAKI pharmacovigilance.
* `/matching/` - Propensity-score matching benchmark for all 24 drug-pair datasets.
* `/analysis/` - Post-CML analysis pipeline for DAKI pharmacovigilance.

## Setup & Requirements
To run this project, you will need:
- Python 3.8+
- `econml`
- `dowhy`
- `scikit-learn`
- `pandas`
- `numpy`

## Citation
If you use this code or methodology, please cite:
> Dimitsaki, S., Bagnis, C. I., Natsiavas, P., & Jaulent, M.-C. (2026). A causal machine learning framework for pharmacovigilance signal detection in electronic health records: Drug-induced acute kidney injury. *Proceedings of Machine Learning Research*, 323, 1-81.
