# Behavioural_decline_MLMs — Notebook Overview

This notebook implements preprocessing, cohort construction, and mixed-effects linear modelling (MLM) for investigating behavioural and cognitive decline in ADNI participants. It combines diagnosis timelines, comorbidity flags (GI, UTI, sleep), CSF biomarkers, and cognitive/functional outcomes to fit linear and quadratic MLMs with random intercepts and slopes.

**Overview**
- **Purpose:** Quantify longitudinal cognitive and functional decline and evaluate how comorbidities and CSF biomarkers (p‑tau181, Aβ42, and their ratio) predict trajectories.
- **Scope:** Data preprocessing, cohort derivation, diagnosis transition analysis, subgroup comparisons, MLM fitting, plotting, and saving results.

**Data Inputs**
- Required CSVs (store in `data/` and change the date for the date that you download the data on): `DXSUM_{date}.csv`, `PTDEMOG_{date}.csv`, `ADAS_{date}.csv`,`MEDHIST_{date}.csv`, `RECMHIST_{date}.csv`, `APOERES_{date}.csv`, plus outcome files used later (e.g., `CDR_{date}.csv`, `MMSE_{date}.csv`, `FAQ_{date}.csv`).

**Main Processing Steps**
- Merge preprocessed ADAS/clinical data with demographic and comorbidity flags.
- Create cohort subsets based on diagnostic transitions (First → Last: CN, MCI, AD).
- Build per-participant baseline covariates and standardize biomarker measures.
- Construct RID-level flags for comorbidities using both strict and inclusive timing rules.

**Models & Key Functions**
- **`quadratic_MLM_with_plot()`**: Fits quadratic MLMs (time and time^2), prints significant fixed effects with 95% CIs, computes marginal/conditional R², saves CSVs and trajectory plots to `quadratic_model_results/`.
- **`linear_MLM_with_plot()`**: Fits linear MLMs and saves outputs to `linear_model_results/`.
- **`compare_lin_quad()`**: Convenience wrapper to compute BIC for linear vs quadratic fits.
- **FAQ handling:** Uses a logit transform for FAQ (0–30 bounded outcomes), fits MLM on logit scale, and back-transforms effect sizes to FAQ units.

**Subgroup & Outcome Analyses**
- Cohorts: All patients, Progressors (end MCI/AD), AD converters, AD at entry, MCI converters.
- Outcomes modelled: `TOTAL13` (ADAS-Cog13), `CDRSB`, `MMSCORE`, `FAQTOTAL`.
- Comorbidity comparisons: Sleep, GI, UTI groups vs non-affected.

**Outputs**
- Plots: Saved PNGs in `linear_model_results/plots/` and `quadratic_model_results/plots/`.
- Tables: CSVs of significant model predictors in `.../tables/`.
- Intermediate datasets: `data/CSF_AD.csv`, `data/demo_and_comorb.csv` (created by preprocessing).

**Dependencies**
- Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels` (and standard library), see requirements.txt file.

**How to run**
1. Ensure the `data/` folder contains the required CSV files listed above (fetch from ADNI, https://adni.loni.usc.edu/data-samples/adni-data/).
2. Open and run the notebook `Behavioural_decline_MLMs.ipynb` end-to-end in a Python 3.8+ environment with the dependencies installed (e.g., a conda environment).
3. Results will be saved under `linear_model_results/` and `quadratic_model_results/` and intermediate CSVs in `data/`.

**Notes & Recommendations**
- Small missing-data checks and `.dropna()` steps are used in the notebook; review those before reproducing results for different cohorts.
- If you plan to run analyses headlessly, convert critical cells to a script and call from the command line after setting up the environment.

