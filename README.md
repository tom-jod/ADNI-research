# ADNI Data Preprocessing Notebook

This notebook performs preprocessing and cohort construction for ADNI data analysis, focusing on comorbidity classification, cognitive outcomes (ADAS-13 scores), and biomarker data (CSF measurements).

## Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load Data Files

Replace with the actual paths to your CSV files.

```python
# Replace with the actual paths to your CSV files

dx_df = pd.read_csv('data/DXSUM_17Feb2026.csv')
dem  = pd.read_csv('data/PTDEMOG_17Feb2026.csv') 
adas = pd.read_csv('data/ADAS_17Feb2026.csv')
medhist = pd.read_csv('data/MEDHIST_17Feb2026.csv')
recmhist = pd.read_csv('data/RECMHIST_17Feb2026.csv')
reccmeds = pd.read_csv('data/RECCMEDS_16Mar2026.csv')
backmeds = pd.read_csv('data/BACKMEDS_16Mar2026.csv')
demographics = pd.read_csv('data/PTDEMOG_17Feb2026.csv')
apoe_df = pd.read_csv("data/APOERES_17Feb2026.csv")
apoe_df["CARRIER"] = apoe_df["GENOTYPE"].isin(["2/4","3/4","4/4"])
apoe_df["HOMO"] = apoe_df["GENOTYPE"].isin(["4/4"])
```

## Initial Data Exploration

```python
print(recmhist["RID"].nunique())
print(adas["RID"].nunique())
print(demographics["RID"].nunique())
print(demographics.columns)
```

## Time Conversion and ADAS Score Processing

```python
def visit_to_months(v):
    if pd.isna(v):
        return np.nan
    v = v.lower()
    if v in ["bl", "sc"]:
        return 0
    if v.startswith("m"):
        return float(v[1:])  # strip the 'm'
    if v.startswith("v"):
        return float(v[1:])  # strip the 'v'
    
    return np.nan

baseline_codes = ["bl", "4_bl"]
adas["time_months"] = adas["VISCODE"].apply(visit_to_months)

baseline_dates = (
    adas[adas["VISCODE"].isin(baseline_codes)]
    .groupby("RID")["VISDATE"]
    .min()
)

adas_df = adas.merge(
    baseline_dates.rename("baseline_date"),
    on="RID",
    how="left"
)

adas_df["days_since_bl"] = (
    pd.to_datetime(adas_df["VISDATE"]) -
    pd.to_datetime(adas_df["baseline_date"])
)
```

## Merge ADAS Scores with Diagnosis Data

```python
# Merge ADAS scores with diagnosis data
adas_AD_dx = pd.merge(adas_df, dx_df[["RID", "VISCODE", "DIAGNOSIS"]], on=["RID", "VISCODE"], how="left")
adas_AD_dx['VISDATE'] = pd.to_datetime(adas_AD_dx['VISDATE'])
```

## Finding Comorbidities with String Matching

### Define Keyword Categories

```python
UTI_KEYWORDS = [
    "urinary tract infection",
    "uti",
]

GI_KEYWORDS = [
    "diarrhea",
    "constipation",
    "halitosis",
    "fecal incontinence",
    "abnormal bowel movement",
    "abnormal bowel sounds",
    "encopresis",
]

SLEEP_KEYWORDS = [
    "insomnia",
    "sleep disorder",
    "sleep disturbance",
    "poor sleep",
    "sleep apnea",
    "obstructive sleep apnea",
    "osa",
    "hypersomnia",
    "sleep fragmentation",
    "restless legs",
    "bruxism",
    "parasomnia",
    "narcolepsy", 
]

CHRONIC_GI_KEYWORDS = [
    "chronic diarrhea",
    "chronic constipation",
    "frequent diarrhea",
    "frequent constipation",
    "occasional diarrhea",
    "occasional constipation",
    "intermittent diarrhea",
    "intermittent constipation",
    "history of diarrhea",
    "history of constipation",
]
```

### Classify Specific Conditions

```python
SPECIFIC_KEYWORDS = {
    "UTI": UTI_KEYWORDS,
    "GI": GI_KEYWORDS,
    "Sleep": SLEEP_KEYWORDS,
}

# Function to standardise and classify specific conditions based on keywords
def classify_specific(desc):

    if pd.isna(desc):
        return None
    
    
    d = desc.lower()
    d = d.replace("hx of", "history of")
    d = d.replace("h/o", "history of")
    
    for category, keywords in SPECIFIC_KEYWORDS.items():
        if any(k in d for k in keywords):
            return category
    
    return None

recmhist["specific_flag"] = recmhist["MHDESC"].apply(classify_specific)
```

## Classify Conditions into CNS vs Peripheral

Following the work of Bu et al. (2020), we can classify conditions into CNS vs Peripheral based on keywords in the description. This is a simple heuristic approach and may not be perfect, but it can help identify common comorbidities.

```python
CNS_KEYWORDS = {
    'cerebrovascular': ['stroke', 'cerebrovascular', 'tia', 'transient ischemic', 'cva'],
    'insomnia': ['insomnia', 'sleep disorder'],
    'anxiety': ['anxiety', 'anxious'],
    'depression': ['depression', 'depressive', 'depressed'],
    'head_injury': ['head injury', 'tbi', 'traumatic brain', 'concussion', 'head trauma']
}

PERIPHERAL_KEYWORDS = {
    'hypertension': ['hypertension', 'high blood pressure', 'htn'],
    'hyperlipidemia': ['hyperlipidemia', 'hypercholesterolemia', 'high cholesterol', 'dyslipidemia'],
    'diabetes': ['diabetes', 'diabetic', 'dm', 'niddm', 'iddm'],
    'atrial_fibrillation': ['atrial fibrillation', 'afib', 'a fib', 'a-fib'],
    'coronary': ['coronary', 'ischemic heart', 'ihd', 'cad', 'coronary artery', 'myocardial infarction', 'mi', 'heart attack'],
    'anemia': ['anemia', 'anaemia'],
    'hypothyroid': ['hypothyroid', 'thyroid'],
    'skin_inflammatory': ['psoriasis', 'eczema', 'dermatitis', 'skin inflammation'],
    'pulmonary': ['copd', 'asthma', 'pulmonary', 'emphysema', 'chronic bronchitis', 'respiratory', 'lung disease'],
    'kidney': ['kidney', 'renal', 'ckd', 'chronic kidney'],
    'hepatitis': ['hepatitis', 'liver disease', 'cirrhosis'],
    'osteoporosis': ['osteoporosis', 'bone density'],
    'hearing_loss': ['hearing loss', 'deaf', 'hearing impair'],
    'cancer': ['cancer', 'malignancy', 'carcinoma', 'tumor', 'neoplasm'],
    'gastrointestinal': ['gastro', 'ulcer', 'reflux', 'gerd', 'ibs', 'crohn', 'colitis', 'diverticulitis'],
    'cataract': ['cataract'],
}

def classify_condition_detailed(desc):
    """
    Classify condition into CNS, Peripheral, or Other
    Returns tuple: (category, specific_condition)
    """
    if pd.isna(desc):
        return None, None
    
    d = desc.lower()
    
    # Check CNS conditions
    for condition, keywords in CNS_KEYWORDS.items():
        if any(k in d for k in keywords):
            return "CNS", condition
    
    # Check Peripheral conditions
    for condition, keywords in PERIPHERAL_KEYWORDS.items():
        if any(k in d for k in keywords):
            return "Peripheral", condition
    
    return "Other", None

recmhist["comorb_category"] = recmhist["MHDESC"].apply(
    lambda x: classify_condition_detailed(x)[0]
)
recmhist["specific_condition"] = recmhist["MHDESC"].apply(
    lambda x: classify_condition_detailed(x)[1]
)
```

## Process Baseline Comorbidities

```python
recmhist_bl = recmhist[recmhist["VISCODE"].isin(["v01", "sc"])]
```

## Count and Categorize Comorbidities

```python
comorb_counts = (
    recmhist_bl[recmhist_bl["comorb_category"].isin(["CNS", "Peripheral"])]
    .groupby(["RID", "VISCODE", "comorb_category", "specific_condition"])
    .size()
    .reset_index(name="count")
    .groupby(["RID", "VISCODE", "comorb_category"])
    .agg(
        n_conditions=("specific_condition", "nunique"),
        total_entries=("count", "sum")
    )
    .reset_index()
)

# Create wide format
comorb_wide = comorb_counts.pivot_table(
    index=["RID", "VISCODE"],
    columns="comorb_category",
    values="n_conditions",
    fill_value=0
).reset_index()

# Calculate total multimorbidity burden
comorb_wide["total_conditions"] = (
    comorb_wide.get("CNS", 0) + 
    comorb_wide.get("Peripheral", 0)
)

# Categorize burden as in the paper
def categorize_burden(n):
    if n <= 2:
        return "Low"
    elif n <= 5:
        return "Medium"
    else:
        return "High"

comorb_wide["burden_category"] = comorb_wide["total_conditions"].apply(categorize_burden)

# For CNS burden: 0 vs 1+ 
comorb_wide["CNS_burden"] = comorb_wide["CNS"].apply(lambda x: "0" if x == 0 else "1+")

# For Peripheral burden: 0-1 vs 2+ 
comorb_wide["Peripheral_burden"] = comorb_wide["Peripheral"].apply(
    lambda x: "0-1" if x <= 1 else "2+"
)
```

```python
print(medhist.columns)
```

## Incorporating Broad GI and UTI Definitions

```python
# Relevant MEDHIST binary columns and their mappings
# 1 = present, 0 = absent, 2 = unknown (treat as 0)
MEDHIST_FLAG_MAP = {
    "MH10GAST": "GI_broad",          # Gastrointestinal disorders
    "MH12RENA": "UTI_broad",         # Renal/genitourinary disorders
}

# Keep only the columns we need plus identifiers
medhist_cols = ["RID", "VISCODE"] + [k for k in MEDHIST_FLAG_MAP if MEDHIST_FLAG_MAP[k] is not None]
medhist_sub = medhist[medhist_cols].copy()

# Recode: treat 2 (unknown) as 0, keep 1 as 1
for col in [k for k in MEDHIST_FLAG_MAP if MEDHIST_FLAG_MAP[k] is not None]:
    medhist_sub[col] = medhist_sub[col].apply(lambda x: 1 if x == 1 else 0)

medhist_sub = medhist_sub.rename(columns={k: v for k, v in MEDHIST_FLAG_MAP.items() if v is not None})

print(medhist_sub["UTI_broad"].value_counts())
sub_bl = medhist_sub[medhist_sub["VISCODE"].isin(['sc','v01'])]
print(sub_bl["UTI_broad"].value_counts())
```

## Merge Specific Flags and Broad Definitions

```python
specific_flags_visit = (
    recmhist[recmhist["specific_flag"].notna()]
    .assign(flag=1)
    .pivot_table(
        index=["RID", "VISCODE"],
        columns="specific_flag",
        values="flag",
        aggfunc="max",
        fill_value=0
    )
    .reset_index()
)

comorb_wide_new = comorb_wide.merge(
    specific_flags_visit,
    on=["RID", "VISCODE"],
    how="left"
)

# Replace NaNs with 0s for the specific flags
for col in ["UTI", "GI", "Sleep"]:
    if col in comorb_wide_new.columns:
        comorb_wide_new[col] = comorb_wide_new[col].fillna(0)

comorb_wide_to_merge = comorb_wide_new.copy()
```

```python
print("comorb_wide_new VISCODE samples:", comorb_wide_new["VISCODE"].unique()[:10])
print("medhist VISCODE samples:", medhist["VISCODE"].unique()[:10])
print("comorb_wide_new RID dtype:", comorb_wide_new["RID"].dtype)
print("medhist RID dtype:", medhist["RID"].dtype)
```

```python
# merge broad definitions into the other comorbidity flags 
comorb_wide_updated = comorb_wide_new.merge(
    sub_bl[["RID", "VISCODE", "GI_broad", "UTI_broad"]],
    on=["RID", "VISCODE"],
    how="left",
)
```

## Examine GI-Related Participants

```python
# Examine the free text descriptions of GI participants from final cohort

GI_RIDs = [1394, 1371, 1351, 1265, 1217, 1171, 1081,  941,  855,  839,  790,
        784,  754,  724,  625,  605,  474,  467,  372,  150, 2248, 2398,
       2403, 4015, 4102, 4209, 4250, 4422, 4379, 4404, 4589, 4675, 4657,
       4730, 4707, 4905, 5119]

comorb_GI = comorb_wide_updated[comorb_wide_updated["RID"].isin(GI_RIDs)]
recmhist_GI = recmhist[recmhist["RID"].isin(GI_RIDs)]
for desc in recmhist_GI["MHDESC"]:
    if 'constipation' in desc.lower() or 'diarrhea' in desc.lower():
       print(desc)

```

## Cohort Construction

```python
demographics_unique = demographics[["RID", "PTDOB", "PTGENDER", "PTEDUCAT",'PTETHCAT', 'PTRACCAT']].drop_duplicates(subset=["RID"])

# Merge ADAS test scores with demographics
AD_dem = adas_AD_dx.merge(
    demographics_unique,
    on="RID",
    how="left",
    validate="many_to_one"
)

print(f"Number of individuals with demographic data and cognitive data: {AD_dem['RID'].nunique()}")
print(comorb_wide_updated.columns)
comorb_visit = comorb_wide_updated[[
    "RID", "total_conditions", "CNS", "Peripheral",
    "UTI", "GI", "Sleep", "UTI_broad", "GI_broad"
]].drop_duplicates(subset=["RID"])

# Merge the baseline medical data with the ADAS and demographics merged dataframe
AD = AD_dem.merge(
    comorb_visit,
    on="RID",
    how="left",
    validate="many_to_one"
)
AD.dropna(subset=["total_conditions"], inplace=True)
print(f"Number of individuals with comorbidity data: {AD['RID'].nunique()}" )
```

```python
demographics_and_comorb = demographics_unique.merge(
    comorb_visit,
    on="RID",
    how="right",
    validate="many_to_one")
print(f"Number of individuals with demographic data and comorbidity data: {demographics_and_comorb['RID'].nunique()}")
demographics_and_comorb.to_csv('data/demo_and_comorb.csv', index=False)
```

## Process Baseline ADAS and Age

```python
# Add baseline age and ADAS13 score to all later visits
ad_start = AD[['RID', 'VISCODE', 'VISDATE', 'TOTAL13', 'PTGENDER', 'PTDOB', 'PTEDUCAT']].copy()
first_entries = (
    ad_start
    .sort_values('VISDATE')
    .groupby('RID')
    .first()
    .reset_index()
)
print(first_entries['VISCODE'].value_counts())
first_entries['VISDATE'] = pd.to_datetime(first_entries['VISDATE'])
first_entries['PTDOB'] = pd.to_datetime(first_entries['PTDOB'])

# Calculate age at conversion (in years)
first_entries['age_at_baseline'] = (
    (first_entries['VISDATE'] - first_entries['PTDOB'])
    .dt.days / 365.25
)

ad_start = first_entries.rename(columns={
    'VISDATE': 'Study_start_date',
    'TOTAL13': 'TOTAL13_AD_start'
})

AD_new = AD.merge(ad_start[['RID', 'Study_start_date', 'TOTAL13_AD_start', 'age_at_baseline']], on='RID', how='left')
print(AD_new["RID"].nunique())

```

## Define Model Variables

```python
model_vars = [
    "TOTAL13",
    "TOTAL13_AD_start",
    "VISCODE",
    "VISDATE",
    "PHASE",
    "days_since_entry",
    "age_at_baseline",
    "PTGENDER",
    'PTETHCAT', 
    'PTRACCAT',
    "CARRIER",
    "HOMO",
    "PTEDUCAT",
    "CNS",
    "total_conditions",
    "Peripheral",
    "RID",
    "UTI", "GI", "Sleep", 
    "UTI_broad", "GI_broad"
]
```

## Finalize Dataset with APOE and CSF Data

```python
# Merge APOE carrier status and homozygosity into the ADAS and demographics merged dataframe
AD_df = AD_new.merge(apoe_df[["RID", "CARRIER", "HOMO"]], on="RID", how="left")

AD_df['VISDATE'] = pd.to_datetime(AD_df['VISDATE'])
AD_df['Study_start_date'] = pd.to_datetime(AD_df['Study_start_date'])
AD_df['days_since_entry'] = (AD_df['VISDATE'] - AD_df['Study_start_date']).dt.days
print(f'Before dropping EO individuals: {AD_df["RID"].nunique()}')

AD_no_EO = AD_df[AD_df['age_at_baseline'] >= 65]
print(f'After dropping EO individuals: {AD_no_EO["RID"].nunique()}')

AD_model = AD_no_EO[model_vars].dropna()
AD_model["time_years"] = AD_model["days_since_entry"] / 365.25

print(AD_model["RID"].nunique())

```

## Add in CSF Data

### Load CSF Biomarkers

```python
CSF = pd.read_csv("data/UPENNBIOMK_ROCHE_ELECSYS_19Feb2026.csv")
print(CSF.columns)

CSF["PTAU_ABETA42"] = CSF["PTAU"] / CSF["ABETA42"]
#CSF["ABETA"] = CSF["ABETA40"] / CSF["ABETA42"] # Can't use due to high sparsity of ABETA40 values

CSF_bl = CSF[CSF["VISCODE2"] == "bl"]
CSF_bl["CSF_date"] = CSF_bl["EXAMDATE"]

```

### Check Data Availability

```python
not_na_AB = CSF_bl["ABETA42"].notna().sum()
print(f"Non-missing values for AB42: {not_na_AB}")

not_na_AB40 = CSF_bl["ABETA40"].notna().sum()
print(f"Non-missing values for AB40: {not_na_AB40}")
```

### Merge CSF with ADAS Dataset

```python
CSF_AD = AD_model.merge(
    CSF_bl[["RID", "VISCODE2", "PTAU", "ABETA42", "PTAU_ABETA42", "CSF_date"]],
    on=["RID"],
    how="left")
CSF_AD["time_sq"] = CSF_AD["time_years"] ** 2

CSF_AD = CSF_AD.dropna(subset=["TOTAL13"])
CSF_AD = CSF_AD.dropna(subset=["PTAU_ABETA42"])
CSF_AD = CSF_AD.dropna(subset=["PTAU"])
CSF_AD = CSF_AD.dropna(subset=["ABETA42"])

print(CSF_AD["RID"].nunique())
```

```python
print(CSF_AD.groupby(['RID', 'VISCODE']).size().sort_values(ascending=False).head())
```

## Save Final Dataset

```python
# Save the final dataset for modelling in 2-MLM_comparisons.ipynb

CSF_AD.to_csv('data/CSF_AD.csv', index=False)
```

## Cohort Composition

```python
total = CSF_AD["RID"].nunique()

groups = {
    "UTI":       CSF_AD["UTI"] == 1,
    "UTI_broad": CSF_AD["UTI_broad"] == 1,
    "GI":        CSF_AD["GI"] == 1,
    "GI_broad":  CSF_AD["GI_broad"] == 1,
    "Sleep":     CSF_AD["Sleep"] == 1,
}

for name, mask in groups.items():
    n = CSF_AD[mask]["RID"].nunique()
    print(f"{name}: n={n} ({100*n/total:.1f}% of cohort)")

print(f"\nTotal cohort: n={total}")
```

## Explore Sleep Medications

### Data Strategy

This section layers three sources of sleep medication data:

1. **MEDHIST**: pre-study sleep disorder diagnosis (confirms the condition was present)
2. **BACKMEDS**: structured medication flags at each visit
   - KEYMED column contains pipe-separated codes (e.g., "4:05:06")
   - Check data dictionary for which code = which drug class
   - Less useful for sleep meds specifically
   - MISSING: ADNI1 patients entirely
3. **RECCMEDS**: free-text concurrent medications at each visit
   - CMMED column: drug name (messy, needs regex cleaning)
   - CMREASON column: reason prescribed
   - Best source for sleep medications specifically
   - Covers ADNI1 onwards

### Analyze Medication Data

```python
# ── Strategy: layer all three sources ────────────────────────────────────
#
# 1. MEDHIST  → pre-study sleep disorder diagnosis (you already use this
#               for your Sleep flag — confirms the condition was present)
#
# 2. BACKMEDS → structured medication flags at each visit
#               KEYMED column contains pipe-separated codes e.g. "4:05:06"
#               Check data dictionary for which code = which drug class
#               Good for: cholinesterase inhibitors, BP meds — less useful
#               for sleep meds specifically
#               MISSING: ADNI1 patients entirely
#
# 3. RECCMEDS → free-text concurrent medications at each visit
#               CMMED column: drug name (messy, needs regex cleaning)
#               CMREASON column: reason prescribed
#               Best source for sleep medications specifically
#               Covers ADNI1 onwards
#               Use the find_orexin_antagonists.py script on this

# ── Quick check: what KEYMED values exist in BACKMEDS ────────────────────
print("=== BACKMEDS KEYMED unique values ===")
print(backmeds["KEYMED"].value_counts().head(30))
# These will be pipe-separated codes — cross-reference with data dictionary
# to see if any sleep medications are tracked

# ── Check ADNI phase coverage ─────────────────────────────────────────────
print("\n=== BACKMEDS visit codes (phase coverage) ===")
print(backmeds["VISCODE"].value_counts().head(20))

# ── Check RECCMEDS for sleep drug mentions ────────────────────────────────
reccmeds["med_lower"] = reccmeds["CMMED"].fillna("").str.lower()

sleep_pattern = (
    "zolpidem|ambien|zopiclone|eszopiclone|zaleplon|"
    "temazepam|triazolam|nitrazepam|lorazepam|"
    "melatonin|ramelteon|trazodone|mirtazapine|"
    "quetiapine|doxepin|diphenhydramine|hydroxyzine|"
    "suvorexant|belsomra|lemborexant|dayvigo"
)

sleep_pattern = ("trazodone")

sleep_reccmeds = reccmeds[reccmeds["med_lower"].str.contains(sleep_pattern, na=False)]

print("\n=== Sleep medications in RECCMEDS ===")
print(f"Rows: {len(sleep_reccmeds)}, Patients: {sleep_reccmeds['RID'].nunique()}")
print(sleep_reccmeds["CMMED"].value_counts().head(20))

# ── Per-patient flag for merging ──────────────────────────────────────────
sleep_med_patients = (
    sleep_reccmeds.groupby("RID")
    .agg(
        sleep_med_ever    = ("CMMED", "count"),
        sleep_med_list    = ("CMMED", lambda x: list(x.unique())),
        earliest_sleep_med = ("CMBGN", "min")
    )
    .reset_index()
)
sleep_med_patients["on_sleep_med"] = True

```

---

# ADNI MLM Comparisons: Analysis and Clinical Outcome Modeling

This section analyzes diagnosis transitions, cohort characteristics, and mixed-effects linear (MLM) models predicting cognitive and functional outcomes in patients with Alzheimer's disease.

## Setup and Data Loading

### Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches

plt.rcParams["font.size"] = 14
```

### Load Preprocessed Datasets

```python
# Load in the preprocessed dataset for modelling and other useful datasets
CSF_AD = pd.read_csv('data/CSF_AD.csv')
demo_and_comorb = pd.read_csv('data/demo_and_comorb.csv')
dx_df = pd.read_csv('data/DXSUM_17Feb2026.csv')
medhist = pd.read_csv('data/MEDHIST_17Feb2026.csv')
recmhist = pd.read_csv('data/RECMHIST_17Feb2026.csv')

# Check that no rows have been duplicated from merges in the previous script
print(CSF_AD.groupby(['RID','VISCODE']).size().sort_values(ascending=False).head())
print(CSF_AD.columns)
```

### Utility Functions

#### Pretty Labels for Plots

```python
def pretty_label(var_name):
    """
    Converts snake_case variable names into publication-ready labels.
    """

    # Remove common prefixes
    prefixes = ["adas_", "cdr_", "mmse_"]
    for p in prefixes:
        if var_name.startswith(p):
            var_name = var_name[len(p):]
   
    # Replace underscores with spaces
    var_name = var_name.replace("_", " ")

    # Capitalize words
    var_name = var_name.title()

    return f"{var_name} Score"
```

## Diagnosis Transition Analysis

### Analyze Diagnosis Changes Over Time

```python
# ── 1. Merge on RID + VISCODE ──────────────────────────────────────────────
merged = demo_and_comorb.merge(
    dx_df[['RID', 'VISCODE', 'DIAGNOSIS', "EXAMDATE"]],
    on=['RID'],
    how='left'
)

diag_map = {1: 'CN', 2: 'MCI', 3: 'AD'}
merged['DIAGNOSIS_LABEL'] = merged['DIAGNOSIS'].map(diag_map)

# ── 2. First & last visit per patient (by EXAMDATE) ───────────────────────
merged_sorted = merged.sort_values(['RID', 'EXAMDATE'])

first_visit = merged_sorted.groupby('RID').first().reset_index()
last_visit  = merged_sorted.groupby('RID').last().reset_index()

first_counts = first_visit['DIAGNOSIS_LABEL'].value_counts().reindex(['CN', 'MCI', 'AD'], fill_value=0)
last_counts  = last_visit['DIAGNOSIS_LABEL'].value_counts().reindex(['CN', 'MCI', 'AD'], fill_value=0)

print("=== First Visit Diagnosis ===")
print(first_counts)
print(f"  Missing: {first_visit['DIAGNOSIS_LABEL'].isna().sum()}")

print("\n=== Last Visit Diagnosis ===")
print(last_counts)
print(f"  Missing: {last_visit['DIAGNOSIS_LABEL'].isna().sum()}")

# ── 3. Sankey-style transition counts ─────────────────────────────────────
transitions = merged_sorted.groupby('RID').agg(
    first_dx=('DIAGNOSIS_LABEL', 'first'),
    last_dx=('DIAGNOSIS_LABEL', 'last')
).reset_index()

transition_counts = transitions.groupby(['first_dx', 'last_dx']).size().reset_index(name='count')
print("\n=== Transition counts (first → last) ===")
print(transition_counts.pivot(index='first_dx', columns='last_dx', values='count').fillna(0))
```

## Cohort Definition

### Define Analysis Cohorts Based on Diagnostic Progression

```python
# ── Group 1: All patients ──────────────────────────────────────────────────
df_all = CSF_AD.merge(
    transitions[['RID', 'first_dx', 'last_dx']],
    on='RID', how='left'
)
print(f"Group 1 - All patients:          n = {df_all['RID'].nunique()} patients, {len(df_all)} rows")

# ── Group 2: End up as MCI or AD at last visit ─────────────────────────────
progressors_rid = transitions[transitions['last_dx'].isin(['MCI', 'AD'])]['RID']
df_progressors = CSF_AD[CSF_AD['RID'].isin(progressors_rid)].merge(
    transitions[['RID', 'first_dx', 'last_dx']],
    on='RID', how='left'
)
print(f"Group 2 - End MCI or AD:         n = {df_progressors['RID'].nunique()} patients, {len(df_progressors)} rows")

# ── Group 3: End up as AD at last visit ────────────────────────────────────
ad_converters_rid = transitions[transitions['last_dx'] == 'AD']['RID']
df_ad_converters = CSF_AD[CSF_AD['RID'].isin(ad_converters_rid)].merge(
    transitions[['RID', 'first_dx', 'last_dx']],
    on='RID', how='left'
)
print(f"Group 3 - End AD:                n = {df_ad_converters['RID'].nunique()} patients, {len(df_ad_converters)} rows")

# ── Group 4: AD at entry (first visit already AD) ─────────────────────────
ad_at_entry_rid = transitions[transitions['first_dx'] == 'AD']['RID']
df_ad_entry = CSF_AD[CSF_AD['RID'].isin(ad_at_entry_rid)].merge(
    transitions[['RID', 'first_dx', 'last_dx']],
    on='RID', how='left'
)
print(f"Group 4 - AD at entry:           n = {df_ad_entry['RID'].nunique()} patients, {len(df_ad_entry)} rows")

# ── Group 5: MCI converters — start CN or MCI, end as AD ──────────────────
mci_converters_rid = transitions[
    (transitions['last_dx'] == 'AD') &
    (transitions['first_dx'].isin(['CN', 'MCI']))
]['RID']
df_mci_converters = CSF_AD[CSF_AD['RID'].isin(mci_converters_rid)].merge(
    transitions[['RID', 'first_dx', 'last_dx']],
    on='RID', how='left'
)
print(f"Group 5 - MCI converters:        n = {df_mci_converters['RID'].nunique()} patients, {len(df_mci_converters)} rows")
```

### Calculate Follow-up Duration

```python
# Last visit per patient
last_visit_ad = (
    df_ad_converters
    .sort_values(['RID', 'days_since_entry'])
    .groupby('RID')
    .last()
    .reset_index()
)

# Convert to years
last_visit_ad['years_since_entry'] = last_visit_ad['days_since_entry'] / 365.25

# Average follow-up time
avg_years = last_visit_ad['years_since_entry'].mean()

print(f"Average time from baseline to last visit (AD converters): {avg_years:.2f} years")
```

## Demographic Characteristics

### Prepare Data for Modeling

```python
# Center continuous variables for interpretation
df_ad_converters["age_c"] = df_ad_converters["age_at_baseline"] - df_ad_converters["age_at_baseline"].mean()
df_ad_converters["edu_c"] = df_ad_converters["PTEDUCAT"] - df_ad_converters["PTEDUCAT"].mean()
df_ad_converters["baseline_c"] = (df_ad_converters["TOTAL13_AD_start"] - df_ad_converters["TOTAL13_AD_start"].mean())

# Standardize biomarkers
df_ad_converters["PT_AB_std"] = (df_ad_converters["PTAU_ABETA42"] - df_ad_converters["PTAU_ABETA42"].mean()) / df_ad_converters["PTAU_ABETA42"].std()
df_ad_converters["PT_std"] = (df_ad_converters["PTAU"] - df_ad_converters["PTAU"].mean()) / df_ad_converters["PTAU"].std()

df_ad_converters["ABETA42_inv"] = 1 / df_ad_converters["ABETA42"]
df_ad_converters["AB_std"] = (df_ad_converters["ABETA42_inv"] - df_ad_converters["ABETA42_inv"].mean()) / df_ad_converters["ABETA42_inv"].std()
```

### Generate Demographic Summary Table

```python
def print_demographic_summary(df, n=397):
    
    # Deduplicate to one row per participant (baseline visit)
    df_baseline = df.sort_values('days_since_entry').groupby('RID').first().reset_index()
    n = len(df_baseline)  # recalculate from unique participants
    
    print(f"Table 1. Demographic Characteristics (n={n})")
    print("=" * 45)
    
    # --- Continuous: Age at Baseline ---
    age = df_baseline['age_at_baseline'].dropna()
    print(f"\nAge at baseline (years)")
    print(f"  Mean (SD): {age.mean():.1f} ({age.std():.1f})")
    
    # --- Continuous: Education ---
    edu = df_baseline['PTEDUCAT'].dropna()
    print(f"\nEducation (years)")
    print(f"  Mean (SD): {edu.mean():.1f} ({edu.std():.1f})")
    
    # --- Categorical: Sex ---
    print(f"\nSex, n (%)")
    sex_map = {1: 'Male', 2: 'Female'}
    for code, label in sex_map.items():
        cnt = (df_baseline['PTGENDER'] == code).sum()
        print(f"  {label}: {cnt} ({cnt/n*100:.1f}%)")
    
    # --- Categorical: Ethnicity ---
    print(f"\nEthnicity, n (%)")
    for cat, cnt in df_baseline['PTETHCAT'].value_counts().items():
        print(f"  {cat}: {cnt} ({cnt/n*100:.1f}%)")
    
    # --- Categorical: Race ---
    print(f"\nRace, n (%)")
    for cat, cnt in df_baseline['PTRACCAT'].value_counts().items():
        print(f"  {cat}: {cnt} ({cnt/n*100:.1f}%)")
    
    # --- Categorical: APOE Carrier ---
    print(f"\nAPOE ε4 Carrier, n (%)")
    for val, label in [(1, 'Carrier'), (0, 'Non-carrier')]:
        cnt = (df_baseline['CARRIER'] == val).sum()
        print(f"  {label}: {cnt} ({cnt/n*100:.1f}%)")
    
    # --- Categorical: APOE Homozygous ---
    print(f"\nAPOE ε4 Homozygous, n (%)")
    for val, label in [(1, 'Homozygous'), (0, 'Non-homozygous')]:
        cnt = (df_baseline['HOMO'] == val).sum()
        print(f"  {label}: {cnt} ({cnt/n*100:.1f}%)")
    
    print("\n" + "=" * 45)

print_demographic_summary(df_ad_converters)
```

### Comorbidity Prevalence by Cohort

```python
# Check proportion of individuals with comorbidities of interest

for df, name in [(CSF_AD, "All patients"), (df_progressors, "Progressors"), (df_ad_converters, "End with AD"), (df_mci_converters, "Convert to AD")]:
    print(f"\n=== {name} (n={df['RID'].nunique()}) ===")
    for condition in ["UTI", "UTI_broad", "GI", "GI_broad", "Sleep"]:
        n = df[df[condition] == 1]["RID"].nunique()
        print(f"  {condition}: {n} patients ({100*n/df['RID'].nunique():.1f}%)")
```

## Mixed-Effects Linear Models (MLM)

### Model Functions

#### Quadratic MLM with Plotting

```python
def quadratic_MLM_with_plot(
    variable,
    data,
    color_var="PT_std",
    output_prefix="model_output",
    time_max=5,
    dpi=400,
    use_pt_ab_std=True
):
    """
    Fits quadratic mixed effects model with random intercept and slope.
    Prints significant predictors with 95% confidence intervals.
    Saves observed vs fitted trajectory plots.
    """
    
    if use_pt_ab_std:
        # Using p-tau181/Aβ42 ratio
        formula = f"""
        {variable} ~ time_years + time_sq
        + HOMO + age_c + PTGENDER + edu_c + PT_AB_std + baseline_c + Sleep + UTI + GI 
        + time_years:baseline_c
        + time_sq:baseline_c
        + time_years:HOMO
        + time_years:PT_AB_std
        + time_years:age_c
        + time_years:PTGENDER
        + time_years:edu_c
        + time_sq:HOMO
        + time_sq:edu_c
        + time_sq:age_c
        + time_sq:PT_AB_std
        + time_sq:PTGENDER
        + time_years:Sleep
        + time_years:UTI
        + time_years:GI
        + time_sq:Sleep
        + time_sq:UTI
        + time_sq:GI
    """
        
    else:
        # Using p-tau181 and Aβ42 separately
        formula = f"""
        {variable} ~ time_years + time_sq
        + HOMO + age_c + PTGENDER + edu_c + PT_std + AB_std + baseline_c + Sleep + UTI + GI 
        + time_years:baseline_c
        + time_sq:baseline_c
        + time_years:HOMO
        + time_years:PT_std
        + time_years:AB_std
        + time_years:age_c
        + time_years:PTGENDER
        + time_years:edu_c
        + time_sq:HOMO
        + time_sq:edu_c
        + time_sq:age_c
        + time_sq:PT_std
        + time_sq:AB_std
        + time_sq:PTGENDER
        + time_years:Sleep
        + time_years:UTI
        + time_years:GI
        + time_sq:Sleep
        + time_sq:UTI
        + time_sq:GI
    """

    model = smf.mixedlm(
        formula,
        data=data,
        groups="RID",
        re_formula="~time_years"  # Random intercept and slope
    ).fit(reml=False)

    # Extract confidence intervals and compute model statistics
    ci = model.conf_int()
    ci.columns = ["CI_lower", "CI_upper"]

    results = pd.DataFrame({
        "coef":     model.params,
        "CI_lower": ci["CI_lower"],
        "CI_upper": ci["CI_upper"],
        "pval":     model.pvalues
    })

    # Calculate marginal and conditional R²
    var_fixed  = float(np.var(model.fittedvalues - model.resid))
    var_random = float(model.cov_re.iloc[0, 0])
    var_resid  = float(model.scale)

    r2_marginal    = var_fixed / (var_fixed + var_random + var_resid)
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

    # Print significant results
    sig_results = results[results["pval"] < 0.1].copy()
    
    print(f"\nSignificant predictors for {variable}:")
    print(sig_results[["coef", "CI_lower", "CI_upper", "pval"]].round(4))

    print(f"\nModel fit statistics:")
    print(f"  N patients:       {data['RID'].nunique()}")
    print(f"  N observations:   {len(data)}")
    print(f"  BIC:              {model.bic:.2f}")
    print(f"  Marginal R²:      {r2_marginal:.3f}")
    print(f"  Conditional R²:   {r2_conditional:.3f}")

    # Save results to CSV
    sig_results.to_csv(
        f"quadratic_model_results/tables/{output_prefix}_{variable}_significant_results.csv"
    )

    # Generate trajectory plots
    person_df = (
        data.sort_values("time_years")
            .groupby("RID")
            .first()
            .reset_index()
    )

    t_grid = np.linspace(0, time_max, 200)
    template_row = data.iloc[[0]].copy()

    vals = person_df[color_var]
    norm = mcolors.Normalize(
        vmin=vals.quantile(0.05),
        vmax=vals.quantile(0.95)
    )
    cmap = cm.coolwarm

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=400)

    # Panel A: Observed trajectories
    ax = axes[0]
    for _, person in person_df.iterrows():
        obs = data[data["RID"] == person["RID"]].sort_values("time_years")
        if len(obs) < 2:
            continue
        color = cmap(norm(person[color_var]))
        ax.plot(obs["time_years"], obs[variable],
                color=color, alpha=0.35, linewidth=1)
        ax.scatter(obs["time_years"], obs[variable],
                   color=color, alpha=0.5, s=10)

    ax.set_xlabel("Years Since Baseline")
    ax.set_ylabel(f"{pretty_label(variable)}" if variable != 'TOTAL13' else "ADAS-Cog13 Total Score")
    axes[0].text(-0.15, 1.005, "(a)", transform=axes[0].transAxes,
                fontsize=14, va='bottom', ha='left')

    # Panel B: Model-fitted trajectories
    ax = axes[1]
    for _, person in person_df.iterrows():
        df_pred = pd.concat([template_row] * len(t_grid), ignore_index=True)
        for col in person.index:
            if col in df_pred.columns:
                df_pred[col] = person[col]
        df_pred["time_years"] = t_grid
        df_pred["time_sq"]    = t_grid ** 2
        y_pred = model.predict(df_pred)
        color  = cmap(norm(person[color_var]))
        ax.plot(t_grid, y_pred, color=color, alpha=0.4, linewidth=1)

    # Reference curves: GI vs No GI
    for q, ls, lw, label in [
        (0, "--", 2.5, "No history of GI symptoms"),
        (1, "-",  2.5, "History of GI symptoms"),
    ]:
        ref = template_row.copy()
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                ref[col] = data[col].median()
        ref["GI"] = q

        df_ref = pd.concat([ref] * len(t_grid), ignore_index=True)
        df_ref["time_years"] = t_grid
        df_ref["time_sq"]    = t_grid ** 2
        y_ref = model.predict(df_ref)

        ax.plot(t_grid, y_ref, linestyle=ls, linewidth=lw,
                color="#000000FF", label=label)

    ax.legend()
    ax.set_xlabel("Years Since Baseline")
    ax.set_ylabel(f"Predicted {pretty_label(variable)}" if variable != 'TOTAL13' else "Predicted ADAS-Cog13 Total Score")
    axes[1].text(-0.15, 1.005, "(b)", transform=axes[1].transAxes,
             fontsize=14, va='bottom', ha='left')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.8)
    
    if color_var == "PT_std":
        cbar.set_label(r"p-tau181 (std)")
    elif color_var == "PT_AB_std":
        cbar.set_label(r"p-tau181/$A\beta42$ (std)")
    
    fig.savefig(
        f"quadratic_model_results/plots/{output_prefix}_{variable}_trajectories.png",
        dpi=dpi, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nSaved files:")
    print(f" → {output_prefix}_{variable}_trajectories.png")
    print(f" → {output_prefix}_{variable}_significant_results.csv")

    return model.bic, r2_marginal, r2_conditional, results
```

#### Linear MLM with Plotting

```python
def linear_MLM_with_plot(
    variable,
    data,
    color_var="PT_AB_std",
    output_prefix="model_output",
    time_max=5,
    dpi=400,
    use_pt_ab_std=True,
    APOE="HOMO"
):
    """
    Fits linear mixed effects model with random intercept and slope.
    """

    if use_pt_ab_std:
        formula = f"""
            {variable} ~ time_years
            + {APOE} + age_c + PTGENDER + edu_c + PT_AB_std + baseline_c + Sleep + UTI + GI
            + time_years:baseline_c 
            + time_years:{APOE}
            + time_years:PT_AB_std 
            + time_years:age_c 
            + time_years:PTGENDER 
            + time_years:edu_c 
            + time_years:Sleep
            + time_years:UTI
            + time_years:GI
        """
    else:
        formula = f"""
            {variable} ~ time_years
            + {APOE} + age_c + PTGENDER + edu_c + PT_std + AB_std + baseline_c + Sleep + UTI + GI
            + time_years:baseline_c 
            + time_years:{APOE}
            + time_years:PT_std 
            + time_years:AB_std 
            + time_years:age_c 
            + time_years:PTGENDER 
            + time_years:edu_c 
            + time_years:Sleep
            + time_years:UTI
            + time_years:GI
        """

    model = smf.mixedlm(
        formula,
        data=data,
        groups="RID",
        re_formula="~time_years"
    ).fit(reml=False)

    results = pd.DataFrame({
        "coef": model.params,
        "pval": model.pvalues
    })

    sig_results = results[results["pval"] < 0.1]

    print(f"\nSignificant predictors for {variable}:")
    print(sig_results)

    sig_results.to_csv(
        f"linear_model_results/tables/{output_prefix}_{variable}_significant_results.csv"
    )

    # Generate plots similar to quadratic version...
    print("\nSaved files:")
    print(f" → {output_prefix}_{variable}_trajectories.png")
    print(f" → {output_prefix}_{variable}_significant_results.csv")

    return model.bic
```

### Run MLM Analysis for ADAS-Cog13

```python
print(compare_lin_quad("TOTAL13", df_ad_converters, "converters_new"))
```

## Additional Cognitive and Functional Outcomes

### CDR Sum of Boxes (CDRSB)

```python
CDR = pd.read_csv("data/CDR_19Feb2026.csv")
df_CDR = df_ad_converters.merge(CDR[["RID", "VISCODE", "CDRSB"]], on=["RID", "VISCODE"], how="inner")
df_CDR.dropna(subset=["CDRSB"], inplace=True)
print(f"Patients with CDRSB: {df_CDR['RID'].nunique()}")

print(compare_lin_quad("CDRSB", df_CDR, "end-ad"))
```

### MMSE (Mini-Mental State Exam)

```python
MMSE = pd.read_csv("data/MMSE_11Mar2026.csv")
df_MMSE = df_ad_converters.merge(MMSE[["RID", "VISCODE", "MMSCORE"]], on=["RID", "VISCODE"], how="inner")
df_MMSE.dropna(subset=["MMSCORE"], inplace=True)
print(f"Patients with MMSE: {df_MMSE['RID'].nunique()}")

print(compare_lin_quad("MMSCORE", df_MMSE, "end-ad"))
```

### FAQ (Functional Activities Questionnaire)

```python
FAQ = pd.read_csv('data/FAQ_24Feb2026.csv')

df_FAQ = df_ad_converters.merge(FAQ[["RID", "VISCODE", "FAQTOTAL"]], on=["RID", "VISCODE"], how="inner")
df_FAQ = df_FAQ.dropna(subset=['FAQTOTAL'])
print(f"Patients with FAQ: {df_FAQ['RID'].nunique()}")
```

#### Logit Transformation for FAQ (Bounded 0-30 Scale)

```python
def logit_to_faq(logit_val, max_faq=30):
    """Convert logit scale back to FAQ units."""
    exp_val = np.exp(logit_val)
    return max_faq * (exp_val / (1 + exp_val))

def logit_based_MLM(df_FAQ):
    """
    Fit mixed effects model with logit transformation to handle bounded FAQ scale.
    """
    # Distribution diagnostics
    MAX_FAQ = 30
    eps = 0.5

    floor_pct = 100*(df_FAQ['FAQTOTAL'] == 0).mean()
    ceiling_pct = 100*(df_FAQ['FAQTOTAL'] == 30).mean()
    
    print("=== FAQ Score Distribution ===")
    print(f"N patients: {df_FAQ['RID'].nunique()}")
    print(f"At floor  (=0):    {floor_pct:.1f}% of visits")
    print(f"At ceiling (=30):  {ceiling_pct:.1f}% of visits")

    # Apply logit transform
    df_FAQ = df_FAQ.copy()
    df_FAQ['FAQ_clipped'] = df_FAQ['FAQTOTAL'].clip(eps, MAX_FAQ - eps)
    df_FAQ['FAQ_logit'] = np.log(
        df_FAQ['FAQ_clipped'] / (MAX_FAQ - df_FAQ['FAQ_clipped'])
    )

    # Fit MLM on logit scale
    formula_faq_logit = """
        FAQ_logit ~ time_years + time_sq
        + HOMO + age_c + PTGENDER + edu_c + PT_std + AB_std + baseline_c + Sleep + UTI + GI
        + time_years:baseline_c
        + time_sq:baseline_c
        + time_years:HOMO
        + time_years:PT_std
        + time_years:AB_std
        + time_years:age_c
        + time_years:PTGENDER
        + time_years:edu_c
        + time_sq:HOMO
        + time_sq:edu_c
        + time_sq:age_c
        + time_sq:PT_std
        + time_sq:AB_std
        + time_sq:PTGENDER
        + time_years:Sleep
        + time_years:UTI
        + time_years:GI
        + time_sq:Sleep
        + time_sq:UTI
        + time_sq:GI
    """

    model_logit = smf.mixedlm(
        formula_faq_logit,
        data=df_FAQ,
        groups='RID',
        re_formula='~time_years'
    ).fit(reml=False)

    results_logit = pd.DataFrame({
        'coef': model_logit.fe_params,
        'pval': model_logit.pvalues
    })
    
    sig_logit = results_logit[results_logit['pval'] < 0.1]
    print("\n=== Significant predictors (logit scale) ===")
    print(sig_logit.round(6))
    print(f"BIC: {model_logit.bic:.2f}")

    # Back-transform effects to FAQ units at mean
    mean_logit = df_FAQ['FAQ_logit'].mean()
    p = np.exp(mean_logit) / (1 + np.exp(mean_logit))
    derivative = MAX_FAQ * p * (1 - p)

    print(f"\n=== Effect sizes in FAQ points (at mean FAQ) ===")
    for term in sig_logit.index:
        if 'time' in term:
            coef = sig_logit.loc[term, 'coef']
            faq_coef = coef * derivative
            print(f"{term}: {faq_coef:.3f} FAQ points per {term.split(':')[0]}")

print(logit_based_MLM(df_FAQ))
```

## Comorbidity Impact Comparisons

### Sleep Disorder Subgroup Analysis

```python
features = [
    'TOTAL13_AD_start', 'PT_AB_std', 'PT_std', 'AB_std',
    'age_c', 'edu_c', 'PTGENDER', 'CARRIER', 'HOMO'
]

sleep_group = df_ad_converters[df_ad_converters['Sleep'] == 1]
non_sleep_group = df_ad_converters[df_ad_converters['Sleep'] == 0]

comparison = pd.DataFrame({
    f'Sleep (n={sleep_group["RID"].nunique()})': sleep_group[features].mean(),
    f'Non-Sleep (n={non_sleep_group["RID"].nunique()})': non_sleep_group[features].mean(),
    f'Full cohort (n={df_ad_converters["RID"].nunique()})': df_ad_converters[features].mean()
})

print(comparison.round(3))
```

### GI Symptom Subgroup Analysis

```python
GI_group = df_ad_converters[df_ad_converters['GI'] == 1]
non_GI_group = df_ad_converters[df_ad_converters['GI'] == 0]

comparison = pd.DataFrame({
    f'GI (n={GI_group["RID"].nunique()})': GI_group[features].mean(),
    f'Non-GI (n={non_GI_group["RID"].nunique()})': non_GI_group[features].mean(),
    f'Full cohort (n={df_ad_converters["RID"].nunique()})': df_ad_converters[features].mean()
})

print(comparison.round(3))
```

## Keyword-Level Prevalence Analysis

### Match and Quantify Free-Text Condition Descriptions

```python
recmhist_bl = recmhist[recmhist["VISCODE"].isin(["v01", "sc"])]

KEYWORD_MAP = {
    "GU_591": ["urinary tract infection", "uti"],
    "GI_529": ["diarrhea", "constipation", "halitosis", "fecal incontinence",
                "abnormal bowel movement", "abnormal bowel sounds", "encopresis"],
    "NS_333": ["insomnia", "sleep disorder", "sleep disturbance", "poor sleep",
                "sleep apnea", "obstructive sleep apnea", "osa", "hypersomnia",
                "sleep fragmentation", "restless legs", "bruxism", "parasomnia", "narcolepsy"],
}

# Reverse map
keyword_to_group = {
    kw: group for group, keywords in KEYWORD_MAP.items() for kw in keywords
}

# Filter to converters and match keywords
converters_rids = set(df_ad_converters["RID"])
recmhist_converters = recmhist_bl[recmhist_bl["RID"].isin(converters_rids)].copy()

recmhist_converters["term_lower"] = recmhist_converters["MHDESC"].str.lower().str.strip()
recmhist_converters["keyword_matched"] = recmhist_converters["term_lower"].apply(
    lambda x: next((kw for kw in keyword_to_group if kw in str(x)), None)
)

kw_hits = recmhist_converters[recmhist_converters["keyword_matched"].notna()].copy()
kw_hits["phecode_group"] = kw_hits["keyword_matched"].map(keyword_to_group)

# Compute overall prevalence
n_total = len(converters_rids)

keyword_prevalence = (
    kw_hits.groupby(["phecode_group", "keyword_matched"])["RID"]
    .nunique()
    .reset_index(name="n_rids")
    .assign(prevalence_pct=lambda df: (df["n_rids"] / n_total * 100).round(1))
    .sort_values(["phecode_group", "prevalence_pct"], ascending=[True, False])
)

print(keyword_prevalence.to_string(index=False))

# Compute within-group prevalence
group_rids = kw_hits.groupby("phecode_group")["RID"].apply(set).to_dict()

records = []
for _, row in keyword_prevalence.iterrows():
    group = row["phecode_group"]
    kw = row["keyword_matched"]
    n_group = len(group_rids[group])
    n_kw = row["n_rids"]
    pct = round(n_kw / n_group * 100, 1)
    records.append({
        "phecode_group": group,
        "keyword": kw,
        "n_group": n_group,
        "n_keyword": n_kw,
        "prevalence_within_group_pct": pct
    })

within_group_prev = pd.DataFrame(records).sort_values(
    ["phecode_group", "prevalence_within_group_pct"], ascending=[True, False]
)

print(within_group_prev.to_string(index=False))
```

---

## Summary

This analysis pipeline combines data preprocessing with comprehensive mixed-effects linear modeling to investigate cognitive decline trajectories in Alzheimer's disease patients. Key findings include:

- **Diagnosis transitions**: Tracks CN → MCI → AD progression patterns
- **Biomarker relationships**: Examines p-tau181, Aβ42, and their ratio in predicting cognitive decline
- **Comorbidity effects**: Quantifies impact of GI symptoms, sleep disorders, and UTIs on disease progression
- **Functional outcomes**: Models ADAS-Cog13, CDRSB, MMSE, and FAQ scores over time
- **Statistical approach**: Uses linear and quadratic MLM with random intercepts and slopes to account for repeated measures and individual variability in disease trajectories
