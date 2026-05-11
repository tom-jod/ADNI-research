This repository allows for reviewers to replicate and explore the methodology behind "Identifying pre-dementia medical conditions with the greatest impact on clinical outcomes in Alzheimer's disease".

1-preprocessing.ipynb will combine the cognitive tests with demographic data, baseline visit medical history and baseline CSF data. These results are then saved to df_CSF.csv which is loaded into 2-MLM_comparisons.ipynb.

2-MLM_comparisons.ipynb constructs mixed linear models with either linear or linear and quadratic terms with respect to time. These models are then trained to predict behavioural scores (ADAS-Cog, MMSE, CDR-SB or FAQ) following the baseline visit for AD confirmed (n=397) individuals or AD converters (n=212, who start as CN or MCI).

The anaylsis was ran using python version 3.13.9 using standard python libraries.

