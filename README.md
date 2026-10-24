
# Blog Post 6: Matching Parents and Children with SQL in Python

## Project Overview

This project uses longitudinal data from the Panel Study of Income Dynamics (PSID) to investigate intergenerational earnings mobility.

The analysis demonstrates how SQL within Python can be used to:

1. Match birth mothers and adult children using stable individual identifiers.
2. Connect individual records to family-level earnings information.
3. Construct single-year and multi-year labor earnings measures.
4. Estimate intergenerational earnings elasticity (IGE).
5. Examine the sensitivity of IGE estimates to income measurement year and maternal age.

The accompanying `index.qmd` presents the analysis as a tutorial for students familiar with basic Python and SQL.

## Research Question

How does the estimated intergenerational earnings elasticity differ when earnings are measured in a single year versus averaged across several years, and what does this reveal about potential life-cycle measurement bias?

## Repository Structure

```text
Blog6/
├── index.qmd
├── requirements.txt
├── README.md
├── code/
│   ├── 01_load_data.py
│   ├── 02_clean_data.py
│   ├── 03_sql_matching.py
│   ├── 04_ige_analysis.py
│   ├── 05_visualizations.py
│   └── 06_lifecycle_analysis.py
├── data-raw/
│   ├── ind2023er/
│   │   ├── IND2023ER.txt
│   │   └── IND2023ER.do
│   ├── pid23/
│   │   ├── PID23.txt
│   │   └── PID23.do
│   └── J366546/
│       └── J366546.xlsx
├── data/
└── results/
    ├── tables/
    └── figures/
        ├── figure1_single_year_ige.png
        ├── figure2_multi_year_ige.png
        ├── figure3_ige_comparison.png
        ├── figure4_ige_by_mother_age.png
        └── figure5_ige_by_measurement_year.png
```

The `data-raw/` folder must be populated locally with PSID source files before running the scripts. Raw PSID records are excluded from this public-ready package and should not be committed to a public repository without checking data-access and redistribution terms.

The `data/` folder contains generated intermediate datasets and is excluded from version control by `.gitignore`. The scripts generate these files when run with the required inputs.

The `results/tables/` folder contains regression estimates, bootstrap results, and summary tables.

The `results/figures/` folder contains programmatically generated figures; root-level copies of the five figures are referenced by `index.qmd`.

## Data Source

Data are obtained from the Panel Study of Income Dynamics (PSID), Survey Research Center, University of Michigan.

Official website:

https://psidonline.isr.umich.edu/

The project uses:

- PSID 2023 individual file.
- PSID 2023 parent-identification file.
- A customized family income extract containing observations from the 2017, 2019, 2021, and 2023 survey waves.

The customized income extract is stored locally as `J366546.xlsx`.

### Obtaining the Raw PSID Files

The raw data are **not included** in the public-ready repository. To reproduce the project:

1. Visit the [official PSID website](https://psidonline.isr.umich.edu/) and follow its data access and registration instructions, if applicable.
2. Obtain the **2023 Individual File** in fixed-width text format together with its Stata dictionary (`IND2023ER.txt` and `IND2023ER.do`). Place both files under `data-raw/ind2023er/`.
3. Obtain the **2023 Parent Identification File** and its Stata dictionary (`PID23.txt` and `PID23.do`). Place both files under `data-raw/pid23/`.
4. Through the PSID data extraction interface, select the **2017, 2019, 2021, and 2023 family interview variables** listed in the Income Variables table below, including each year's family interview ID and reference-person and spouse labor earnings. Export the extract to Excel (`.xlsx`).
5. The original project used the custom extract `J366546.xlsx`, placed under `data-raw/J366546/`. A new PSID extract may have a different name. Rename it to `J366546.xlsx` **only if it contains the required variables**, or update the expected filename in `code/01_load_data.py`.
6. Before running the scripts, confirm the files and folders match the tree above. The scripts use paths relative to the project root, not the location of the terminal's current working directory.

The identifier `J366546` is a project-specific extract filename, **not** a standard PSID dataset name. This project uses the PSID variable codes documented below; another user can obtain a new extract with the same variables. Follow PSID's applicable access and redistribution conditions and do not publish restricted individual-level records.


### Income Variables

| Survey year | Income year | Family ID | Reference-person earnings | Spouse earnings |
|---|---|---|---|---|
| 2017 | 2016 | ER66002 | ER71293 | ER71321 |
| 2019 | 2018 | ER72002 | ER77315 | ER77343 |
| 2021 | 2020 | ER78002 | ER81642 | ER81670 |
| 2023 | 2022 | ER82002 | ER85496 | ER85524 |

Stable individual identifiers:

- `ER30001`
- `ER30002`

Parent-identification variables:

- `PID2`, `PID3`: Individual identifiers.
- `PID4`, `PID5`: Birth-mother identifiers.

## Software Requirements

Recommended environment:

- Python 3.10 or later
- pandas
- numpy
- openpyxl
- matplotlib
- statsmodels
- SQLite through Python's built-in sqlite3 package
- Quarto CLI

Install the required Python packages from the project root:

```bash
python -m pip install -r requirements.txt
```

If using Conda's `base` environment instead, install packages into that environment and run the `conda run -n base` commands below. `sqlite3` is included with standard Python; it does not require a separate database server. Quarto CLI must be installed separately to render the blog post.

## Reproducing the Analysis

Run the scripts in the following order from the `Blog6` project root:

```bash
python code/01_load_data.py
python code/02_clean_data.py
python code/03_sql_matching.py
python code/04_ige_analysis.py
python code/05_visualizations.py
python code/06_lifecycle_analysis.py
```

For the author's Conda base environment, use:

```bash
conda run -n base python code/01_load_data.py
conda run -n base python code/02_clean_data.py
conda run -n base python code/03_sql_matching.py
conda run -n base python code/04_ige_analysis.py
conda run -n base python code/05_visualizations.py
conda run -n base python code/06_lifecycle_analysis.py
```

### Script Descriptions

**01_load_data.py**

Loads the PSID individual, parent-identification, and family income files. Selects variables required for the analysis.

**02_clean_data.py**

Cleans identifiers, checks data structure, and converts individual records into a longitudinal person-year format.

**03_sql_matching.py**

Uses SQL joins to identify birth mothers and children. Connects individuals to family-year income records and assigns individual labor earnings.

**04_ige_analysis.py**

Constructs single-year and multi-year earnings measures. Estimates IGE regressions, age-group comparisons, and bootstrap differences.

**05_visualizations.py**

Generates the first four figures, including earnings relationships, IGE comparisons, and maternal-age sensitivity.

**06_lifecycle_analysis.py**

Constructs a balanced mother–child sample, estimates year-specific IGE regressions with HC1 robust confidence intervals, and generates Figure 5.

### Generated Outputs and Replication Checks

The scripts create intermediate CSV datasets under `data/`, summary and regression tables under `results/tables/`, and PNG figures under `results/figures/`. In particular, check for:

- `results/tables/ige_regression_results.csv`
- `results/tables/ige_bootstrap_comparison.csv`
- `results/tables/ige_age_group_comparison.csv`
- `results/tables/ige_by_measurement_year.csv`
- `results/tables/ige_balanced_multiyear.csv`
- `results/figures/figure1_single_year_ige.png` through `figure5_ige_by_measurement_year.png`

After a successful run, compare the generated tables with the reference results below: the common-sample comparison should contain **1,027 mother–child pairs** with IGE point estimates **0.0322** (single year) and **0.0641** (multi-year); the balanced sample should contain **489 pairs** with four-wave IGE **0.0432**. Confirm that the paired-bootstrap difference and confidence interval also agree with the reported values. Minor numerical variation can arise from software versions, though the bootstrap uses a fixed random seed in the supplied script.

The blog's displayed tables are written in `index.qmd` rather than dynamically loaded from the generated CSV files. **If you change the underlying data or model, regenerate the outputs and update the blog's displayed numbers accordingly.** The five figures are saved programmatically by the plotting scripts; copies at the repository root are used by the relative image links in `index.qmd`. If the figures are regenerated, update these root-level copies before rendering or publishing.

If a script fails, check the first error message, verify the exact raw input filenames and required variable columns, and rerun the scripts in sequence. The raw-data-dependent end-to-end run has **not been independently verified in this packaged copy**, because the original PSID source files are excluded.

## Main Results

The parent–child matching procedure produces 4,242 adult mother–child pairs with available age information.

The single-year 2022 regression uses 1,178 pairs.

The multi-year regression uses 1,602 pairs.

The common-sample comparison uses 1,027 pairs:

| Earnings measure | IGE |
|---|---:|
| Single-year earnings | 0.0322 |
| Multi-year earnings | 0.0641 |

The paired-bootstrap difference is 0.0318, with a 95% confidence interval of [-0.0081, 0.0759].

The interval includes zero, so the difference is not statistically significant at the 5% level.

### Measurement-Year Sensitivity

The balanced sample contains 489 mother–child pairs with positive earnings in all four income years.

| Income year | Single-year IGE |
|---|---:|
| 2016 | 0.0410 |
| 2018 | 0.0051 |
| 2020 | 0.0305 |
| 2022 | 0.0558 |

The IGE based on four-wave average earnings is 0.0432.

These results illustrate measurement-year sensitivity but do not establish that life-cycle effects alone explain the variation.

## Methodological Limitations

The analysis focuses on birth mothers' individual labor earnings rather than total parental household income.

Multi-year averages use positive observed nominal earnings and exclude zero-income years.

The data contain four intermittent income observations rather than complete lifetime earnings histories.

The analysis does not apply PSID survey weights or inflation adjustments.

HC1 robust standard errors do not account for possible clustering among siblings or mothers.

The IGE estimates describe intergenerational earnings associations and should not be interpreted as causal effects.

## Repository and replication notes

The blog source is `index.qmd` at the project root; its image references point to figure files in the same directory. The five published figures are also saved under `results/figures/`. Run all six scripts before rendering to regenerate analysis outputs. The README documents expected raw input filenames; these must be obtained separately from PSID. The archive omits raw and intermediate person-level records to avoid unintended redistribution.

The scripts were inspected for output-writing operations, but the full pipeline has not been rerun in this delivery because the raw PSID source files are not included. Reported estimates are preserved from the supplied analysis, not independently recomputed.

## Rendering the Blog Post

After running the analysis and generating the figures, render the Quarto document:

```bash
quarto render index.qmd --to html
```

Check that all five figures, tables, SQL code examples, and mathematical expressions display correctly.

If the blog post is moved into a separate website repository, copy the figures into the appropriate relative directory and update the image paths in `index.qmd`.

## References

Panel Study of Income Dynamics (PSID), Survey Research Center, Institute for Social Research, University of Michigan.

Solon, G. (1992). Intergenerational Income Mobility in the United States. American Economic Review, 82(3), 393–408.

Haider, S., & Solon, G. (2006). Life-Cycle Variation in the Association between Current and Lifetime Earnings. American Economic Review, 96(4), 1308–1320.
