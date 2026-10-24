
# ============================================================
# Blog Post 6
# 06 - Life-Cycle and Measurement-Year Sensitivity
#
# Purpose:
# 1. Match mother-child earnings across four PSID waves.
# 2. Construct a balanced sample of mother-child pairs.
# 3. Estimate single-year IGE for each earnings year.
# 4. Estimate multi-year IGE using average earnings.
# 5. Calculate HC1 robust standard errors and 95% CIs.
# 6. Generate Figure 5 with confidence intervals.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Step 0: Set up project directories
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PAIR_KEYS = [
    "child_interview_id",
    "child_person_number",
    "mother_interview_id",
    "mother_person_number",
]

SURVEY_YEARS = [2017, 2019, 2021, 2023]


# ============================================================
# Step 1: Load mother-child identifiers and income panel
# ============================================================

print("\n=== STEP 1: LOAD DATA ===")

pairs = pd.read_csv(
    DATA_DIR / "mother_child_income_2023.csv"
)

income = pd.read_csv(
    DATA_DIR / "psid_individual_income_long.csv"
)

print("Mother-child records:", len(pairs))
print("Individual-year records:", len(income))

# Retain adult children with observed maternal age.
pairs = pairs.loc[
    (pairs["child_age"] >= 25)
    & pairs["mother_age"].notna(),
    PAIR_KEYS + ["child_age", "mother_age"],
].drop_duplicates(subset=PAIR_KEYS)

# Retain the identifiers and income variables needed.
income = income[
    [
        "interview_id",
        "person_number",
        "year",
        "individual_income",
    ]
].copy()

income["individual_income"] = pd.to_numeric(
    income["individual_income"],
    errors="coerce",
)

income["year"] = pd.to_numeric(
    income["year"],
    errors="coerce",
)

# Check for duplicate person-year records.
duplicates = income.duplicated(
    subset=["interview_id", "person_number", "year"]
).sum()

if duplicates > 0:
    raise ValueError(
        f"Found {duplicates} duplicate person-year records."
    )

print("Eligible adult mother-child pairs:", len(pairs))


# ============================================================
# Step 2: Prepare child and mother income records
# ============================================================

print("\n=== STEP 2: PREPARE EARNINGS RECORDS ===")

child_income = income.rename(
    columns={
        "interview_id": "child_interview_id",
        "person_number": "child_person_number",
        "individual_income": "child_income",
    }
)

mother_income = income.rename(
    columns={
        "interview_id": "mother_interview_id",
        "person_number": "mother_person_number",
        "individual_income": "mother_income",
    }
)

print("Child income records:", len(child_income))
print("Mother income records:", len(mother_income))


# ============================================================
# Step 3: Match earnings by person and survey year
# ============================================================

print("\n=== STEP 3: MATCH MOTHER-CHILD EARNINGS ===")

panel = pairs.merge(
    child_income,
    on=[
        "child_interview_id",
        "child_person_number",
    ],
    how="inner",
    validate="one_to_many",
)

panel = panel.merge(
    mother_income,
    on=[
        "mother_interview_id",
        "mother_person_number",
        "year",
    ],
    how="inner",
    validate="many_to_one",
)

# Keep survey waves used in this analysis.
panel = panel.loc[
    panel["year"].isin(SURVEY_YEARS)
].copy()

# Log earnings require strictly positive income.
panel = panel.loc[
    (panel["child_income"] > 0)
    & (panel["mother_income"] > 0)
].copy()

print("\n=== POSITIVE-EARNINGS PAIRS BY SURVEY YEAR ===")

print(
    panel.groupby("year")
    .size()
    .to_string()
)


# ============================================================
# Step 4: Construct a balanced mother-child sample
# ============================================================

print("\n=== STEP 4: CONSTRUCT BALANCED SAMPLE ===")

# Count distinct survey years for each mother-child pair.
pair_counts = (
    panel.groupby(PAIR_KEYS)["year"]
    .nunique()
    .reset_index(name="n_years")
)

# Require positive earnings for both generations
# in all four survey waves.
balanced_ids = pair_counts.loc[
    pair_counts["n_years"] == len(SURVEY_YEARS),
    PAIR_KEYS,
]

balanced = panel.merge(
    balanced_ids,
    on=PAIR_KEYS,
    how="inner",
    validate="many_to_one",
)

n_balanced_pairs = len(balanced_ids)

print("Balanced mother-child pairs:", n_balanced_pairs)
print("Balanced person-pair-year observations:", len(balanced))

# Each pair should appear exactly once in each wave.
expected_rows = n_balanced_pairs * len(SURVEY_YEARS)

if len(balanced) != expected_rows:
    raise ValueError(
        "Balanced panel validation failed: "
        f"expected {expected_rows} observations, "
        f"found {len(balanced)}."
    )

if n_balanced_pairs < 30:
    print(
        "WARNING: Fewer than 30 balanced pairs. "
        "Interpret estimates cautiously."
    )

if n_balanced_pairs < 3:
    raise ValueError(
        "Insufficient balanced mother-child pairs "
        "for regression."
    )


# ============================================================
# Step 5: Define IGE estimation function
# ============================================================

print("\n=== STEP 5: ESTIMATE SINGLE-YEAR IGE ===")


def estimate_ige(df):
    """
    Estimate:

        log(child_income)
            = alpha + beta * log(mother_income) + error

    Return:
        - Number of observations
        - IGE coefficient
        - HC1 heteroskedasticity-robust standard error
        - Approximate 95% confidence interval

    Note:
        HC1 does not account for clustering among siblings
        or repeated observations from the same mother.
    """

    x = np.log(
        df["mother_income"].to_numpy(dtype=float)
    )

    y = np.log(
        df["child_income"].to_numpy(dtype=float)
    )

    n = len(x)

    if n <= 2:
        raise ValueError(
            "At least three observations are required."
        )

    # Design matrix: intercept + log maternal earnings.
    X = np.column_stack(
        [np.ones(n), x]
    )

    if np.linalg.matrix_rank(X) < 2:
        raise ValueError(
            "Insufficient variation in maternal earnings."
        )

    # Ordinary least squares.
    coefficients = np.linalg.lstsq(
        X, y, rcond=None
    )[0]

    residuals = y - X @ coefficients

    # HC1 heteroskedasticity-robust covariance matrix.
    xtx_inv = np.linalg.inv(X.T @ X)

    meat = X.T @ (
        X * residuals[:, None] ** 2
    )

    k = X.shape[1]

    vcov = (
        n / (n - k)
    ) * xtx_inv @ meat @ xtx_inv

    beta = float(coefficients[1])

    robust_se = float(
        np.sqrt(max(vcov[1, 1], 0))
    )

    return {
        "n": n,
        "beta": beta,
        "robust_se": robust_se,
        "ci_lower": beta - 1.96 * robust_se,
        "ci_upper": beta + 1.96 * robust_se,
    }


# Estimate IGE separately for each survey wave.
results = []

for year in SURVEY_YEARS:

    subset = balanced.loc[
        balanced["year"] == year
    ].copy()

    result = estimate_ige(subset)

    results.append(
        {
            "survey_year": year,
            "income_year": year - 1,
            "n_pairs": result["n"],
            "single_year_ige": result["beta"],
            "robust_se": result["robust_se"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
        }
    )

results_df = pd.DataFrame(results)

print("\n=== SINGLE-YEAR IGE BY MEASUREMENT YEAR ===")

print(
    results_df.round(4).to_string(index=False)
)

# Save results.
single_year_output = (
    TABLE_DIR / "ige_by_measurement_year.csv"
)

results_df.to_csv(
    single_year_output,
    index=False,
)

print("\nSaved:", single_year_output)


# ============================================================
# Step 6: Estimate multi-year IGE on the same sample
# ============================================================

print("\n=== STEP 6: ESTIMATE MULTI-YEAR IGE ===")

# Average positive earnings across four waves.
# Every pair in this balanced sample has positive
# earnings in all four observed years.
average_income = (
    balanced.groupby(PAIR_KEYS, as_index=False)
    .agg(
        child_avg_income=("child_income", "mean"),
        mother_avg_income=("mother_income", "mean"),
    )
)

# Use the same variable names as the regression function.
average_income["child_income"] = (
    average_income["child_avg_income"]
)

average_income["mother_income"] = (
    average_income["mother_avg_income"]
)

multi_result = estimate_ige(
    average_income
)

multi_beta = multi_result["beta"]

print("\n=== MULTI-YEAR IGE: SAME BALANCED SAMPLE ===")

print(
    f"N: {multi_result['n']}"
)

print(
    f"IGE: {multi_result['beta']:.4f}"
)

print(
    f"HC1 robust SE: {multi_result['robust_se']:.4f}"
)

print(
    "95% CI: "
    f"[{multi_result['ci_lower']:.4f}, "
    f"{multi_result['ci_upper']:.4f}]"
)

# Save multi-year results.
multi_year_output = (
    TABLE_DIR / "ige_balanced_multiyear.csv"
)

pd.DataFrame(
    [
        {
            "n_pairs": multi_result["n"],
            "multi_year_ige": multi_result["beta"],
            "robust_se": multi_result["robust_se"],
            "ci_lower": multi_result["ci_lower"],
            "ci_upper": multi_result["ci_upper"],
        }
    ]
).to_csv(
    multi_year_output,
    index=False,
)

print("\nSaved:", multi_year_output)


# ============================================================
# Step 7: Generate Figure 5 with 95% confidence intervals
# ============================================================

print("\n=== STEP 7: GENERATE FIGURE 5 ===")

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = results_df[
    "income_year"
].to_numpy()

beta = results_df[
    "single_year_ige"
].to_numpy()

lower_error = (
    beta
    - results_df["ci_lower"].to_numpy()
)

upper_error = (
    results_df["ci_upper"].to_numpy()
    - beta
)

# Single-year estimates and confidence intervals.
ax.errorbar(
    x,
    beta,
    yerr=np.vstack(
        [lower_error, upper_error]
    ),
    fmt="o-",
    capsize=5,
    markersize=7,
    linewidth=2,
    label="Single-year IGE (95% CI)",
)

# Multi-year estimate.
ax.axhline(
    multi_beta,
    linestyle="--",
    linewidth=2,
    label="Multi-year IGE",
)

# Multi-year confidence interval.
ax.axhspan(
    multi_result["ci_lower"],
    multi_result["ci_upper"],
    alpha=0.12,
    label="Multi-year 95% CI",
)

# Zero reference line.
ax.axhline(
    0,
    color="gray",
    linewidth=1,
    alpha=0.6,
)

ax.set_xticks(x)

ax.set_xlabel(
    "Earnings Reference Year"
)

ax.set_ylabel(
    "Intergenerational Earnings Elasticity (IGE)"
)

ax.set_title(
    "IGE Sensitivity to Earnings Measurement Year",
    fontsize=14,
    fontweight="bold",
)

ax.legend(
    loc="best"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

# Use the actual sample size in the figure note.
fig.text(
    0.5,
    0.01,
    (
        f"Notes: Same {n_balanced_pairs} mother-child pairs "
        "in all four years. Error bars and shaded band "
        "show 95% HC1 robust confidence intervals."
    ),
    ha="center",
    fontsize=8,
)

plt.tight_layout(
    rect=[0, 0.06, 1, 1]
)

figure_output = (
    FIGURE_DIR
    / "figure5_ige_by_measurement_year.png"
)

plt.savefig(
    figure_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nFigure 5 saved:", figure_output)


# ============================================================
# Step 8: Final validation
# ============================================================

print("\n=== STEP 8: FINAL VALIDATION ===")

required_outputs = [
    single_year_output,
    multi_year_output,
    figure_output,
]

for path in required_outputs:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output not found: {path}"
        )

    print("Verified:", path.name)

print("\n=== ANALYSIS SUMMARY ===")

print(
    f"Balanced mother-child pairs: {n_balanced_pairs}"
)

print(
    f"Number of earnings years: {len(SURVEY_YEARS)}"
)

print(
    f"Multi-year IGE: {multi_beta:.4f}"
)

print(
    "\nInterpretation: Single-year IGE estimates may "
    "vary with earnings measurement year. Multi-year "
    "averaging can reduce temporary income fluctuations, "
    "but does not automatically eliminate life-cycle bias."
)

print(
    "\n06_lifecycle_analysis.py completed successfully."
)
