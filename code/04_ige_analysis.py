
"""Blog Post 6: Intergenerational Earnings Elasticity (IGE).

Run from the project root:
    conda run -n base python code/04_ige_analysis.py

Inputs (created by 03_sql_matching.py):
    data/mother_child_income_2023.csv
    data/psid_individual_income_long.csv

Outputs:
    data/psid_average_income.csv
    data/ige_single_year_sample.csv
    data/ige_multi_year_sample.csv
    data/ige_common_sample.csv
    results/tables/ige_regression_results.csv
    results/tables/ige_age_group_comparison.csv
    results/tables/ige_bootstrap_comparison.csv
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PAIR_KEYS = [
    "child_interview_id",
    "child_person_number",
    "mother_interview_id",
    "mother_person_number"
]

SEED = 6400
BOOTSTRAP_REPS = 1000


# ============================================================
# Step 1: Define OLS regression function
# ============================================================

def estimate_ige(df, child_col, mother_col, model_name):
    """OLS log earnings elasticity with HC1 robust standard errors."""

    sample = df.loc[
        (df[child_col] > 0)
        & (df[mother_col] > 0)
    ].copy()

    sample["log_child_income"] = np.log(sample[child_col])
    sample["log_parent_income"] = np.log(sample[mother_col])

    sample = sample.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["log_child_income", "log_parent_income"]
    )

    n = len(sample)

    if n < 3:
        raise ValueError(
            f"{model_name}: fewer than 3 usable pairs"
        )

    x = sample["log_parent_income"].to_numpy(dtype=float)
    y = sample["log_child_income"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(n), x])

    if np.linalg.matrix_rank(X) < 2:
        raise ValueError(
            f"{model_name}: no variation in mother's log earnings"
        )

    coefficients = np.linalg.lstsq(
        X, y, rcond=None
    )[0]

    residuals = y - X @ coefficients

    xtx_inv = np.linalg.inv(X.T @ X)

    meat = X.T @ (
        X * residuals[:, None] ** 2
    )

    vcov = (
        (n / (n - 2))
        * xtx_inv
        @ meat
        @ xtx_inv
    )

    se = float(
        np.sqrt(max(vcov[1, 1], 0.0))
    )

    ss_total = float(
        np.sum((y - y.mean()) ** 2)
    )

    result = {
        "model": model_name,
        "n": n,
        "alpha": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "robust_se": se,
        "ci_lower": float(coefficients[1] - 1.96 * se),
        "ci_upper": float(coefficients[1] + 1.96 * se),
        "r_squared": (
            float(1 - np.sum(residuals ** 2) / ss_total)
            if ss_total > 0
            else np.nan
        )
    }

    print(
        f"{model_name}: "
        f"n={n}, "
        f"beta={result['beta']:.4f}, "
        f"HC1 SE={se:.4f}, "
        f"95% CI=[{result['ci_lower']:.4f}, "
        f"{result['ci_upper']:.4f}]"
    )

    return result, sample


# ============================================================
# Step 2: Define bootstrap functions
# ============================================================

def fast_slope(log_x, log_y):
    """Calculate OLS slope with an intercept."""

    x0 = log_x - log_x.mean()

    denominator = np.dot(x0, x0)

    if denominator <= 1e-12:
        return np.nan

    return float(
        np.dot(
            x0,
            log_y - log_y.mean()
        ) / denominator
    )


def bootstrap_difference(
    df,
    reps=BOOTSTRAP_REPS,
    seed=SEED
):
    """
    Paired bootstrap:
    multi-year IGE minus single-year IGE.

    Resampling is performed at the mother-child pair level.
    Shared mothers across pairs are not clustered.
    """

    cols = [
        "child_income",
        "mother_income",
        "child_avg_income",
        "mother_avg_income"
    ]

    arr = df[cols].to_numpy(dtype=float)

    if (
        len(arr) < 3
        or not np.isfinite(arr).all()
        or not (arr > 0).all()
    ):
        raise ValueError(
            "Bootstrap requires at least 3 pairs "
            "with all four positive earnings."
        )

    z = np.log(arr)

    observed = (
        fast_slope(z[:, 3], z[:, 2])
        - fast_slope(z[:, 1], z[:, 0])
    )

    rng = np.random.default_rng(seed)

    differences = []
    n = len(z)

    for _ in range(reps):

        sampled = z[
            rng.integers(0, n, n)
        ]

        b_multi = fast_slope(
            sampled[:, 3],
            sampled[:, 2]
        )

        b_single = fast_slope(
            sampled[:, 1],
            sampled[:, 0]
        )

        if (
            np.isfinite(b_multi)
            and np.isfinite(b_single)
        ):
            differences.append(
                b_multi - b_single
            )

    if len(differences) < reps * 0.95:
        raise ValueError(
            "Too many bootstrap draws failed."
        )

    lo, hi = np.percentile(
        differences,
        [2.5, 97.5]
    )

    return {
        "n": n,
        "bootstrap_reps": len(differences),
        "difference_multi_minus_single": observed,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_excludes_zero": bool(
            lo > 0 or hi < 0
        )
    }


# ============================================================
# Step 3: Load datasets
# ============================================================

print("\n=== PROJECT DIRECTORY ===")
print(BASE_DIR)

pairs_2023 = pd.read_csv(
    DATA_DIR / "mother_child_income_2023.csv"
)

individual_income = pd.read_csv(
    DATA_DIR / "psid_individual_income_long.csv"
)

print("\n=== DATA LOADED ===")

print("Mother-child pairs:", pairs_2023.shape)
print("Individual income:", individual_income.shape)

if pairs_2023.duplicated(PAIR_KEYS).any():
    raise ValueError(
        "Duplicate mother-child pair IDs in 2023 input."
    )


# ============================================================
# Step 4: Single-year IGE
# ============================================================

single_year_pairs = pairs_2023.loc[
    (pairs_2023["child_age"] >= 25)
    & pairs_2023["mother_age"].notna()
    & (pairs_2023["child_income"] > 0)
    & (pairs_2023["mother_income"] > 0)
].copy()

print("\n=== BASELINE SINGLE-YEAR IGE ===")

single_result, single_sample = estimate_ige(
    single_year_pairs,
    "child_income",
    "mother_income",
    "Single-Year IGE (2022 income)"
)


# ============================================================
# Step 5: Prepare multi-year income panel
# ============================================================

year_mapping = {
    2017: 2016,
    2019: 2018,
    2021: 2020,
    2023: 2022
}

income_panel = individual_income[
    [
        "interview_id",
        "person_number",
        "year",
        "individual_income"
    ]
].copy()

income_panel["income_year"] = (
    income_panel["year"].map(year_mapping)
)

if income_panel["income_year"].isna().any():
    raise ValueError(
        "Unexpected survey year in income panel."
    )

if income_panel.duplicated(
    ["interview_id", "person_number", "year"]
).any():
    raise ValueError(
        "Duplicate person-year observations."
    )


# ============================================================
# Step 6: SQL multi-year averaging
# ============================================================

# Important:
# This averages positive nominal earnings only.
# Zero-income years are excluded.

with sqlite3.connect(":memory:") as conn:

    income_panel.to_sql(
        "income_panel",
        conn,
        if_exists="replace",
        index=False
    )

    query = """
    SELECT
        interview_id,
        person_number,
        COUNT(*) AS n_positive_years,
        AVG(individual_income) AS avg_income,
        MIN(income_year) AS first_income_year,
        MAX(income_year) AS last_income_year

    FROM income_panel

    WHERE individual_income > 0

    GROUP BY
        interview_id,
        person_number
    """

    average_income = pd.read_sql_query(
        query, conn
    )

average_income.to_csv(
    DATA_DIR / "psid_average_income.csv",
    index=False
)

print("\n=== SQL AVERAGING ===")

print(
    "People with positive earnings in >=1 wave:",
    len(average_income)
)

print(
    average_income["n_positive_years"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Step 7: Match average income to mother-child pairs
# ============================================================

pair_ids = pairs_2023.loc[
    (pairs_2023["child_age"] >= 25)
    & pairs_2023["mother_age"].notna(),
    PAIR_KEYS + ["child_age", "mother_age"]
].copy()

child_avg = average_income.rename(
    columns={
        "interview_id": "child_interview_id",
        "person_number": "child_person_number",
        "n_positive_years": "child_n_years",
        "avg_income": "child_avg_income",
        "first_income_year": "child_first_income_year",
        "last_income_year": "child_last_income_year"
    }
)

mother_avg = average_income.rename(
    columns={
        "interview_id": "mother_interview_id",
        "person_number": "mother_person_number",
        "n_positive_years": "mother_n_years",
        "avg_income": "mother_avg_income",
        "first_income_year": "mother_first_income_year",
        "last_income_year": "mother_last_income_year"
    }
)

multi_pairs = pair_ids.merge(
    child_avg,
    on=[
        "child_interview_id",
        "child_person_number"
    ],
    how="left",
    validate="many_to_one"
)

multi_pairs = multi_pairs.merge(
    mother_avg,
    on=[
        "mother_interview_id",
        "mother_person_number"
    ],
    how="left",
    validate="many_to_one"
)


# ============================================================
# Step 8: Multi-year sample and IGE
# ============================================================

multi_sample = multi_pairs.loc[
    (multi_pairs["child_n_years"] >= 2)
    & (multi_pairs["mother_n_years"] >= 2)
    & (multi_pairs["child_avg_income"] > 0)
    & (multi_pairs["mother_avg_income"] > 0)
].copy()

print("\n=== BASELINE MULTI-YEAR IGE ===")

multi_result, _ = estimate_ige(
    multi_sample,
    "child_avg_income",
    "mother_avg_income",
    "Multi-Year IGE (2+ positive waves)"
)


# ============================================================
# Step 9: Common-sample comparison
# ============================================================

common_sample = multi_sample.merge(
    single_year_pairs[
        PAIR_KEYS + [
            "child_income",
            "mother_income"
        ]
    ],
    on=PAIR_KEYS,
    how="inner",
    validate="one_to_one"
)

print("\n=== COMMON-SAMPLE COMPARISON ===")

common_single_result, _ = estimate_ige(
    common_sample,
    "child_income",
    "mother_income",
    "Single-Year IGE (common sample)"
)

common_multi_result, _ = estimate_ige(
    common_sample,
    "child_avg_income",
    "mother_avg_income",
    "Multi-Year IGE (common sample)"
)


# ============================================================
# Step 10: Working-age mother robustness check
# ============================================================

working_age_sample = multi_sample.loc[
    multi_sample["mother_age"].between(35, 64)
].copy()

print("\n=== WORKING-AGE MOTHERS ===")

working_age_result, _ = estimate_ige(
    working_age_sample,
    "child_avg_income",
    "mother_avg_income",
    "Multi-Year IGE (mother age 35-64)"
)


# ============================================================
# Step 11: Save baseline results
# ============================================================

results = pd.DataFrame([
    single_result,
    multi_result,
    common_single_result,
    common_multi_result,
    working_age_result
])

results.to_csv(
    TABLE_DIR / "ige_regression_results.csv",
    index=False
)

single_sample.to_csv(
    DATA_DIR / "ige_single_year_sample.csv",
    index=False
)

multi_sample.to_csv(
    DATA_DIR / "ige_multi_year_sample.csv",
    index=False
)

common_sample.to_csv(
    DATA_DIR / "ige_common_sample.csv",
    index=False
)


# ============================================================
# Step 12: Age-group life-cycle analysis
# ============================================================

print("\n=== AGE-GROUP ANALYSIS ===")

age_bins = [
    -np.inf, 45, 55, 65, np.inf
]

age_labels = [
    "Under 45",
    "45-54",
    "55-64",
    "65+"
]

common_sample["mother_age_group"] = pd.cut(
    common_sample["mother_age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)

age_rows = []

for i, label in enumerate(age_labels):

    group = common_sample.loc[
        common_sample["mother_age_group"] == label
    ]

    if len(group) < 30:
        print(
            f"{label}: n={len(group)}; "
            "skipped (fewer than 30 pairs)"
        )
        continue

    try:

        single_group, _ = estimate_ige(
            group,
            "child_income",
            "mother_income",
            f"Single-Year IGE (mother age {label})"
        )

        multi_group, _ = estimate_ige(
            group,
            "child_avg_income",
            "mother_avg_income",
            f"Multi-Year IGE (mother age {label})"
        )

        paired = bootstrap_difference(
            group,
            seed=SEED + i + 1
        )

    except ValueError as exc:
        print(
            f"{label}: skipped due to {exc}"
        )
        continue

    age_rows.append({
        "mother_age_group": label,
        "n": len(group),
        "single_beta": single_group["beta"],
        "single_se": single_group["robust_se"],
        "single_ci_lower": single_group["ci_lower"],
        "single_ci_upper": single_group["ci_upper"],
        "multi_beta": multi_group["beta"],
        "multi_se": multi_group["robust_se"],
        "multi_ci_lower": multi_group["ci_lower"],
        "multi_ci_upper": multi_group["ci_upper"],
        "difference_multi_minus_single": (
            paired["difference_multi_minus_single"]
        ),
        "difference_ci_lower": paired["ci_lower"],
        "difference_ci_upper": paired["ci_upper"],
        "difference_ci_excludes_zero": (
            paired["ci_excludes_zero"]
        )
    })


# ============================================================
# Step 13: Save age-group comparison
# ============================================================

age_results = pd.DataFrame(
    age_rows,
    columns=[
        "mother_age_group",
        "n",
        "single_beta",
        "single_se",
        "single_ci_lower",
        "single_ci_upper",
        "multi_beta",
        "multi_se",
        "multi_ci_lower",
        "multi_ci_upper",
        "difference_multi_minus_single",
        "difference_ci_lower",
        "difference_ci_upper",
        "difference_ci_excludes_zero"
    ]
)

age_results.to_csv(
    TABLE_DIR / "ige_age_group_comparison.csv",
    index=False
)

print("\nAge-group comparison:")

print(
    age_results.round(4).to_string(index=False)
)


# ============================================================
# Step 14: Bootstrap common-sample comparison
# ============================================================

print("\n=== PAIRED BOOTSTRAP: COMMON SAMPLE ===")

bootstrap = bootstrap_difference(
    common_sample
)

bootstrap_result = pd.DataFrame([{
    "comparison": (
        "Common sample: multi-year minus single-year"
    ),
    **bootstrap
}])

bootstrap_result.to_csv(
    TABLE_DIR / "ige_bootstrap_comparison.csv",
    index=False
)

print(
    bootstrap_result.round(4).to_string(index=False)
)


# ============================================================
# Step 15: Final results
# ============================================================

print("\n=== FINAL BASELINE RESULTS ===")

print(
    results[
        [
            "model",
            "n",
            "beta",
            "robust_se",
            "ci_lower",
            "ci_upper",
            "r_squared"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== IMPORTANT LIMITATIONS ===")

print(
    "- Multi-year averages use positive nominal earnings only."
)
print(
    "- Mother-child associations are descriptive, not causal."
)
print(
    "- Age differences suggest, but do not prove, life-cycle bias."
)
print(
    "- Bootstrap resamples pairs, not mother/family clusters."
)
print(
    "- Results are unweighted and do not account for PSID survey design."
)

print("\n04_ige_analysis.py completed successfully.")
