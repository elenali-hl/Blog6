
# ============================================================
# Blog Post 6: Intergenerational Income Mobility Using PSID
# 05 - Data Visualizations
#
# Purpose:
# 1. Visualize single-year mother-child earnings
# 2. Visualize multi-year average earnings
# 3. Compare IGE regression coefficients
# 4. Save publication-ready figures
# ============================================================


# ============================================================
# Step 1: Import packages
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Step 2: Set project directories
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)

DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\n=== PROJECT DIRECTORY ===")
print("Project:", BASE_DIR)


# ============================================================
# Step 3: Load analysis datasets
# ============================================================

single_sample = pd.read_csv(
    DATA_DIR / "ige_single_year_sample.csv"
)

multi_sample = pd.read_csv(
    DATA_DIR / "ige_multi_year_sample.csv"
)

common_sample = pd.read_csv(
    DATA_DIR / "ige_common_sample.csv"
)

results = pd.read_csv(
    TABLE_DIR / "ige_regression_results.csv"
)

print("\n=== DATA LOADED ===")

print("Single-year sample:", single_sample.shape)
print("Multi-year sample:", multi_sample.shape)
print("Common sample:", common_sample.shape)
print("Regression results:", results.shape)


# ============================================================
# Step 4: Define visualization settings
# ============================================================

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})


# ============================================================
# Step 5: Figure 1 - Single-Year IGE Scatter Plot
# ============================================================

# Use log earnings because the IGE model is log-log.

x_single = np.log(
    single_sample["mother_income"]
)

y_single = np.log(
    single_sample["child_income"]
)

# Estimate regression line using the same sample
beta_single, alpha_single = np.polyfit(
    x_single,
    y_single,
    1
)

x_line = np.linspace(
    x_single.min(),
    x_single.max(),
    100
)

y_line = (
    alpha_single
    + beta_single * x_line
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    x_single,
    y_single,
    alpha=0.25,
    s=18,
    color="#457B9D",
    edgecolors="none"
)

ax.plot(
    x_line,
    y_line,
    color="#D1495B",
    linewidth=2.5,
    label=f"OLS fit: beta = {beta_single:.3f}"
)

ax.set_title(
    "Mother-Child Earnings Association: Single Year",
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Log Mother's Labor Earnings (2022)"
)

ax.set_ylabel(
    "Log Adult Child's Labor Earnings (2022)"
)

ax.legend(frameon=False)

ax.grid(
    alpha=0.15
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "figure1_single_year_ige.png",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 1 saved.")
print("Single-year beta:", round(beta_single, 4))


# ============================================================
# Step 6: Figure 2 - Multi-Year IGE Scatter Plot
# ============================================================

x_multi = np.log(
    multi_sample["mother_avg_income"]
)

y_multi = np.log(
    multi_sample["child_avg_income"]
)

beta_multi, alpha_multi = np.polyfit(
    x_multi,
    y_multi,
    1
)

x_line = np.linspace(
    x_multi.min(),
    x_multi.max(),
    100
)

y_line = (
    alpha_multi
    + beta_multi * x_line
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    x_multi,
    y_multi,
    alpha=0.25,
    s=18,
    color="#2A9D8F",
    edgecolors="none"
)

ax.plot(
    x_line,
    y_line,
    color="#E76F51",
    linewidth=2.5,
    label=f"OLS fit: beta = {beta_multi:.3f}"
)

ax.set_title(
    "Mother-Child Earnings Association: Multi-Year Average",
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Log Mother's Average Labor Earnings"
)

ax.set_ylabel(
    "Log Adult Child's Average Labor Earnings"
)

ax.legend(frameon=False)

ax.grid(
    alpha=0.15
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "figure2_multi_year_ige.png",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 2 saved.")
print("Multi-year beta:", round(beta_multi, 4))


# ============================================================
# Step 7: Figure 3 - Compare IGE Coefficients
# ============================================================

plot_data = results.copy()

model_labels = {
    "Single-Year IGE (2022 income)":
        "Single year (2022)",

    "Multi-Year IGE (2+ positive waves)":
        "Multi-year (2+ waves)",

    "Single-Year IGE (common sample)":
        "Single year (common sample)",

    "Multi-Year IGE (common sample)":
        "Multi-year (common sample)",

    "Multi-Year IGE (mother age 35-64)":
        "Multi-year (mother age 35-64)"
}

plot_data["label"] = (
    plot_data["model"].map(model_labels)
)

if plot_data["label"].isna().any():
    raise ValueError(
        "Unexpected model name in regression results."
    )

plot_data = plot_data.sort_values(
    "beta"
).reset_index(drop=True)

y_positions = np.arange(len(plot_data))

lower_errors = (
    plot_data["beta"]
    - plot_data["ci_lower"]
)

upper_errors = (
    plot_data["ci_upper"]
    - plot_data["beta"]
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(
    plot_data["beta"],
    y_positions,
    xerr=[
        lower_errors,
        upper_errors
    ],
    fmt="o",
    color="#264653",
    ecolor="#457B9D",
    markersize=8,
    capsize=5,
    linewidth=2
)

ax.axvline(
    x=0,
    color="gray",
    linestyle="--",
    linewidth=1.5
)

ax.set_yticks(y_positions)

ax.set_yticklabels(
    plot_data["label"]
)

ax.set_xlabel(
    "Estimated Intergenerational Earnings Elasticity"
)

ax.set_title(
    "IGE Estimates Across Alternative Specifications",
    fontweight="bold",
    pad=15
)

ax.grid(
    axis="x",
    alpha=0.15
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "figure3_ige_comparison.png",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 3 saved.")


# ============================================================
# Step 8: Validate saved figures
# ============================================================

figure_files = [
    "figure1_single_year_ige.png",
    "figure2_multi_year_ige.png",
    "figure3_ige_comparison.png"
]

print("\n=== FIGURE VALIDATION ===")

for filename in figure_files:

    filepath = FIGURE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing figure: {filename}"
        )

    print(
        filename,
        "-",
        filepath.stat().st_size,
        "bytes"
    )


# ============================================================
# Step 9: Final summary
# ============================================================

print("\n=== VISUALIZATION SUMMARY ===")

print(
    "Single-year observations:",
    len(single_sample)
)

print(
    "Multi-year observations:",
    len(multi_sample)
)

print(
    "Common-sample observations:",
    len(common_sample)
)

print(
    "Figures created:",
    len(figure_files)
)

print(
    "\n05_visualizations.py completed successfully."
)


# ============================================================
# Figure 4: IGE by Mother's Age Group
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load age-group regression results
age_results = pd.read_csv(
    TABLE_DIR / "ige_age_group_comparison.csv"
)

print("\n=== FIGURE 4: IGE BY MOTHER'S AGE ===")
print(age_results)

# Step 2: Define age groups and positions
age_groups = ["45-54", "55-64", "65+"]

age_results = (
    age_results
    .set_index("mother_age_group")
    .loc[age_groups]
    .reset_index()
)

x = np.arange(len(age_groups))
offset = 0.12

# Step 3: Extract estimates and confidence intervals
single_beta = age_results["single_beta"].to_numpy()
multi_beta = age_results["multi_beta"].to_numpy()

single_lower = age_results["single_ci_lower"].to_numpy()
single_upper = age_results["single_ci_upper"].to_numpy()

multi_lower = age_results["multi_ci_lower"].to_numpy()
multi_upper = age_results["multi_ci_upper"].to_numpy()

single_error = np.vstack([
    single_beta - single_lower,
    single_upper - single_beta
])

multi_error = np.vstack([
    multi_beta - multi_lower,
    multi_upper - multi_beta
])

# Step 4: Plot estimates and 95% confidence intervals
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(
    x - offset,
    single_beta,
    yerr=single_error,
    fmt="o",
    capsize=5,
    markersize=8,
    linewidth=2,
    label="Single-year earnings"
)

ax.errorbar(
    x + offset,
    multi_beta,
    yerr=multi_error,
    fmt="s",
    capsize=5,
    markersize=8,
    linewidth=2,
    label="Multi-year average earnings"
)

ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1,
    color="gray"
)

# Step 5: Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(age_groups)

ax.set_xlabel("Mother's Age in 2023")
ax.set_ylabel("Intergenerational Earnings Elasticity (IGE)")

ax.set_title(
    "Intergenerational Earnings Elasticity by Mother's Age",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

# Display sample sizes below each group
for i, n in enumerate(age_results["n"]):
    ax.annotate(
        f"n = {int(n)}",
        xy=(i, 0),
        xytext=(0, -35),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="gray"
    )

fig.text(
    0.5,
    0.01,
    "Notes: Estimates use the same mother-child pairs within each age group. "
    "Error bars show 95% HC1 robust confidence intervals.",
    ha="center",
    fontsize=8
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

# Step 6: Save figure
output_file = (
    FIGURE_DIR / "figure4_ige_by_mother_age.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 4 saved:")
print(output_file)

print("\n05_visualizations.py completed successfully.")

