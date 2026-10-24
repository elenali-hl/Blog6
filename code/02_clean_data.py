
# ============================================================
# Blog Post 6: Intergenerational Income Mobility Using PSID
# 02 - Clean Individual Data
#
# Purpose:
# 1. Load selected individual data
# 2. Clean age variables and missing-value codes
# 3. Reshape individual data from wide to long format
# 4. Identify current family members
# 5. Validate individual-year observations
# 6. Save cleaned datasets
# ============================================================


# ============================================================
# Step 1: Import packages
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Step 2: Set project directories
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\n=== PROJECT DIRECTORY ===")
print("Project:", BASE_DIR)
print("Data:", DATA_DIR)


# ============================================================
# Step 3: Load selected individual data
# ============================================================

individual_file = (
    DATA_DIR / "psid_individual_selected.csv"
)

if not individual_file.exists():
    raise FileNotFoundError(
        f"Missing file: {individual_file}. "
        "Run 01_load_data.py first."
    )

individual = pd.read_csv(individual_file)

print("\n=== ORIGINAL INDIVIDUAL DATA ===")
print("Shape:", individual.shape)

print("\nFirst five records:")
print(individual.head())


# ============================================================
# Step 4: Define variables for each survey year
# ============================================================

year_vars = {
    2017: [
        "ER34501",
        "ER34502",
        "ER34503",
        "ER34504"
    ],
    2019: [
        "ER34701",
        "ER34702",
        "ER34703",
        "ER34704"
    ],
    2021: [
        "ER34901",
        "ER34902",
        "ER34903",
        "ER34904"
    ],
    2023: [
        "ER35101",
        "ER35102",
        "ER35103",
        "ER35104"
    ]
}

# Verify all required variables exist
required_vars = [
    "ER30001",
    "ER30002"
]

for cols in year_vars.values():
    required_vars.extend(cols)

missing_vars = [
    var for var in required_vars
    if var not in individual.columns
]

if missing_vars:
    raise ValueError(
        f"Missing individual variables: {missing_vars}"
    )

print("\n=== VARIABLE VALIDATION ===")
print("All required variables are available.")


# ============================================================
# Step 5: Clean age variables
# ============================================================

# PSID uses 999 for unavailable age
age_vars = [
    cols[3]
    for cols in year_vars.values()
]

for var in age_vars:

    individual[var] = pd.to_numeric(
        individual[var],
        errors="coerce"
    )

    individual[var] = individual[var].replace(
        999,
        np.nan
    )

print("\n=== AGE CLEANING ===")

for year, cols in year_vars.items():

    age_var = cols[3]

    print(
        year,
        "Missing age:",
        individual[age_var].isna().sum()
    )


# ============================================================
# Step 6: Check individual identifiers
# ============================================================

print("\n=== INDIVIDUAL IDENTIFIERS ===")

print(
    "Total individual records:",
    len(individual)
)

duplicate_ids = individual.duplicated(
    subset=[
        "ER30001",
        "ER30002"
    ]
).sum()

print(
    "Duplicate individual IDs:",
    duplicate_ids
)

if duplicate_ids > 0:
    raise ValueError(
        "Duplicate individual identifiers found."
    )


# ============================================================
# Step 7: Convert wide data to long format
# ============================================================

long_data = []

for year, cols in year_vars.items():

    temp = individual[
        [
            "ER30001",
            "ER30002"
        ] + cols
    ].copy()

    temp.columns = [
        "interview_id",
        "person_number",
        "family_id",
        "sequence_number",
        "relationship",
        "age"
    ]

    temp["year"] = year

    long_data.append(temp)


individual_long = pd.concat(
    long_data,
    ignore_index=True
)

print("\n=== LONG FORMAT BEFORE FILTERING ===")

print(
    "Shape:",
    individual_long.shape
)

print("\nFirst five records:")
print(individual_long.head())


# ============================================================
# Step 8: Identify current family members
# ============================================================

# PSID sequence numbers 1-20 identify
# current family unit members.
#
# Sequence numbers outside this range may refer
# to people who moved out, died, or are otherwise
# not current members of the family unit.

individual_long = individual_long[
    individual_long["sequence_number"].between(
        1,
        20
    )
].copy()

print("\n=== CURRENT FAMILY MEMBERS ===")

print(
    "Shape:",
    individual_long.shape
)

print("\nObservations by year:")
print(
    individual_long["year"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Step 9: Validate family identifiers
# ============================================================

print("\n=== FAMILY IDENTIFIER CHECK ===")

missing_family_ids = (
    individual_long["family_id"].isna()
    | (individual_long["family_id"] <= 0)
).sum()

print(
    "Missing or invalid family IDs:",
    missing_family_ids
)

if missing_family_ids > 0:
    raise ValueError(
        "Current family members have invalid family IDs."
    )


# ============================================================
# Step 10: Check duplicate person-year records
# ============================================================

duplicate_person_year = (
    individual_long.duplicated(
        subset=[
            "interview_id",
            "person_number",
            "year"
        ]
    ).sum()
)

print("\n=== DUPLICATE CHECK ===")

print(
    "Duplicate person-year records:",
    duplicate_person_year
)

if duplicate_person_year > 0:
    raise ValueError(
        "Duplicate person-year records found."
    )


# ============================================================
# Step 11: Examine relationship codes
# ============================================================

print("\n=== RELATIONSHIP CODES ===")

print(
    individual_long["relationship"]
    .value_counts()
    .head(15)
)

# Reference Persons must have:
# relationship = 10
# sequence_number = 1

rp_records = individual_long[
    (individual_long["relationship"] == 10)
    & (individual_long["sequence_number"] == 1)
]

# Legal Spouses
spouse_records = individual_long[
    individual_long["relationship"] == 20
]

print("\nReference Persons by year:")
print(
    rp_records["year"]
    .value_counts()
    .sort_index()
)

print("\nLegal Spouses by year:")
print(
    spouse_records["year"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Step 12: Check age distribution
# ============================================================

print("\n=== AGE DISTRIBUTION ===")

print(
    individual_long["age"].describe()
)

print("\nAge summary by year:")

age_summary = (
    individual_long
    .groupby("year")
    .agg(
        total_records=("age", "size"),
        valid_ages=("age", "count"),
        mean_age=("age", "mean"),
        median_age=("age", "median"),
        min_age=("age", "min"),
        max_age=("age", "max")
    )
    .reset_index()
)

print(
    age_summary.to_string(index=False)
)


# ============================================================
# Step 13: Sort cleaned data
# ============================================================

individual_long = (
    individual_long
    .sort_values(
        [
            "year",
            "family_id",
            "sequence_number"
        ]
    )
    .reset_index(drop=True)
)

print("\n=== SORTED DATA ===")

print(
    individual_long.head(10)
)


# ============================================================
# Step 14: Save cleaned datasets
# ============================================================

individual.to_csv(
    DATA_DIR / "psid_individual_clean.csv",
    index=False
)

individual_long.to_csv(
    DATA_DIR / "psid_individual_long.csv",
    index=False
)

print("\n=== FILES SAVED ===")

print(
    "data/psid_individual_clean.csv"
)

print(
    "data/psid_individual_long.csv"
)


# ============================================================
# Step 15: Final validation
# ============================================================

print("\n=== FINAL DATA VALIDATION ===")

print(
    "Clean individual records:",
    len(individual)
)

print(
    "Long-format person-year records:",
    len(individual_long)
)

print(
    "Duplicate person-year records:",
    individual_long.duplicated(
        subset=[
            "interview_id",
            "person_number",
            "year"
        ]
    ).sum()
)

print("\nRecords by survey year:")

print(
    individual_long["year"]
    .value_counts()
    .sort_index()
)

print(
    "\n02_clean_data.py completed successfully."
)
