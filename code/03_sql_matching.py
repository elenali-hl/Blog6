
# ============================================================
# Blog Post 6: Intergenerational Income Mobility Using PSID
# 03 - SQL Matching and Income Availability
#
# Purpose:
# 1. Identify mother-child pairs using SQL
# 2. Reshape family income to long format
# 3. Match individuals to family income using SQL
# 4. Assign labor income to RP and legal spouse
# 5. Examine age and income availability
# 6. Save datasets for IGE analysis
# ============================================================


# ============================================================
# Step 1: Import packages and set directories
# ============================================================

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "tables"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== PROJECT DIRECTORY ===")
print("Project:", BASE_DIR)


# ============================================================
# Step 2: Load cleaned datasets
# ============================================================

individual = pd.read_csv(
    DATA_DIR / "psid_individual_clean.csv"
)

individual_long = pd.read_csv(
    DATA_DIR / "psid_individual_long.csv"
)

pid = pd.read_csv(
    DATA_DIR / "psid_pid_selected.csv"
)

income = pd.read_csv(
    DATA_DIR / "psid_family_income_selected.csv"
)

print("\n=== DATA LOADED ===")

print("Individual:", individual.shape)
print("Individual long:", individual_long.shape)
print("PID:", pid.shape)
print("Family income:", income.shape)


# ============================================================
# Step 3: Prepare valid mother-child identifiers
# ============================================================

# Zero is used for unavailable identifiers.
# Keep records with valid mother identifiers.

valid_mothers = pid[
    (pid["mother_interview_id"] > 0)
    & (pid["mother_person_number"] > 0)
    & (pid["child_interview_id"] > 0)
    & (pid["child_person_number"] > 0)
].copy()

print("\n=== VALID MOTHER IDENTIFIERS ===")
print("Records:", len(valid_mothers))


# ============================================================
# Step 4: Match mother-child pairs using SQL
# ============================================================

# Individual identifier table
individual_ids = individual[
    ["ER30001", "ER30002"]
].copy()

individual_ids.columns = [
    "interview_id",
    "person_number"
]

conn = sqlite3.connect(":memory:")

valid_mothers.to_sql(
    "parent_ids",
    conn,
    if_exists="replace",
    index=False
)

individual_ids.to_sql(
    "individual_ids",
    conn,
    if_exists="replace",
    index=False
)

# INNER JOIN requires both child and mother
# to be present in the individual identifier file.

query = """
SELECT DISTINCT
    p.child_interview_id,
    p.child_person_number,
    p.mother_interview_id,
    p.mother_person_number

FROM parent_ids AS p

INNER JOIN individual_ids AS c
    ON p.child_interview_id = c.interview_id
   AND p.child_person_number = c.person_number

INNER JOIN individual_ids AS m
    ON p.mother_interview_id = m.interview_id
   AND p.mother_person_number = m.person_number
"""

parent_child = pd.read_sql_query(query, conn)

conn.close()

print("\n=== SQL MOTHER-CHILD MATCHING ===")

print("Matched pairs:", len(parent_child))

print(
    "Duplicate pairs:",
    parent_child.duplicated().sum()
)

parent_child.to_csv(
    DATA_DIR / "parent_child_matches.csv",
    index=False
)


# ============================================================
# Step 5: Convert family income to long format
# ============================================================

income_vars = {
    2017: ["ER66002", "ER71293", "ER71321"],
    2019: ["ER72002", "ER77315", "ER77343"],
    2021: ["ER78002", "ER81642", "ER81670"],
    2023: ["ER82002", "ER85496", "ER85524"]
}

income_long_list = []

for year, cols in income_vars.items():

    temp = income[cols].copy()

    temp.columns = [
        "family_id",
        "rp_income",
        "spouse_income"
    ]

    temp["year"] = year

    income_long_list.append(temp)

income_long = pd.concat(
    income_long_list,
    ignore_index=True
)

income_long = income_long[
    income_long["family_id"].notna()
    & (income_long["family_id"] > 0)
].copy()

income_long["family_id"] = (
    income_long["family_id"].astype("int64")
)

print("\n=== FAMILY INCOME LONG FORMAT ===")

print("Shape:", income_long.shape)

print("\nRecords by year:")
print(
    income_long["year"]
    .value_counts()
    .sort_index()
)

print(
    "\nDuplicate family-year records:",
    income_long.duplicated(
        ["family_id", "year"]
    ).sum()
)


# ============================================================
# Step 6: Match individuals with family income using SQL
# ============================================================

conn = sqlite3.connect(":memory:")

individual_long.to_sql(
    "individuals",
    conn,
    if_exists="replace",
    index=False
)

income_long.to_sql(
    "family_income",
    conn,
    if_exists="replace",
    index=False
)

query = """
SELECT
    i.interview_id,
    i.person_number,
    i.family_id,
    i.year,
    i.age,
    i.sequence_number,
    i.relationship,
    f.rp_income,
    f.spouse_income

FROM individuals AS i

LEFT JOIN family_income AS f
    ON i.family_id = f.family_id
   AND i.year = f.year
"""

matched_income = pd.read_sql_query(query, conn)

conn.close()

print("\n=== SQL FAMILY INCOME MATCHING ===")

print("Shape:", matched_income.shape)

print(
    "Matched records:",
    matched_income["rp_income"].notna().sum()
)

print(
    "Unmatched records:",
    matched_income["rp_income"].isna().sum()
)

print(
    "Duplicate person-year records:",
    matched_income.duplicated(
        ["interview_id", "person_number", "year"]
    ).sum()
)


# ============================================================
# Step 7: Assign individual labor income
# ============================================================

matched_income["individual_income"] = np.nan

# Reference Person
rp_mask = (
    (matched_income["relationship"] == 10)
    & (matched_income["sequence_number"] == 1)
)

matched_income.loc[
    rp_mask, "individual_income"
] = matched_income.loc[
    rp_mask, "rp_income"
]

# Legal spouse
spouse_mask = matched_income["relationship"] == 20

matched_income.loc[
    spouse_mask, "individual_income"
] = matched_income.loc[
    spouse_mask, "spouse_income"
]

# Cohabiting partners (code 22) are not assigned
# spouse income until the official income
# variable definition is verified.

print("\n=== INDIVIDUAL LABOR INCOME ===")

print("Reference Persons:", rp_mask.sum())
print("Legal Spouses:", spouse_mask.sum())

print("\nIncome summary:")
print(
    matched_income["individual_income"].describe()
)

print(
    "\nMissing individual income:",
    matched_income["individual_income"].isna().sum()
)

print(
    "Zero income:",
    (matched_income["individual_income"] == 0).sum()
)

print(
    "Negative income:",
    (matched_income["individual_income"] < 0).sum()
)


# ============================================================
# Step 8: Examine income availability by age group
# ============================================================

income_check = matched_income.copy()

income_check["age_group"] = pd.cut(
    income_check["age"],
    bins=[0, 17, 24, 34, 44, 54, 64, 120],
    labels=[
        "Under 18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+"
    ]
)

age_income_summary = (
    income_check
    .groupby("age_group", observed=True)
    .agg(
        total_records=("individual_income", "size"),
        income_available=("individual_income", "count"),
        positive_income=(
            "individual_income",
            lambda x: (x > 0).sum()
        ),
        average_income=("individual_income", "mean")
    )
    .reset_index()
)

print("\n=== INCOME BY AGE GROUP ===")

print(
    age_income_summary.to_string(index=False)
)

age_income_summary.to_csv(
    RESULTS_DIR / "income_by_age_group.csv",
    index=False
)


# ============================================================
# Step 9: Match 2023 mother-child ages
# ============================================================

age_2023 = individual_long[
    individual_long["year"] == 2023
][
    ["interview_id", "person_number", "age"]
].copy()

pairs_age = parent_child.merge(
    age_2023.rename(columns={
        "interview_id": "child_interview_id",
        "person_number": "child_person_number",
        "age": "child_age"
    }),
    on=[
        "child_interview_id",
        "child_person_number"
    ],
    how="left",
    validate="many_to_one"
)

pairs_age = pairs_age.merge(
    age_2023.rename(columns={
        "interview_id": "mother_interview_id",
        "person_number": "mother_person_number",
        "age": "mother_age"
    }),
    on=[
        "mother_interview_id",
        "mother_person_number"
    ],
    how="left",
    validate="many_to_one"
)

print("\n=== MOTHER-CHILD AGE CHECK ===")

print("Total preliminary pairs:", len(pairs_age))

print(
    "Children with 2023 age:",
    pairs_age["child_age"].notna().sum()
)

print(
    "Mothers with 2023 age:",
    pairs_age["mother_age"].notna().sum()
)

print("\nChild age summary:")
print(pairs_age["child_age"].describe())

print("\nMother age summary:")
print(pairs_age["mother_age"].describe())


# ============================================================
# Step 10: Identify adult mother-child pairs
# ============================================================

adult_pairs = pairs_age[
    (pairs_age["child_age"] >= 25)
    & (pairs_age["mother_age"].notna())
].copy()

print("\n=== ADULT MOTHER-CHILD PAIRS ===")

print("Adult pairs:", len(adult_pairs))

print("\nChild age summary:")
print(adult_pairs["child_age"].describe())

print("\nMother age summary:")
print(adult_pairs["mother_age"].describe())

print(
    "\nMothers aged 35-64:",
    adult_pairs["mother_age"]
    .between(35, 64)
    .sum()
)


# ============================================================
# Step 11: Match mother-child income for 2023
# ============================================================

income_2023 = matched_income[
    matched_income["year"] == 2023
][
    ["interview_id", "person_number", "individual_income"]
].copy()

pairs_income = adult_pairs.merge(
    income_2023.rename(columns={
        "interview_id": "child_interview_id",
        "person_number": "child_person_number",
        "individual_income": "child_income"
    }),
    on=[
        "child_interview_id",
        "child_person_number"
    ],
    how="left",
    validate="many_to_one"
)

pairs_income = pairs_income.merge(
    income_2023.rename(columns={
        "interview_id": "mother_interview_id",
        "person_number": "mother_person_number",
        "individual_income": "mother_income"
    }),
    on=[
        "mother_interview_id",
        "mother_person_number"
    ],
    how="left",
    validate="many_to_one"
)

print("\n=== MOTHER-CHILD INCOME AVAILABILITY ===")

print("Adult pairs:", len(pairs_income))

print(
    "Children with assigned income:",
    pairs_income["child_income"].notna().sum()
)

print(
    "Mothers with assigned income:",
    pairs_income["mother_income"].notna().sum()
)

print(
    "Both with assigned income:",
    (
        pairs_income["child_income"].notna()
        & pairs_income["mother_income"].notna()
    ).sum()
)

print(
    "Both with positive income:",
    (
        (pairs_income["child_income"] > 0)
        & (pairs_income["mother_income"] > 0)
    ).sum()
)


# ============================================================
# Step 12: Examine positive-income sample
# ============================================================

positive_pairs = pairs_income[
    (pairs_income["child_income"] > 0)
    & (pairs_income["mother_income"] > 0)
].copy()

print("\n=== POSITIVE-INCOME SAMPLE ===")

print("Total pairs:", len(positive_pairs))

print("\nChild age summary:")
print(positive_pairs["child_age"].describe())

print("\nMother age summary:")
print(positive_pairs["mother_age"].describe())

print("\nMother age groups:")
print(
    pd.cut(
        positive_pairs["mother_age"],
        bins=[0, 44, 54, 64, 120],
        labels=[
            "Under 45",
            "45-54",
            "55-64",
            "65+"
        ]
    ).value_counts().sort_index()
)

print("\nChild income summary:")
print(positive_pairs["child_income"].describe())

print("\nMother income summary:")
print(positive_pairs["mother_income"].describe())


# ============================================================
# Step 13: Save analysis datasets
# ============================================================

income_long.to_csv(
    DATA_DIR / "psid_family_income_long.csv",
    index=False
)

matched_income.to_csv(
    DATA_DIR / "psid_individual_income_long.csv",
    index=False
)

pairs_age.to_csv(
    DATA_DIR / "mother_child_pairs_age.csv",
    index=False
)

pairs_income.to_csv(
    DATA_DIR / "mother_child_income_2023.csv",
    index=False
)

positive_pairs.to_csv(
    DATA_DIR / "mother_child_positive_income_2023.csv",
    index=False
)

print("\n=== FILES SAVED ===")

print("data/parent_child_matches.csv")
print("data/psid_family_income_long.csv")
print("data/psid_individual_income_long.csv")
print("data/mother_child_pairs_age.csv")
print("data/mother_child_income_2023.csv")
print("data/mother_child_positive_income_2023.csv")


# ============================================================
# Step 14: Final validation
# ============================================================

print("\n=== FINAL VALIDATION ===")

checks = {
    "Mother-child pairs": len(parent_child),
    "Family income records": len(income_long),
    "Matched person-years": len(matched_income),
    "Adult mother-child pairs": len(adult_pairs),
    "Positive-income pairs": len(positive_pairs)
}

for name, value in checks.items():
    print(f"{name}: {value}")

print("\n03_sql_matching.py completed successfully.")
