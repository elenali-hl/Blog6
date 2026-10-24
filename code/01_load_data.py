
"""
Blog Post 6: Matching Parents and Children with SQL in Python
01_load_data.py

Purpose:
1. Load original PSID individual-level data.
2. Load parent identification data (PID).
3. Load family income data.
4. Validate identifiers and data structure.
5. Save selected datasets for subsequent cleaning and SQL matching.

Required packages:
    pandas
    openpyxl

Run:
    python code/01_load_data.py
"""

from pathlib import Path
import re
import pandas as pd


# ============================================================
# Step 1: Set up project directories
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data-raw"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BLOG POST 6 - STEP 01: LOAD PSID DATA")
print("=" * 60)

print("Project directory:", BASE_DIR)


# ============================================================
# Step 2: Define helper functions
# ============================================================

def require_file(path):
    """Check whether a required input file exists."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required PSID input file is missing: {path}"
        )

    return path


def read_selected_fixed_width(
    txt_path,
    do_path,
    variables,
    rename=None
):
    """
    Read selected variables from PSID fixed-width files
    using column positions recorded in Stata .do files.
    """

    require_file(txt_path)
    require_file(do_path)

    dictionary = do_path.read_text(encoding="latin-1")

    positions = {
        name: (int(start) - 1, int(end))
        for name, start, end in re.findall(
            r"\b((?:ER|PID)\d+)\s+(\d+)\s*-\s*(\d+)",
            dictionary
        )
    }

    missing = [
        name for name in variables
        if name not in positions
    ]

    if missing:
        raise ValueError(
            f"Missing variables in {do_path.name}: {missing}"
        )

    data = pd.read_fwf(
        txt_path,
        colspecs=[positions[name] for name in variables],
        names=variables,
        header=None,
        dtype="Int64"
    )

    if rename:
        data = data.rename(columns=rename)

    return data


def validate_unique_keys(
    df,
    keys,
    label,
    allow_missing=False
):
    """Check missing and duplicated identifiers."""

    missing = df[keys].isna().any(axis=1)

    duplicate = (
        df.loc[~missing]
        .duplicated(subset=keys)
        .sum()
    )

    print(f"\n{label} validation:")
    print("Total rows:", len(df))
    print("Missing keys:", missing.sum())
    print("Duplicate keys:", duplicate)

    if duplicate > 0:
        raise ValueError(
            f"Duplicate identifiers detected in {label}"
        )

    if missing.any() and not allow_missing:
        raise ValueError(
            f"Missing identifiers detected in {label}"
        )


# ============================================================
# Step 3: Load PSID individual-level data
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: LOAD INDIVIDUAL DATA")
print("=" * 60)

individual_txt = (
    RAW_DIR / "ind2023er" / "IND2023ER.txt"
)

individual_do = (
    RAW_DIR / "ind2023er" / "IND2023ER.do"
)

individual_vars = [
    "ER30001", "ER30002",
    "ER34501", "ER34502", "ER34503", "ER34504",
    "ER34701", "ER34702", "ER34703", "ER34704",
    "ER34901", "ER34902", "ER34903", "ER34904",
    "ER35101", "ER35102", "ER35103", "ER35104"
]

individual = read_selected_fixed_width(
    individual_txt,
    individual_do,
    individual_vars
)

print("Individual data shape:", individual.shape)
print(individual.head())

validate_unique_keys(
    individual,
    ["ER30001", "ER30002"],
    "Individual data"
)


# ============================================================
# Step 4: Load PSID parent identification data
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: LOAD PARENT IDENTIFICATION DATA")
print("=" * 60)

pid_txt = RAW_DIR / "pid23" / "PID23.txt"
pid_do = RAW_DIR / "pid23" / "PID23.do"

pid_vars = [
    "PID2",
    "PID3",
    "PID4",
    "PID5"
]

pid_rename = {
    "PID2": "child_interview_id",
    "PID3": "child_person_number",
    "PID4": "mother_interview_id",
    "PID5": "mother_person_number"
}

pid = read_selected_fixed_width(
    pid_txt,
    pid_do,
    pid_vars,
    rename=pid_rename
)

print("PID data shape:", pid.shape)
print(pid.head())

# Identify records with positive child identifiers

valid_child = (
    pid["child_interview_id"].gt(0)
    &
    pid["child_person_number"].gt(0)
)

validate_unique_keys(
    pid.loc[valid_child],
    [
        "child_interview_id",
        "child_person_number"
    ],
    "PID valid child records"
)

# Check available mother identifiers

valid_mother = (
    pid["mother_interview_id"].gt(0)
    &
    pid["mother_person_number"].gt(0)
)

print(
    "\nRecords with valid mother identifiers:",
    valid_mother.sum()
)


# ============================================================
# Step 5: Load original PSID family income data
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: LOAD FAMILY INCOME DATA")
print("=" * 60)

income_file = (
    RAW_DIR / "J366546" / "J366546.xlsx"
)

require_file(income_file)

income = pd.read_excel(
    income_file,
    engine="openpyxl"
)

print("Income data shape:", income.shape)
print("Income columns:")
print(income.columns.tolist())

# Expected income variables by survey year

income_vars = {
    2017: [
        "ER66002",
        "ER71293",
        "ER71321"
    ],
    2019: [
        "ER72002",
        "ER77315",
        "ER77343"
    ],
    2021: [
        "ER78002",
        "ER81642",
        "ER81670"
    ],
    2023: [
        "ER82002",
        "ER85496",
        "ER85524"
    ]
}

required_income = [
    variable
    for columns in income_vars.values()
    for variable in columns
]

missing_income = sorted(
    set(required_income) - set(income.columns)
)

if missing_income:
    raise ValueError(
        f"Missing income variables: {missing_income}"
    )

print("\nAll required income variables are available.")


# ============================================================
# Step 6: Validate family identifiers by survey year
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: VALIDATE FAMILY INCOME DATA")
print("=" * 60)

for year, columns in income_vars.items():

    family_id_column = columns[0]

    family_ids = pd.to_numeric(
        income[family_id_column],
        errors="coerce"
    )

    valid_ids = family_ids.gt(0)

    n_valid = valid_ids.sum()

    n_duplicates = (
        family_ids.loc[valid_ids]
        .duplicated()
        .sum()
    )

    print(f"\nSurvey year: {year}")
    print("Family ID variable:", family_id_column)
    print("Valid family IDs:", n_valid)
    print("Duplicate positive IDs:", n_duplicates)

    if n_duplicates > 0:
        raise ValueError(
            f"Duplicate family IDs detected in {year}. "
            "Inspect the original data before proceeding."
        )


# ============================================================
# Step 7: Save selected datasets
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: SAVE SELECTED DATASETS")
print("=" * 60)

outputs = {
    "psid_individual_selected.csv": individual,
    "psid_pid_selected.csv": pid,
    "psid_family_income_selected.csv": income
}

for filename, dataframe in outputs.items():

    output_path = DATA_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path.name} "
        f"({len(dataframe):,} rows)"
    )


# ============================================================
# Step 8: Final summary
# ============================================================

print("\n" + "=" * 60)
print("STEP 01 COMPLETED")
print("=" * 60)

print("Individual records:", len(individual))
print("PID records:", len(pid))
print("Income records:", len(income))

print("\nOutput directory:", DATA_DIR)

print("\nGenerated files:")

for filename in outputs:
    print("-", filename)

print("\nAll required datasets have been saved.")
print("Next step: Run 02_clean_data.py")
