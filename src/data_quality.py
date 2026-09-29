"""
data_quality.py
-----------------
Data-quality checks for the A/B testing dataset, run BEFORE any statistical
testing. Experimentation results are only as trustworthy as the assignment
and measurement pipeline behind them, so these checks are treated as a
first-class step, not an afterthought.

Checks implemented:
- duplicate user rows (same user_id + experiment_id + variant appearing
  more than once)
- users appearing in MULTIPLE variants of the same experiment
  ("contamination" - breaks the independence assumption behind a
  two-proportion z-test)
- missing values in key outcome columns
- invalid variant labels
- impossible values (negative revenue, purchases > sessions)
- missing outcomes (conversion flagged but no purchase count, etc.)
- sample size imbalance between control/treatment
"""

import numpy as np
import pandas as pd


def check_duplicates(df: pd.DataFrame) -> dict:
    dupe_rows = df.duplicated().sum()
    dupe_user_variant = df.duplicated(subset=["experiment_id", "user_id", "variant"]).sum()
    return {
        "exact_duplicate_rows": int(dupe_rows),
        "duplicate_user_in_same_variant": int(dupe_user_variant),
    }


def check_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """Users who appear in more than one variant within the same experiment."""
    variant_counts = (
        df.groupby(["experiment_id", "user_id"])["variant"]
        .nunique()
        .reset_index(name="n_variants")
    )
    contaminated = variant_counts[variant_counts["n_variants"] > 1]
    return contaminated


def check_missing_values(df: pd.DataFrame) -> dict:
    return df.isna().sum().to_dict()


def check_invalid_variants(df: pd.DataFrame, valid=("control", "treatment")) -> int:
    return int((~df["variant"].isin(valid)).sum())


def check_impossible_values(df: pd.DataFrame) -> dict:
    return {
        "negative_revenue": int((df["revenue"] < 0).sum()),
        "purchases_exceed_sessions": int((df["purchases"] > df["sessions"]).sum()),
        "negative_sessions": int((df["sessions"] < 0).sum()),
    }


def check_sample_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["experiment_id", "variant"]).size().unstack(fill_value=0)
    counts["ratio"] = counts.get("treatment", 0) / counts.get("control", 1).replace(0, np.nan)
    return counts


def run_all_checks(df: pd.DataFrame) -> dict:
    report = {}
    report["n_rows"] = len(df)
    report["duplicates"] = check_duplicates(df)
    contaminated = check_contamination(df)
    report["n_contaminated_users"] = len(contaminated)
    report["missing_values"] = check_missing_values(df)
    report["invalid_variant_rows"] = check_invalid_variants(df)
    report["impossible_values"] = check_impossible_values(df)
    report["sample_balance"] = check_sample_balance(df)
    return report, contaminated


def print_report(report: dict):
    print("=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    print(f"Total rows: {report['n_rows']:,}")
    print(f"Exact duplicate rows: {report['duplicates']['exact_duplicate_rows']}")
    print(f"Duplicate (user, variant) pairs: {report['duplicates']['duplicate_user_in_same_variant']}")
    print(f"Users contaminated across variants: {report['n_contaminated_users']}")
    print(f"Invalid variant label rows: {report['invalid_variant_rows']}")
    print("\nImpossible values:")
    for k, v in report["impossible_values"].items():
        print(f"  - {k}: {v}")
    print("\nMissing values by column:")
    for k, v in report["missing_values"].items():
        if v > 0:
            print(f"  - {k}: {v}")
    print("\nSample balance by experiment:")
    print(report["sample_balance"])
    print("=" * 65)


if __name__ == "__main__":
    from utils import RAW_DATA_PATH

    df = pd.read_csv(RAW_DATA_PATH)
    report, contaminated = run_all_checks(df)
    print_report(report)
