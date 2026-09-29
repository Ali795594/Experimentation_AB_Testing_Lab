"""
data_cleaning.py
-----------------
Cleans the raw A/B test dataset based on the issues found by
data_quality.py.

Documented decisions:
- Exact duplicate rows dropped.
- Contaminated users (appearing in both variants of the same experiment)
  are DROPPED ENTIRELY from that experiment - we cannot know which variant
  actually influenced their behavior, so including them in either arm
  would bias the comparison.
- Rows with impossible values (negative revenue, purchases > sessions) are
  dropped rather than "fixed" - these look like tracking/logging bugs, not
  simple sign errors, so guessing the correct value is not defensible.
- Missing `device` / `country_or_region` -> labeled "Unknown" (kept).
- Missing `revenue` for a converted user -> dropped (can't compute revenue
  impact without it). Missing `average_order_value` for NON-purchasers is
  expected (NaN by construction, not a data-quality issue) and is left as
  NaN.
"""

import numpy as np
import pandas as pd

from data_quality import check_contamination


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df.drop_duplicates()
    n_dupes_removed = before - len(df)

    # Remove contaminated users entirely (from all their rows in that experiment)
    contaminated = check_contamination(df)
    contam_keys = set(zip(contaminated["experiment_id"], contaminated["user_id"]))
    if contam_keys:
        mask = df.apply(
            lambda r: (r["experiment_id"], r["user_id"]) in contam_keys, axis=1
        )
        n_contam_rows = int(mask.sum())
        df = df[~mask]
    else:
        n_contam_rows = 0

    # Drop impossible values
    before_impossible = len(df)
    df = df[(df["revenue"].isna()) | (df["revenue"] >= 0)]
    df = df[df["purchases"] <= df["sessions"]]
    n_impossible_removed = before_impossible - len(df)

    # Drop rows where a converted user is missing revenue (can't be fixed)
    before_missing_rev = len(df)
    bad_revenue = (df["conversion"] == 1) & (df["revenue"].isna())
    df = df[~bad_revenue]
    n_missing_revenue_removed = before_missing_rev - len(df)

    # Fill non-essential categorical missingness
    df["device"] = df["device"].fillna("Unknown")
    df["country_or_region"] = df["country_or_region"].fillna("Unknown")

    # Non-purchaser NaN AOV is expected; leave as-is but explicitly flag it
    df["average_order_value"] = df["average_order_value"].where(
        df["purchases"] > 0, other=np.nan
    )

    df = df.reset_index(drop=True)

    print(f"Removed {n_dupes_removed} exact duplicate rows.")
    print(f"Removed {n_contam_rows} rows from {len(contam_keys)} contaminated users.")
    print(f"Removed {n_impossible_removed} rows with impossible values.")
    print(f"Removed {n_missing_revenue_removed} converted rows missing revenue.")
    print(f"Rows after cleaning: {len(df):,} (from {before:,})")

    return df


def build_sql_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe renamed/ordered to match sql/01_schema.sql."""
    sql_df = df.rename(columns={"experiment_date": "exp_date"})[
        [
            "experiment_id", "user_id", "exp_date", "variant", "device",
            "country_or_region", "customer_type", "sessions", "purchases",
            "revenue", "conversion", "average_order_value",
        ]
    ]
    return sql_df


if __name__ == "__main__":
    from utils import RAW_DATA_PATH, CLEAN_DATA_PATH, BASE_DIR, ensure_dirs
    import os

    ensure_dirs()
    raw = pd.read_csv(RAW_DATA_PATH)
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_DATA_PATH}")

    sql_export_path = os.path.join(BASE_DIR, "data", "processed", "ab_test_users_sql.csv")
    build_sql_export(cleaned).to_csv(sql_export_path, index=False)
    print(f"Saved SQL-ready export -> {sql_export_path}")
