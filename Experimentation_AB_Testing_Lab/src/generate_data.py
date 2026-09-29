"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC A/B testing dataset covering three
e-commerce experiments:

    1. checkout_redesign     - a meaningful, statistically AND practically
                                significant improvement in conversion
    2. cta_redesign          - a small effect that is NOT statistically
                                significant (noise-level difference)
    3. product_page_redesign - a statistically significant lift that is
                                too small to matter practically (illustrates
                                the difference between statistical and
                                practical significance)

Run:
    python src/generate_data.py

Output:
    data/raw/ab_test_users.csv

Deterministic (fixed random seed). Deliberately contains data-quality
issues that a real analytics pipeline must catch: duplicate users, users
contaminated across variants, missing outcomes, impossible values, and
sample imbalance in one experiment.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 11
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ab_test_users.csv")

DEVICES = ["desktop", "mobile", "tablet"]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
CUSTOMER_TYPES = ["new", "returning"]

EXPERIMENTS = {
    "checkout_redesign": {
        "start": datetime(2023, 3, 1),
        "days": 21,
        "n_per_arm": 9000,
        "split": (0.5, 0.5),
        "control_conversion": 0.070,
        "treatment_conversion": 0.086,     # meaningful relative lift (~23%)
        "control_aov": 62.0,
        "treatment_aov": 64.0,
        "aov_sd": 18.0,
    },
    "cta_redesign": {
        "start": datetime(2023, 5, 10),
        "days": 14,
        "n_per_arm": 2200,
        "split": (0.6, 0.4),               # deliberate allocation imbalance
        "control_conversion": 0.048,
        "treatment_conversion": 0.051,     # small, likely-noise effect
        "control_aov": 45.0,
        "treatment_aov": 45.5,
        "aov_sd": 14.0,
    },
    "product_page_redesign": {
        "start": datetime(2023, 8, 1),
        "days": 30,
        "n_per_arm": 45000,
        "split": (0.5, 0.5),
        "control_conversion": 0.101,
        "treatment_conversion": 0.1025,    # tiny lift, but huge N -> significant
        "control_aov": 55.0,
        "treatment_aov": 55.2,
        "aov_sd": 16.0,
    },
}


def simulate_experiment(name, cfg, rng):
    rows = []
    n_control = int(cfg["n_per_arm"] * cfg["split"][0] / 0.5)
    n_treatment = int(cfg["n_per_arm"] * cfg["split"][1] / 0.5)

    for variant, n_users, conv_rate, aov_mean in [
        ("control", n_control, cfg["control_conversion"], cfg["control_aov"]),
        ("treatment", n_treatment, cfg["treatment_conversion"], cfg["treatment_aov"]),
    ]:
        user_ids = [f"{name[:3].upper()}-U{rng.integers(10**7, 10**8-1)}" for _ in range(n_users)]
        devices = rng.choice(DEVICES, size=n_users, p=[0.45, 0.45, 0.10])
        regions = rng.choice(REGIONS, size=n_users, p=[0.35, 0.30, 0.25, 0.10])
        customer_types = rng.choice(CUSTOMER_TYPES, size=n_users, p=[0.55, 0.45])
        exp_days = rng.integers(0, cfg["days"], size=n_users)

        # Each user has 1 session in this simplified experiment (one exposure)
        sessions = np.ones(n_users, dtype=int)
        converted = rng.random(n_users) < conv_rate
        purchases = converted.astype(int)

        revenue = np.zeros(n_users)
        purchasing = converted
        n_purchasing = purchasing.sum()
        if n_purchasing > 0:
            revenue[purchasing] = np.clip(
                rng.normal(aov_mean, cfg["aov_sd"], size=n_purchasing), 5, None
            )
        revenue = np.round(revenue, 2)
        aov = np.where(purchases > 0, revenue / np.maximum(purchases, 1), np.nan)

        for i in range(n_users):
            rows.append(
                {
                    "experiment_id": name,
                    "user_id": user_ids[i],
                    "experiment_date": (cfg["start"] + timedelta(days=int(exp_days[i]))).strftime("%Y-%m-%d"),
                    "variant": variant,
                    "device": devices[i],
                    "country_or_region": regions[i],
                    "customer_type": customer_types[i],
                    "sessions": int(sessions[i]),
                    "purchases": int(purchases[i]),
                    "revenue": float(revenue[i]),
                    "conversion": int(converted[i]),
                    "average_order_value": round(float(aov[i]), 2) if not np.isnan(aov[i]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def inject_data_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()

    # 1) Duplicate rows (pipeline re-ingestion issue) - ~0.3%
    dup_sample = df.sample(frac=0.003, random_state=rng.integers(1, 10**6))
    df = pd.concat([df, dup_sample], ignore_index=True)

    # 2) Contamination: a handful of users appear in BOTH variants of the
    #    SAME experiment (e.g. logged out/in on a different device)
    for exp_name in df["experiment_id"].unique():
        exp_mask = df["experiment_id"] == exp_name
        exp_df = df[exp_mask]
        control_users = exp_df[exp_df["variant"] == "control"]["user_id"]
        n_contam = max(1, int(0.001 * len(exp_df)))
        contam_users = control_users.sample(min(n_contam, len(control_users)), random_state=42).tolist()
        contam_rows = exp_df[exp_df["user_id"].isin(contam_users)].copy()
        contam_rows["variant"] = "treatment"
        df = pd.concat([df, contam_rows], ignore_index=True)

    # 3) Missing outcome values
    for col, frac in [("conversion", 0.0), ("revenue", 0.003), ("device", 0.01), ("country_or_region", 0.008)]:
        if frac > 0:
            mask = rng.random(len(df)) < frac
            df.loc[mask, col] = np.nan

    # 4) Impossible values: negative revenue / purchases > sessions
    err_mask = rng.random(len(df)) < 0.001
    df.loc[err_mask, "revenue"] = -df.loc[err_mask, "revenue"].abs() - 1
    err_mask2 = rng.random(len(df)) < 0.0008
    df.loc[err_mask2, "purchases"] = df.loc[err_mask2, "sessions"] + rng.integers(1, 3, size=err_mask2.sum())

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_dfs = [simulate_experiment(name, cfg, rng) for name, cfg in EXPERIMENTS.items()]
    df = pd.concat(all_dfs, ignore_index=True)
    df = inject_data_quality_issues(df, rng)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} user rows across {df['experiment_id'].nunique()} experiments -> {OUTPUT_PATH}")
    print(df.groupby(["experiment_id", "variant"]).size())
    print("\nMissing values:\n", df.isna().sum())


if __name__ == "__main__":
    main()
