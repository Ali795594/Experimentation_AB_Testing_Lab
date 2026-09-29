"""
segment_analysis.py
----------------------
Breaks down each experiment's conversion results by device, region, and
customer_type.

IMPORTANT CAUTIONS (enforced in the output, not just mentioned):
- Segment cuts have much smaller sample sizes than the overall experiment,
  so confidence intervals are wider and results are noisier. We flag any
  segment with fewer than MIN_SEGMENT_N users per arm as "low sample -
  interpret with caution".
- Running many segment cuts is a form of multiple comparisons: if we test
  enough slices, some will look "significant" by chance alone. We do NOT
  apply a formal correction (e.g. Bonferroni) by default, but we explicitly
  label this analysis as EXPLORATORY and print a warning about the
  multiple-comparisons risk, consistent with treating subgroup findings as
  hypothesis-generating, not confirmatory.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

MIN_SEGMENT_N = 200  # per arm, below this we flag low sample size


def analyze_segment_cut(df: pd.DataFrame, experiment_id: str, segment_col: str) -> pd.DataFrame:
    exp = df[df["experiment_id"] == experiment_id]
    rows = []
    for segment_value, seg_df in exp.groupby(segment_col, observed=True):
        control = seg_df[seg_df["variant"] == "control"]
        treatment = seg_df[seg_df["variant"] == "treatment"]
        n_c, n_t = len(control), len(treatment)
        if n_c == 0 or n_t == 0:
            continue

        rate_c = control["conversion"].sum() / n_c
        rate_t = treatment["conversion"].sum() / n_t

        count = np.array([treatment["conversion"].sum(), control["conversion"].sum()])
        nobs = np.array([n_t, n_c])
        try:
            z_stat, p_value = proportions_ztest(count, nobs)
        except Exception:
            z_stat, p_value = np.nan, np.nan

        low_sample = (n_c < MIN_SEGMENT_N) or (n_t < MIN_SEGMENT_N)

        rows.append(
            {
                "experiment_id": experiment_id,
                "segment_type": segment_col,
                "segment_value": segment_value,
                "n_control": n_c,
                "n_treatment": n_t,
                "control_rate": round(rate_c, 4),
                "treatment_rate": round(rate_t, 4),
                "relative_diff_pct": round(
                    ((rate_t - rate_c) / rate_c * 100) if rate_c > 0 else np.nan, 2
                ),
                "p_value": p_value,
                "significant_uncorrected": (p_value < 0.05) if not np.isnan(p_value) else False,
                "low_sample_warning": low_sample,
            }
        )
    return pd.DataFrame(rows)


def full_segment_report(df: pd.DataFrame) -> pd.DataFrame:
    segment_cols = ["device", "country_or_region", "customer_type"]
    all_results = []
    for experiment_id in sorted(df["experiment_id"].unique()):
        for col in segment_cols:
            all_results.append(analyze_segment_cut(df, experiment_id, col))
    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, SEGMENT_RESULTS_PATH

    df = pd.read_csv(CLEAN_DATA_PATH)
    report = full_segment_report(df)

    n_tests = len(report)
    n_sig = report["significant_uncorrected"].sum()
    print(f"Ran {n_tests} exploratory segment-level tests; {n_sig} came back "
          f"'significant' at p<0.05 WITHOUT multiple-comparisons correction.")
    print("This is exploratory: with many cuts, some 'significant' results are "
          "expected by chance alone. Treat subgroup findings as hypotheses to "
          "confirm in a dedicated follow-up test, not standalone conclusions.\n")

    print(report.to_string(index=False))

    n_low_sample = report["low_sample_warning"].sum()
    print(f"\n{n_low_sample} of {n_tests} segment cuts have fewer than "
          f"{MIN_SEGMENT_N} users in at least one arm - interpret those with extra caution.")

    report.to_csv(SEGMENT_RESULTS_PATH, index=False)
    print(f"\nSaved -> {SEGMENT_RESULTS_PATH}")
