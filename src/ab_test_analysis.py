"""
ab_test_analysis.py
---------------------
Core statistical testing for each experiment using statsmodels.

For each experiment we report:
    control sample, treatment sample, control conversion, treatment
    conversion, absolute difference, relative difference, 95% confidence
    interval, p-value, effect size (Cohen's h), statistical
    interpretation, practical interpretation, limitations.

Null hypothesis (per experiment):   H0: p_treatment == p_control
Alternative hypothesis:             H1: p_treatment != p_control

We use a two-proportion z-test (statsmodels proportions_ztest) for
conversion, and an independent two-sample t-test (statsmodels / scipy)
for revenue and average order value.

We ALWAYS report both statistical significance (p-value) and practical
significance (the actual size of the effect, e.g. relative lift and a
rough revenue-impact estimate) - never "treatment won because p < 0.05"
alone.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy import stats

# A lift below this relative threshold is treated as practically
# negligible even if statistically significant - documented, adjustable
# assumption, not a universal rule.
PRACTICAL_SIGNIFICANCE_THRESHOLD = 0.10  # 10% relative lift


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def analyze_conversion(df: pd.DataFrame, experiment_id: str) -> dict:
    exp = df[df["experiment_id"] == experiment_id]
    control = exp[exp["variant"] == "control"]
    treatment = exp[exp["variant"] == "treatment"]

    n_control, n_treatment = len(control), len(treatment)
    conv_control = control["conversion"].sum()
    conv_treatment = treatment["conversion"].sum()

    rate_control = conv_control / n_control
    rate_treatment = conv_treatment / n_treatment

    count = np.array([conv_treatment, conv_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # 95% CI for the difference in proportions (normal approximation)
    diff = rate_treatment - rate_control
    se_diff = np.sqrt(
        rate_control * (1 - rate_control) / n_control
        + rate_treatment * (1 - rate_treatment) / n_treatment
    )
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    relative_lift = diff / rate_control if rate_control > 0 else np.nan
    effect_size = cohens_h(rate_treatment, rate_control)

    statistically_significant = p_value < 0.05
    practically_significant = abs(relative_lift) >= PRACTICAL_SIGNIFICANCE_THRESHOLD

    if statistically_significant and practically_significant:
        interpretation = (
            "Statistically significant AND practically meaningful: the observed "
            "lift is unlikely to be due to chance and is large enough to matter "
            "for the business."
        )
    elif statistically_significant and not practically_significant:
        interpretation = (
            "Statistically significant but NOT practically meaningful: with this "
            "sample size, even a small, low-impact difference can produce a low "
            "p-value. The effect is real but likely too small to justify the cost "
            "of shipping the change on its own."
        )
    elif not statistically_significant and practically_significant:
        interpretation = (
            "The observed lift looks practically large, but is NOT statistically "
            "significant - the sample size is too small to be confident this "
            "isn't just noise. Needs a larger sample before acting."
        )
    else:
        interpretation = (
            "Neither statistically nor practically significant: no reliable "
            "evidence of a difference between variants."
        )

    return {
        "experiment_id": experiment_id,
        "metric": "conversion",
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "control_rate": round(rate_control, 5),
        "treatment_rate": round(rate_treatment, 5),
        "absolute_difference": round(diff, 5),
        "relative_difference_pct": round(relative_lift * 100, 2),
        "ci_95_low": round(ci_low, 5),
        "ci_95_high": round(ci_high, 5),
        "z_statistic": round(z_stat, 3),
        "p_value": p_value,
        "effect_size_cohens_h": round(effect_size, 4),
        "statistically_significant": statistically_significant,
        "practically_significant": practically_significant,
        "interpretation": interpretation,
    }


def analyze_revenue(df: pd.DataFrame, experiment_id: str) -> dict:
    """Independent two-sample t-test on per-user revenue (includes zeros for
    non-converters, so this captures overall revenue-per-user impact)."""
    exp = df[df["experiment_id"] == experiment_id]
    control_rev = exp[exp["variant"] == "control"]["revenue"].fillna(0)
    treatment_rev = exp[exp["variant"] == "treatment"]["revenue"].fillna(0)

    t_stat, p_value = stats.ttest_ind(treatment_rev, control_rev, equal_var=False)

    mean_control = control_rev.mean()
    mean_treatment = treatment_rev.mean()
    diff = mean_treatment - mean_control
    relative_diff = diff / mean_control if mean_control > 0 else np.nan

    return {
        "experiment_id": experiment_id,
        "metric": "revenue_per_user",
        "mean_control": round(mean_control, 2),
        "mean_treatment": round(mean_treatment, 2),
        "absolute_difference": round(diff, 2),
        "relative_difference_pct": round(relative_diff * 100, 2),
        "t_statistic": round(t_stat, 3),
        "p_value": p_value,
        "statistically_significant": p_value < 0.05,
    }


def run_all_experiments(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for experiment_id in sorted(df["experiment_id"].unique()):
        results.append(analyze_conversion(df, experiment_id))
    return pd.DataFrame(results)


def run_all_revenue(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for experiment_id in sorted(df["experiment_id"].unique()):
        results.append(analyze_revenue(df, experiment_id))
    return pd.DataFrame(results)


def print_conversion_results(results: pd.DataFrame):
    for _, row in results.iterrows():
        print("=" * 75)
        print(f"Experiment: {row['experiment_id']}")
        print(f"  Control:   n={row['n_control']:,}   conversion={row['control_rate']:.4f}")
        print(f"  Treatment: n={row['n_treatment']:,}   conversion={row['treatment_rate']:.4f}")
        print(f"  Absolute diff: {row['absolute_difference']:.4f}   "
              f"Relative diff: {row['relative_difference_pct']:.2f}%")
        print(f"  95% CI (absolute diff): [{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}]")
        print(f"  z = {row['z_statistic']:.3f}   p = {row['p_value']:.5g}   "
              f"Cohen's h = {row['effect_size_cohens_h']:.4f}")
        print(f"  Statistically significant: {row['statistically_significant']}   "
              f"Practically significant: {row['practically_significant']}")
        print(f"  Interpretation: {row['interpretation']}")
    print("=" * 75)


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, RESULTS_PATH

    df = pd.read_csv(CLEAN_DATA_PATH)
    conv_results = run_all_experiments(df)
    rev_results = run_all_revenue(df)

    print_conversion_results(conv_results)
    print("\nRevenue-per-user results:")
    print(rev_results.to_string(index=False))

    conv_results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved conversion results -> {RESULTS_PATH}")
