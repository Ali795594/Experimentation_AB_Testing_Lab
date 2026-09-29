"""
experiment_simulator.py
--------------------------
A simple, statistically correct EDUCATIONAL simulator for A/B test planning.
Not a web app - just functions you call from a script or notebook.

Two tools:
1. required_sample_size(): given a baseline conversion rate, a minimum
   detectable effect (relative), significance level, and desired power,
   returns the required sample size per arm (analytic formula via
   statsmodels' NormIndPower).
2. simulate_power_curve(): Monte Carlo simulation showing, for a range of
   sample sizes, what fraction of simulated experiments would correctly
   detect a true effect of a given size - a hands-on illustration of
   statistical power.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Returns the required sample size PER ARM to detect a relative lift of
    `minimum_detectable_effect_relative` over `baseline_rate`, at the given
    significance level and power, using Cohen's effect size for
    proportions.
    """
    treatment_rate = baseline_rate * (1 + minimum_detectable_effect_relative)
    h = 2 * np.arcsin(np.sqrt(treatment_rate)) - 2 * np.arcsin(np.sqrt(baseline_rate))

    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(h), alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))


def sample_size_table(
    baseline_rate: float,
    relative_effects=(0.02, 0.05, 0.10, 0.20, 0.30),
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    rows = []
    for effect in relative_effects:
        n = required_sample_size(baseline_rate, effect, alpha, power)
        rows.append(
            {
                "baseline_rate": baseline_rate,
                "relative_effect": effect,
                "target_rate": round(baseline_rate * (1 + effect), 4),
                "required_n_per_arm": n,
                "alpha": alpha,
                "power": power,
            }
        )
    return pd.DataFrame(rows)


def simulate_power_curve(
    baseline_rate: float,
    true_relative_effect: float,
    sample_sizes,
    n_simulations: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Monte Carlo simulation: for each candidate sample size (per arm), run
    n_simulations synthetic experiments with a KNOWN true effect and record
    the fraction of times a two-proportion z-test correctly detects a
    significant difference. That fraction IS the empirical statistical
    power for that sample size.
    """
    rng = np.random.default_rng(seed)
    treatment_rate = baseline_rate * (1 + true_relative_effect)

    rows = []
    for n in sample_sizes:
        detections = 0
        for _ in range(n_simulations):
            control_conversions = rng.binomial(n, baseline_rate)
            treatment_conversions = rng.binomial(n, treatment_rate)
            count = np.array([treatment_conversions, control_conversions])
            nobs = np.array([n, n])
            if count[0] == 0 and count[1] == 0:
                p_value = 1.0
            else:
                _, p_value = proportions_ztest(count, nobs)
            if p_value < alpha:
                detections += 1
        empirical_power = detections / n_simulations
        rows.append({"n_per_arm": n, "empirical_power": round(empirical_power, 3)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Sample size required to detect various effect sizes ===")
    print("(baseline conversion = 8%, alpha = 0.05, power = 0.80)\n")
    table = sample_size_table(baseline_rate=0.08)
    print(table.to_string(index=False))

    print("\n=== Empirical power curve (Monte Carlo simulation) ===")
    print("True effect: +20% relative lift over an 8% baseline\n")
    power_curve = simulate_power_curve(
        baseline_rate=0.08,
        true_relative_effect=0.20,
        sample_sizes=[500, 1000, 2000, 4000, 8000],
        n_simulations=1000,
    )
    print(power_curve.to_string(index=False))
    print(
        "\nInterpretation: as sample size increases, the empirical power "
        "(fraction of simulated experiments that correctly detect the true "
        "effect) rises toward the target of 80%, illustrating why "
        "under-powered experiments risk a false 'no effect' conclusion."
    )
