"""
generate_dashboard.py
-----------------------
Builds a SINGLE self-contained HTML file with tabbed dashboards summarizing
all three A/B test experiments - just a web browser needed.

Run:
    python src/generate_dashboard.py

Output:
    dashboard/ab_testing_dashboard.html
"""

import os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def kpi_card(label, value, sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {sub_html}
    </div>"""


def img_block(title, b64):
    return f"""
    <div class="chart-block">
        <h3>{title}</h3>
        <img src="data:image/png;base64,{b64}" />
    </div>"""


def df_to_html_table(df, float_cols=None, pct_cols=None):
    float_cols = float_cols or []
    pct_cols = pct_cols or []
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:,.4f}")
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.2f}%")
    return df.to_html(classes="data-table", index=False, border=0)


# ---------------------------------------------------------------- charts --

def chart_sample_sizes(df):
    counts = df.groupby(["experiment_id", "variant"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Users")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_conversion_by_variant(df):
    conv = df.groupby(["experiment_id", "variant"])["conversion"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    conv.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Conversion Rate")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_experiment_ci(conv_results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(conv_results))
    diffs = conv_results["absolute_difference"].values
    ci_low = conv_results["ci_95_low"].values
    ci_high = conv_results["ci_95_high"].values
    colors = ["#55A868" if s else "#C44E52" for s in conv_results["statistically_significant"]]
    ax.errorbar(diffs, y_pos, xerr=[diffs - ci_low, ci_high - diffs], fmt="o", color="black", ecolor="gray", capsize=4)
    for i, c in enumerate(colors):
        ax.scatter(diffs[i], y_pos[i], color=c, zorder=5, s=60)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conv_results["experiment_id"])
    ax.set_xlabel("Absolute difference in conversion (treatment - control), 95% CI")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_revenue_boxplot(df):
    converters = df[df["conversion"] == 1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=converters, x="experiment_id", y="revenue", hue="variant", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_device_breakdown(df):
    counts = df.groupby(["experiment_id", "device"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Users")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_segment_conversion(segment_report, experiment_id, segment_type):
    sub = segment_report[
        (segment_report["experiment_id"] == experiment_id) & (segment_report["segment_type"] == segment_type)
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(sub))
    ax.bar(x - width / 2, sub["control_rate"], width, label="control", color="#4C72B0")
    ax.bar(x + width / 2, sub["treatment_rate"], width, label="treatment", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["segment_value"], rotation=15, ha="right")
    ax.set_ylabel("Conversion Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_power_curve(power_curve):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(power_curve["n_per_arm"], power_curve["empirical_power"], marker="o", color="#4C72B0")
    ax.axhline(0.8, color="gray", linestyle="--", label="80% power target")
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Empirical statistical power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


# ------------------------------------------------------------------ HTML --

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Experimentation & A/B Testing Dashboard</title>
<style>
  :root {{ --accent:#4C72B0; --bg:#f4f6f9; --card:#ffffff; --text:#222; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:var(--bg); margin:0; color:var(--text); }}
  header {{ background:linear-gradient(135deg,#2c3e63,#4C72B0); color:white; padding:28px 32px; }}
  header h1 {{ margin:0; font-size:1.6rem; }}
  header p {{ margin:6px 0 0; opacity:0.85; font-size:0.95rem; }}
  .tabs {{ display:flex; gap:4px; padding:0 24px; background:#e7eaf0; border-bottom:1px solid #d3d8e2; flex-wrap: wrap;}}
  .tab-btn {{ padding:12px 20px; cursor:pointer; border:none; background:transparent; font-size:0.95rem; font-weight:600; color:#555; border-bottom:3px solid transparent; }}
  .tab-btn.active {{ color:var(--accent); border-bottom:3px solid var(--accent); }}
  .tab-content {{ display:none; padding:24px 32px 48px; max-width:1400px; margin:0 auto; }}
  .tab-content.active {{ display:block; }}
  .kpi-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .kpi-card {{ background:var(--card); border-radius:10px; padding:18px 24px; box-shadow:0 1px 4px rgba(0,0,0,0.08); min-width:200px; flex:1; }}
  .kpi-value {{ font-size:1.5rem; font-weight:700; color:var(--accent); }}
  .kpi-label {{ font-size:0.85rem; color:#777; margin-top:4px; }}
  .kpi-sub {{ font-size:0.78rem; color:#999; margin-top:2px; }}
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .chart-block {{ background:var(--card); border-radius:10px; padding:16px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .chart-block.full {{ grid-column:1 / -1; }}
  .chart-block h3 {{ margin:0 0 10px; font-size:1rem; color:#444; }}
  .chart-block img {{ width:100%; height:auto; border-radius:6px; }}
  table.data-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  table.data-table th, table.data-table td {{ padding:6px 10px; border-bottom:1px solid #eee; text-align:right; }}
  table.data-table th:first-child, table.data-table td:first-child {{ text-align:left; }}
  .note {{ background:#fff8e1; border-left:4px solid #DD8452; padding:10px 16px; border-radius:6px; font-size:0.85rem; margin-bottom:16px; }}
  footer {{ text-align:center; padding:20px; color:#999; font-size:0.8rem; }}
  @media (max-width:900px) {{ .chart-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Experimentation & A/B Testing Dashboard</h1>
  <p>Generated locally from data/processed/ Open this file directly in any browser.</p>
</header>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab(0)">Experiment Overview</button>
  <button class="tab-btn" onclick="showTab(1)">Statistical Results</button>
  <button class="tab-btn" onclick="showTab(2)">Segment Analysis</button>
  <button class="tab-btn" onclick="showTab(3)">Sample Size / Power</button>
</div>

<div class="tab-content active" id="tab-0">
  <div class="kpi-row">{kpi_row_1}</div>
  <div class="chart-grid">
    {chart_sample_sizes}
    {chart_conversion_variant}
  </div>
</div>

<div class="tab-content" id="tab-1">
  <div class="note">Green marker = statistically significant (p &lt; 0.05); red marker = not significant. Statistical significance alone does not imply practical significance - see the results table below.</div>
  <div class="chart-grid">
    <div class="chart-block full">{chart_ci}</div>
    <div class="chart-block full">{results_table}</div>
    <div class="chart-block full">{chart_revenue_box}</div>
  </div>
</div>

<div class="tab-content" id="tab-2">
  <div class="note">Exploratory analysis - many segment cuts increase the risk of false positives from multiple comparisons. Treat findings here as hypotheses, not conclusions.</div>
  <div class="chart-grid">
    {chart_device_breakdown}
    {chart_segment_device}
    <div class="chart-block full">{segment_table}</div>
  </div>
</div>

<div class="tab-content" id="tab-3">
  <div class="chart-grid">
    {chart_power}
    <div class="chart-block full">{sample_size_table}</div>
  </div>
</div>

<footer>Experimentation & A/B Testing Intelligence Platform - dashboard generated by src/generate_dashboard.py</footer>

<script>
function showTab(i) {{
  document.querySelectorAll('.tab-content').forEach((el, idx) => el.classList.toggle('active', idx === i));
  document.querySelectorAll('.tab-btn').forEach((el, idx) => el.classList.toggle('active', idx === i));
}}
</script>
</body>
</html>
"""


def build_dashboard(df, conv_results, segment_report, sample_size_tbl, power_curve, out_path):
    n_experiments = df["experiment_id"].nunique()
    n_users = len(df)
    n_sig = conv_results["statistically_significant"].sum()

    kpi_row_1 = (
        kpi_card("Experiments Analyzed", f"{n_experiments}")
        + kpi_card("Total Users", f"{n_users:,}")
        + kpi_card("Statistically Significant", f"{n_sig} / {len(conv_results)}")
        + kpi_card("Practically Significant", f"{conv_results['practically_significant'].sum()} / {len(conv_results)}")
    )

    results_table_cols = [
        "experiment_id", "n_control", "n_treatment", "control_rate", "treatment_rate",
        "relative_difference_pct", "p_value", "statistically_significant", "practically_significant",
    ]
    results_table_df = conv_results[results_table_cols].copy()
    results_table_df["p_value"] = results_table_df["p_value"].map(lambda x: f"{x:.4g}")

    html = PAGE_TEMPLATE.format(
        kpi_row_1=kpi_row_1,
        chart_sample_sizes=img_block("Sample Size by Experiment and Variant", chart_sample_sizes(df)),
        chart_conversion_variant=img_block("Conversion Rate by Experiment and Variant", chart_conversion_by_variant(df)),
        chart_ci=img_block("Conversion Lift with 95% Confidence Intervals", chart_experiment_ci(conv_results)),
        results_table=f"<h3>Full Statistical Results</h3>{df_to_html_table(results_table_df)}",
        chart_revenue_box=img_block("Revenue Distribution (Converters Only)", chart_revenue_boxplot(df)),
        chart_device_breakdown=img_block("Device Breakdown by Experiment", chart_device_breakdown(df)),
        chart_segment_device=img_block(
            "checkout_redesign: Conversion by Device", chart_segment_conversion(segment_report, "checkout_redesign", "device")
        ),
        segment_table=f"<h3>Segment-Level Results (Exploratory)</h3>{df_to_html_table(segment_report, pct_cols=['relative_diff_pct'])}",
        chart_power=img_block("Empirical Power Curve (Monte Carlo Simulation)", chart_power_curve(power_curve)),
        sample_size_table=f"<h3>Required Sample Size per Arm by Effect Size (baseline=8%)</h3>{df_to_html_table(sample_size_tbl)}",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved self-contained HTML dashboard -> {out_path}")
    print("Open this file directly in any web browser.")


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, BASE_DIR
    from ab_test_analysis import run_all_experiments
    from segment_analysis import full_segment_report
    from experiment_simulator import sample_size_table, simulate_power_curve

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError("Run the pipeline first (see README) - clean data not found.")

    df = pd.read_csv(CLEAN_DATA_PATH)
    conv_results = run_all_experiments(df)
    segment_report = full_segment_report(df)
    sample_size_tbl = sample_size_table(baseline_rate=0.08)
    power_curve = simulate_power_curve(
        baseline_rate=0.08, true_relative_effect=0.20,
        sample_sizes=[500, 1000, 2000, 4000, 8000], n_simulations=800,
    )

    out_path = os.path.join(BASE_DIR, "dashboard", "ab_testing_dashboard.html")
    build_dashboard(df, conv_results, segment_report, sample_size_tbl, power_curve, out_path)
