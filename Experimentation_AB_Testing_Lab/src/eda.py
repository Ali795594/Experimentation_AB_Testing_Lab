"""
eda.py
-------
Generates core EDA charts for the A/B testing project and saves them to
visualizations/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def plot_sample_sizes(df, out_dir):
    counts = df.groupby(["experiment_id", "variant"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Users")
    ax.set_title("Sample Size by Experiment and Variant")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_sizes.png"), dpi=120)
    plt.close(fig)


def plot_conversion_by_variant(df, out_dir):
    conv = df.groupby(["experiment_id", "variant"])["conversion"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    conv.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate by Experiment and Variant")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conversion_by_variant.png"), dpi=120)
    plt.close(fig)


def plot_revenue_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    converters = df[df["conversion"] == 1]
    sns.boxplot(data=converters, x="experiment_id", y="revenue", hue="variant", ax=ax)
    ax.set_title("Revenue Distribution (Converters Only) by Experiment/Variant")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "revenue_distribution.png"), dpi=120)
    plt.close(fig)


def plot_device_breakdown(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = df.groupby(["experiment_id", "device"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Users")
    ax.set_title("Device Breakdown by Experiment")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "device_breakdown.png"), dpi=120)
    plt.close(fig)


def plot_customer_type_conversion(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    conv = df.groupby(["experiment_id", "customer_type"])["conversion"].mean().unstack()
    conv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate by Customer Type")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "customer_type_conversion.png"), dpi=120)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plot_sample_sizes(df, out_dir)
    plot_conversion_by_variant(df, out_dir)
    plot_revenue_distribution(df, out_dir)
    plot_device_breakdown(df, out_dir)
    plot_customer_type_conversion(df, out_dir)
    print(f"Saved 5 EDA charts -> {out_dir}")


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, VIZ_DIR

    df = pd.read_csv(CLEAN_DATA_PATH)
    run_eda(df, VIZ_DIR)
