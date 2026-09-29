"""Shared path/IO helpers for the A/B testing intelligence project."""

import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "ab_test_users.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ab_test_users_clean.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "processed", "experiment_results_summary.csv")
SEGMENT_RESULTS_PATH = os.path.join(BASE_DIR, "data", "processed", "segment_results_summary.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")


def ensure_dirs():
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
