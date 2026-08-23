"""
main_eda.py
------------
Week 1 (Module M2) EDA entry point.

Loads the latest versioned/processed dataset produced by main_week1.py,
runs the full EDA suite, saves plots to eda_outputs/, and writes a single
markdown report (eda_outputs/EDA_REPORT.md) summarizing the findings.

Usage:
    python main_week1.py     # if not already run, to produce the processed data
    python main_eda.py
"""

import os
import sys
import logging
import yaml
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from src.eda import run_full_eda
from src.eda_report import render_markdown_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main_eda")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    processed_path = os.path.join(base_dir, cfg["paths"]["processed_data_dir"], "trips_features_train.parquet")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed training dataset not found at {processed_path}. Run `python main_week1.py` first."
        )

    logger.info("Loading processed TRAINING features from %s (EDA runs on train split only "
                "-- test data must stay unseen)", processed_path)
    df = pd.read_parquet(processed_path)

    out_dir = os.path.join(base_dir, "eda_outputs")
    results = run_full_eda(df, out_dir)

    report_md = render_markdown_report(results, images_dir_relative=".")
    report_path = os.path.join(out_dir, "EDA_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print("\n===== EDA SUMMARY =====")
    print(f"Rows analyzed          : {results['overview']['n_rows']}")
    print(f"Plots saved to         : {out_dir}")
    print(f"Report                 : {report_path}")
    print(f"Top correlate w/ target: "
          f"{max(results['correlation']['correlation_with_target'].items(), key=lambda kv: abs(kv[1]))}")
    print("========================\n")


if __name__ == "__main__":
    main()
