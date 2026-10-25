"""
CLI Script to execute Customer Analytics & RFM Segmentation (Phase 3).

Loads clean transactions, runs RFM calculation, scores quintiles (1-5),
assigns customer segments, calculates segment revenue contributions,
and prints the complete customer portfolio distribution.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_raw_dataset
from src.cleaning import execute_cleaning_pipeline
from src.rfm import CustomerAnalytics


def run_rfm_cli(
    data_path: str = "data/raw/online_retail_II.xlsx",
    sample_rows: int = None,
):
    print("=" * 70)
    print("PHASE 3: CUSTOMER ANALYTICS & RFM SEGMENTATION")
    print("=" * 70)

    print(f"\n[1/3] Ingesting & cleaning transactions from {data_path}...")
    raw_df, _ = load_raw_dataset(data_path, nrows=sample_rows)
    _, tiers, stats = execute_cleaning_pipeline(raw_df)
    customer_df = tiers["customer_transactions"]
    print(f" -> Processed {len(customer_df):,} verified customer transactions across {stats['unique_customers_identified']:,} unique customers.")

    print("\n[2/3] Executing RFM segmentation engine...")
    analytics = CustomerAnalytics(customer_df)
    segmented_rfm, summary = analytics.run_rfm_analysis()

    print("\n[3/3] Segment Distribution & Revenue Contribution:")
    print("-" * 75)
    seg_table = summary["segment_table"]
    
    # Formatted display
    print(f"{'Segment':<24} | {'Customers':>9} | {'Cust %':>7} | {'Revenue (£)':>13} | {'Rev %':>6} | {'AOV (£)':>8}")
    print("-" * 75)
    for _, row in seg_table.iterrows():
        aov = round(row["total_revenue"] / row["customer_count"], 2) if row["customer_count"] > 0 else 0.0
        print(
            f"{row['segment']:<24} | "
            f"{row['customer_count']:>9,d} | "
            f"{row['customer_share_pct']:>6.2f}% | "
            f"{row['total_revenue']:>13,.2f} | "
            f"{row['revenue_share_pct']:>5.1f}% | "
            f"{aov:>8.2f}"
        )
    print("-" * 75)
    print(f"Total Portfolio Revenue: £{summary['total_revenue']:,.2f} across {summary['total_customers']:,} customers.")
    print("=" * 75)
    print("\n[SUCCESS] Phase 3 RFM Analysis completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFM segmentation analysis.")
    parser.add_argument("--data-path", default="data/raw/online_retail_II.xlsx", help="Raw dataset path")
    parser.add_argument("--sample", type=int, default=None, help="Optional row limit for quick inspection")
    args = parser.parse_args()

    run_rfm_cli(data_path=args.data_path, sample_rows=args.sample)
