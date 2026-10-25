"""
CLI Script to execute Phase 2 pipeline:
1. Ingest raw Online Retail II dataset (or sample if configured)
2. Clean data and partition into Gross, Net, and Customer tiers
3. Load into indexed SQLite analytical database
4. Run EDA engine and print key business findings
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_raw_dataset
from src.cleaning import execute_cleaning_pipeline
from src.database import AnalyticalDatabase
from src.eda import EDAEngine


def run_phase2_pipeline(
    data_path: str = "data/raw/online_retail_II.xlsx",
    db_path: str = "data/ecommerce_analytical.db",
    sample_rows: int = None,
):
    print("=" * 70)
    print("PHASE 2: CLEANING, ANALYTICAL SQLITE DATABASE & EDA ENGINE")
    print("=" * 70)

    # 1. Ingestion
    print(f"\n[1/4] Ingesting raw dataset from {data_path}...")
    raw_df, meta = load_raw_dataset(data_path, nrows=sample_rows)
    print(f" -> Ingested {len(raw_df):,} raw records from {len(meta['sheets_loaded'])} sheet(s).")

    # 2. Cleaning & Sales Tier Partitioning
    print("\n[2/4] Executing cleaning pipeline and separating sales tiers...")
    df_clean, tiers, stats = execute_cleaning_pipeline(raw_df)
    print(f" -> Baseline Cleaned Records: {stats['cleaned_base_count']:,}")
    print(f" -> Gross Sales Records:     {stats['gross_sales_count']:,} (£{stats['gross_revenue']:,.2f})")
    print(f" -> Net Sales Records:       {stats['net_sales_count']:,} (£{stats['net_revenue']:,.2f})")
    print(f" -> Customer Transactions:   {stats['customer_transactions_count']:,} (Unique Custs: {stats['unique_customers_identified']:,})")
    print(f" -> Refund Volume:           £{stats['refund_total']:,.2f}")

    # 3. SQLite Database Build
    print(f"\n[3/4] Initializing and populating SQLite database at {db_path}...")
    db = AnalyticalDatabase(db_path=db_path)
    counts = db.load_clean_data(df_clean, if_exists="replace")
    print(f" -> Stored {counts['transactions']:,} transactions")
    print(f" -> Built {counts['customers']:,} customer profiles")
    print(f" -> Built {counts['products']:,} product catalog records")

    # 4. EDA Analytics
    print("\n[4/4] Running EDA Engine...")
    eda = EDAEngine(df_clean)
    kpis = eda.compute_summary_kpis()
    concentration = eda.compute_market_concentration_hhi()
    pareto = eda.compute_product_pareto()
    prices = eda.compute_unit_price_dynamics()

    print("\n" + "-" * 50)
    print("EXECUTIVE SUMMARY & BUSINESS METRICS")
    print("-" * 50)
    print(f"Gross Revenue:         £{kpis['total_gross_revenue']:,.2f}")
    print(f"Net Revenue:           £{kpis['total_net_revenue']:,.2f}")
    print(f"Refund Rate:           {kpis['refund_ratio_pct']}% (£{kpis['total_refunds']:,.2f})")
    print(f"Total Orders:          {kpis['total_orders']:,}")
    print(f"Average Order Value:   £{kpis['avg_order_value']:.2f}")
    print(f"UK Revenue Share:      {concentration['uk_revenue_share_pct']}% (HHI: {concentration['hhi_index']} - {concentration['concentration_level']})")
    print(f"Product Pareto:        {pareto['product_percentage_for_80_rev']}% of catalog produces 80% of revenue (Holds: {pareto['pareto_holds']})")
    print(f"Unit Price (Median):   £{prices['median_unit_price']:.2f} (IQR: £{prices['p25_unit_price']:.2f} - £{prices['p75_unit_price']:.2f})")
    print("-" * 50)
    print("\n[SUCCESS] Phase 2 pipeline executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Cleaning, SQLite DB and EDA pipeline.")
    parser.add_argument("--data-path", default="data/raw/online_retail_II.xlsx", help="Raw dataset path")
    parser.add_argument("--db-path", default="data/ecommerce_analytical.db", help="SQLite database output path")
    parser.add_argument("--sample", type=int, default=None, help="Optional row limit for quick inspection")
    args = parser.parse_args()

    run_phase2_pipeline(data_path=args.data_path, db_path=args.db_path, sample_rows=args.sample)
