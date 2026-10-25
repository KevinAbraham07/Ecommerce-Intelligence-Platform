"""
Script to execute the raw dataset ingestion and generate the data quality audit report.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.download_dataset import ensure_dataset
from src.ingestion import load_raw_dataset
from src.validation import perform_data_audit
from src.reporting import save_audit_report


def main():
    print("=== Step 1: Checking Raw Dataset ===")
    dataset_path = ensure_dataset()

    print(f"=== Step 2: Ingesting Raw Data from {dataset_path.name} ===")
    print("Reading file (this may take a few moments for large multi-sheet Excel)...")
    raw_df, metadata = load_raw_dataset(dataset_path)
    print(f"Loaded {len(raw_df):,} records with {len(raw_df.columns)} columns.")

    print("=== Step 3: Performing Data Quality Audit ===")
    audit = perform_data_audit(raw_df)

    print("=== Step 4: Generating Audit Reports ===")
    reports_dir = ROOT_DIR / "reports"
    md_path, json_path = save_audit_report(audit, metadata, reports_dir)
    print(f"Markdown report generated: {md_path}")
    print(f"JSON audit data generated: {json_path}")
    print("\n--- Summary Highlights ---")
    print(f"Total Records: {audit['total_records']:,}")
    print(f"Date Coverage: {audit['date_range']['start']} to {audit['date_range']['end']}")
    print(f"Missing Customer IDs: {audit['missingness']['missing_customer_id_count']:,} ({audit['missingness']['missing_customer_id_pct']}%)")
    print(f"Cancellations: {audit['integrity']['cancellations_count']:,} ({audit['integrity']['cancellations_pct']}%)")
    print(f"Duplicates: {audit['integrity']['duplicate_records_count']:,} ({audit['integrity']['duplicate_records_pct']}%)")
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()
