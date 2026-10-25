"""
Report generator for Data Quality Audit.
Formats data audit results into readable Markdown and JSON artifacts.
"""

from pathlib import Path
from typing import Any, Dict
import json


def format_markdown_report(audit: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Format audit results as a thorough Markdown report."""
    md = []
    md.append("# Data Quality & Ingestion Audit Report")
    md.append("## UCI Online Retail II Dataset\n")
    
    md.append("### 1. Ingestion Overview")
    md.append(f"- **Source File**: `{metadata.get('source_file')}`")
    md.append(f"- **File Size**: `{metadata.get('file_size_mb')} MB`")
    md.append(f"- **Total Rows Read**: `{audit['total_records']:,}`")
    md.append(f"- **Date Range**: `{audit['date_range']['start']}` to `{audit['date_range']['end']}` ({audit['date_range']['duration_days']} days)")
    if "sheets_loaded" in metadata and metadata["sheets_loaded"]:
        md.append("- **Sheets Loaded**:")
        for s in metadata["sheets_loaded"]:
            md.append(f"  - `{s['sheet_name']}`: {s['rows_read']:,} rows")
    md.append("")

    md.append("### 2. High-Level Entity Counts")
    md.append(f"- **Unique Invoices**: `{audit['entities']['unique_invoices']:,}`")
    md.append(f"- **Unique Customers (non-null)**: `{audit['entities']['unique_customers']:,}`")
    md.append(f"- **Unique Stock Codes**: `{audit['entities']['unique_stock_codes']:,}`")
    md.append(f"- **Unique Countries**: `{audit['entities']['unique_countries']:,}`")
    md.append("")

    md.append("### 3. Missing Value Analysis")
    md.append("| Column | Missing Count | Missing % | Data Type |")
    md.append("|---|---|---|---|")
    for row in audit["missingness"]["table"]:
        md.append(f"| {row['column']} | {row['missing_count']:,} | {row['missing_percentage']}% | {row['dtype']} |")
    md.append("")
    md.append(f"> **Key Finding**: `Customer ID` is missing in `{audit['missingness']['missing_customer_id_count']:,}` rows ({audit['missingness']['missing_customer_id_pct']}%). ")
    md.append("> Guest / guest-checkout transactions must not be blindly imputed or discarded for macro sales trends, but must be segregated for customer-level analytics (RFM, cohorts, repeat purchase prediction).\n")

    md.append("### 4. Integrity & Anomaly Audit")
    integ = audit["integrity"]
    md.append(f"- **Duplicate Rows**: `{integ['duplicate_records_count']:,}` ({integ['duplicate_records_pct']}%)")
    md.append(f"- **Cancellations (Invoice starting with 'C')**: `{integ['cancellations_count']:,}` ({integ['cancellations_pct']}%)")
    md.append(f"- **Negative Quantities**: `{integ['negative_quantity_count']:,}`")
    md.append(f"- **Negative Quantities without 'C' prefix**: `{integ['negative_quantity_without_c_flag']:,}` (damages, inventory write-offs, manual corrections)")
    md.append(f"- **Zero Quantities**: `{integ['zero_quantity_count']:,}`")
    md.append(f"- **Zero Unit Prices**: `{integ['zero_price_count']:,}`")
    md.append(f"- **Negative Unit Prices**: `{integ['negative_price_count']:,}` (e.g. bad debt adjustments)")
    md.append(f"- **Administrative / Non-Product Stock Codes**: `{integ['admin_stock_codes_count']:,}` (e.g. POST, D, M, BANK CHARGES)")
    md.append("")

    md.append("### 5. Cleaning Recommendations for Downstream Modules")
    md.append("1. **Gross vs. Net Revenue Policy**:")
    md.append("   - Gross sales: Positive quantity and positive price transactions.")
    md.append("   - Net sales: Include cancellations/returns, or net against parent purchases where applicable.")
    md.append("2. **Market-Basket Analysis Policy**:")
    md.append("   - Must exclude cancellations and non-product stock codes (`POST`, `D`, `M`, etc.) to prevent false co-occurrences.")
    md.append("3. **Customer Analytics Policy (RFM & Retention)**:")
    md.append("   - Strictly evaluate customers with valid `Customer ID`.")
    md.append("   - Returns should offset monetary spend accurately without generating negative lifetime customer values.")
    md.append("")
    return "\n".join(md)


def save_audit_report(
    audit: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save both Markdown and JSON reports to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_content = format_markdown_report(audit, metadata)
    md_path = output_dir / "data_quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_path = output_dir / "data_quality_report.json"
    full_data = {"metadata": metadata, "audit": audit}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, default=str)

    return md_path, json_path
