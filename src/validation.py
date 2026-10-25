"""
Data Validation and Quality Audit Module.

Provides comprehensive validation routines for:
- Schema and required column presence
- Missing values and customer ID coverage
- Duplicate record identification
- Cancellation and return flags
- Quantity and unit price anomalies
- Non-product stock codes (fees, manual adjustments, postage)
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from src.ingestion import EXPECTED_COLUMNS


# Common administrative / non-product stock codes observed in Online Retail II
ADMIN_STOCK_CODES = {
    "POST",         # Postage
    "D",            # Discount
    "M",            # Manual
    "BANK CHARGES", # Bank charges
    "PADS",         # Pads to match all cushions
    "DOT",          # Dotcom postage
    "CRUK",         # CRUK charity contribution
    "AMAZONFEE",    # Amazon fee
    "S",            # Samples
    "TEST",         # Test product
    "ADJUST",       # Adjustment
    "ADJUST2",      # Adjustment 2
}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check if all expected columns are present in the dataframe.
    
    Returns:
        (is_valid, missing_columns_list)
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count and percentage of missing values per column.
    """
    total = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total * 100).round(2)
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": [missing_count[c] for c in df.columns],
        "missing_percentage": [missing_pct[c] for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    return report.sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)


def identify_cancellations(df: pd.DataFrame, invoice_col: str = "Invoice") -> pd.Series:
    """
    Identify cancellation transactions. In Online Retail II, cancellations
    typically have an invoice number starting with 'C' (e.g. 'C489449').
    """
    invoice_str = df[invoice_col].astype(str).str.strip().str.upper()
    return invoice_str.str.startswith("C")


def identify_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> int:
    """
    Count full or subset duplicate records.
    """
    target_subset = subset if subset else [c for c in EXPECTED_COLUMNS if c in df.columns]
    return int(df.duplicated(subset=target_subset, keep="first").sum())


def perform_data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute full data quality audit and return comprehensive statistics.
    """
    is_valid, missing_cols = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")

    total_rows = len(df)
    
    # Missingness
    missingness_df = compute_missingness(df)
    missing_customer_count = int(df["Customer ID"].isnull().sum())
    missing_customer_pct = round(missing_customer_count / total_rows * 100, 2)
    missing_desc_count = int(df["Description"].isnull().sum())
    missing_desc_pct = round(missing_desc_count / total_rows * 100, 2)

    # Duplicates
    dup_count = identify_duplicates(df)
    dup_pct = round(dup_count / total_rows * 100, 2)

    # Cancellations (Invoice starting with 'C')
    is_cancel = identify_cancellations(df)
    cancel_count = int(is_cancel.sum())
    cancel_pct = round(cancel_count / total_rows * 100, 2)

    # Quantity anomalies
    neg_qty = df["Quantity"] < 0
    zero_qty = df["Quantity"] == 0
    neg_qty_count = int(neg_qty.sum())
    zero_qty_count = int(zero_qty.sum())
    
    # Cancellations vs negative quantity mismatch:
    # Cancellations that have positive quantity or negative quantity without 'C'
    neg_qty_without_c = int((neg_qty & (~is_cancel)).sum())
    cancel_with_pos_qty = int((is_cancel & (df["Quantity"] > 0)).sum())

    # Price anomalies
    neg_price = df["Price"] < 0
    zero_price = df["Price"] == 0
    neg_price_count = int(neg_price.sum())
    zero_price_count = int(zero_price.sum())

    # Admin / non-product stock codes
    stock_clean = df["StockCode"].astype(str).str.strip().str.upper()
    admin_stock_mask = stock_clean.isin(ADMIN_STOCK_CODES)
    admin_stock_count = int(admin_stock_mask.sum())

    # Date span
    invoice_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    min_date = invoice_dates.min()
    max_date = invoice_dates.max()

    # Unique entities
    unique_invoices = int(df["Invoice"].nunique())
    unique_customers = int(df["Customer ID"].dropna().nunique())
    unique_products = int(df["StockCode"].nunique())
    unique_countries = int(df["Country"].nunique())

    audit_summary = {
        "total_records": total_rows,
        "date_range": {
            "start": str(min_date),
            "end": str(max_date),
            "duration_days": (max_date - min_date).days if pd.notnull(min_date) else None,
        },
        "entities": {
            "unique_invoices": unique_invoices,
            "unique_customers": unique_customers,
            "unique_stock_codes": unique_products,
            "unique_countries": unique_countries,
        },
        "missingness": {
            "missing_customer_id_count": missing_customer_count,
            "missing_customer_id_pct": missing_customer_pct,
            "missing_description_count": missing_desc_count,
            "missing_description_pct": missing_desc_pct,
            "table": missingness_df.to_dict(orient="records"),
        },
        "integrity": {
            "duplicate_records_count": dup_count,
            "duplicate_records_pct": dup_pct,
            "cancellations_count": cancel_count,
            "cancellations_pct": cancel_pct,
            "negative_quantity_count": neg_qty_count,
            "zero_quantity_count": zero_qty_count,
            "negative_quantity_without_c_flag": neg_qty_without_c,
            "cancellations_with_positive_quantity": cancel_with_pos_qty,
            "negative_price_count": neg_price_count,
            "zero_price_count": zero_price_count,
            "admin_stock_codes_count": admin_stock_count,
        },
    }
    return audit_summary
