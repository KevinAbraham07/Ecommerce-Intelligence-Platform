"""
Data Cleaning and Transformation Pipeline for E-commerce Intelligence.

This module enforces reproducible, transparent data cleaning transformations,
separating transaction scopes according to rigorous analytical accounting:
  - Gross Sales (all legitimate product orders, quantity > 0, price > 0)
  - Net Sales (accounting for returns / cancellations and negative adjustments)
  - Customer Transactions (cleaned transactions with valid customer IDs for cohort/RFM modeling)

Standardizes data types, isolates administrative codes, extracts temporal features,
and calculates line item amounts.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.ingestion import STANDARDIZED_COLUMN_MAP
from src.validation import ADMIN_STOCK_CODES, identify_cancellations


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw column names to canonical snake_case names.
    Preserves any extra columns (like sheet_source).
    """
    df = df.copy()
    rename_dict = {k: v for k, v in STANDARDIZED_COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename_dict)


def clean_base_transactions(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    filter_admin_codes: bool = True,
) -> pd.DataFrame:
    """
    Perform baseline standardization and normalization on raw or renamed transactions.
    
    Operations:
      1. Standardize column names if not already mapped.
      2. Clean string fields (invoice, stock_code, description, country).
      3. Parse invoice_date to datetime and extract year, month, year_month, day_of_week, hour.
      4. Convert customer_id to nullable integer (Int64).
      5. Flag cancellations (is_cancellation).
      6. Flag administrative / non-product items (is_admin_code).
      7. Calculate line_total (quantity * unit_price).
      8. Optionally deduplicate exact transaction records.
      9. Optionally exclude administrative stock codes (fees, postage, discounts, manual).
    
    Returns:
        Standardized baseline DataFrame.
    """
    cleaned = standardize_column_names(df)

    # 1. Strip and format string identifiers
    cleaned["invoice"] = cleaned["invoice"].astype(str).str.strip()
    cleaned["stock_code"] = cleaned["stock_code"].astype(str).str.strip().str.upper()
    
    if "description" in cleaned.columns:
        cleaned["description"] = cleaned["description"].astype(str).str.strip()
        # Replace string 'nan' or empty string with NaN
        cleaned["description"] = cleaned["description"].replace({"": np.nan, "nan": np.nan, "NAN": np.nan})
    else:
        cleaned["description"] = np.nan

    if "country" in cleaned.columns:
        cleaned["country"] = cleaned["country"].astype(str).str.strip()
    else:
        cleaned["country"] = "Unknown"

    # 2. Parse dates & temporal features
    cleaned["invoice_date"] = pd.to_datetime(cleaned["invoice_date"], errors="coerce")
    cleaned["year"] = cleaned["invoice_date"].dt.year
    cleaned["month"] = cleaned["invoice_date"].dt.month
    cleaned["year_month"] = cleaned["invoice_date"].dt.to_period("M").astype(str)
    cleaned["day_of_week"] = cleaned["invoice_date"].dt.day_name()
    cleaned["hour"] = cleaned["invoice_date"].dt.hour

    # 3. Numeric conversions
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce").fillna(0).astype(int)
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce").fillna(0.0)
    cleaned["line_total"] = (cleaned["quantity"] * cleaned["unit_price"]).round(4)

    # 4. Customer ID as Int64 (nullable int)
    cleaned["customer_id"] = pd.to_numeric(cleaned["customer_id"], errors="coerce").astype("Int64")

    # 5. Cancellation flag
    cleaned["is_cancellation"] = (
        cleaned["invoice"].str.upper().str.startswith("C") | (cleaned["quantity"] < 0)
    )

    # 6. Admin / non-product stock code flag
    cleaned["is_admin_code"] = cleaned["stock_code"].isin(ADMIN_STOCK_CODES)

    # 7. Deduplication
    if drop_duplicates:
        subset_cols = ["invoice", "stock_code", "invoice_date", "quantity", "unit_price", "customer_id"]
        avail_subset = [c for c in subset_cols if c in cleaned.columns]
        cleaned = cleaned.drop_duplicates(subset=avail_subset, keep="first").reset_index(drop=True)

    # 8. Filter non-product admin items if requested
    if filter_admin_codes:
        cleaned = cleaned[~cleaned["is_admin_code"]].reset_index(drop=True)

    return cleaned


def separate_sales_tiers(df_clean: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Partition the baseline cleaned dataset into three distinct analytical scopes:
    
    1. 'gross_sales':
       - Completed product orders only.
       - quantity > 0 AND unit_price > 0.
       - is_cancellation == False.
       - Used for demand volume, product affinity, basket sizing, price distributions.
       
    2. 'net_sales':
       - Completed product orders PLUS cancellations/returns.
       - Captures real accounting cash flow: positive revenue minus refunds.
       - Filters out zero-price anomalies (bad debt, gifts, system errors).
       - unit_price > 0 (or quantity < 0 with unit_price > 0).
       - Used for actual fiscal revenue metrics and net margin modeling.
       
    3. 'customer_transactions':
       - Cleaned transactions with confirmed Customer ID (customer_id.notnull()).
       - Excludes guest / unregistered checkouts.
       - Separates customer-level gross and net activity.
       - Primary foundation for Cohort Analysis, RFM Segmentation, LTV, Repeat modeling.
    
    Returns:
        Dict with keys: 'gross_sales', 'net_sales', 'customer_transactions', 'cancellations_only'.
    """
    # Exclude zero or negative price anomalies (e.g. system adjustments, inventory write-offs)
    valid_price = df_clean["unit_price"] > 0

    gross_mask = valid_price & (df_clean["quantity"] > 0) & (~df_clean["is_cancellation"])
    gross_sales = df_clean[gross_mask].copy().reset_index(drop=True)

    # Net sales: valid positive orders + legitimate returns/cancellations
    # Note: returns have quantity < 0, unit_price > 0 -> line_total is negative
    net_mask = valid_price & (
        ((df_clean["quantity"] > 0) & (~df_clean["is_cancellation"])) |
        ((df_clean["quantity"] < 0) & (df_clean["is_cancellation"]))
    )
    net_sales = df_clean[net_mask].copy().reset_index(drop=True)

    # Cancellations only
    cancellations_mask = valid_price & df_clean["is_cancellation"]
    cancellations_only = df_clean[cancellations_mask].copy().reset_index(drop=True)

    # Customer transactions: valid customer ID and valid price
    cust_mask = valid_price & df_clean["customer_id"].notnull()
    customer_transactions = df_clean[cust_mask].copy().reset_index(drop=True)

    return {
        "gross_sales": gross_sales,
        "net_sales": net_sales,
        "customer_transactions": customer_transactions,
        "cancellations_only": cancellations_only,
    }


def execute_cleaning_pipeline(
    raw_df: pd.DataFrame,
    drop_duplicates: bool = True,
    filter_admin_codes: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, any]]:
    """
    Run full reproducible cleaning pipeline on raw input data.
    
    Returns:
        Tuple of:
          - df_clean: standard baseline dataset
          - tiers: dictionary of sales tiers ('gross_sales', 'net_sales', 'customer_transactions', 'cancellations_only')
          - audit_stats: summary metrics comparing raw to cleaned
    """
    raw_count = len(raw_df)
    df_clean = clean_base_transactions(
        raw_df,
        drop_duplicates=drop_duplicates,
        filter_admin_codes=filter_admin_codes,
    )
    tiers = separate_sales_tiers(df_clean)

    audit_stats = {
        "raw_record_count": raw_count,
        "cleaned_base_count": len(df_clean),
        "gross_sales_count": len(tiers["gross_sales"]),
        "net_sales_count": len(tiers["net_sales"]),
        "cancellations_count": len(tiers["cancellations_only"]),
        "customer_transactions_count": len(tiers["customer_transactions"]),
        "gross_revenue": float(tiers["gross_sales"]["line_total"].sum()),
        "net_revenue": float(tiers["net_sales"]["line_total"].sum()),
        "refund_total": float(tiers["cancellations_only"]["line_total"].sum()),
        "unique_customers_identified": int(tiers["customer_transactions"]["customer_id"].nunique()),
    }

    return df_clean, tiers, audit_stats
