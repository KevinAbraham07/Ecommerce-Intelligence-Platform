"""
Unit tests for data validation, schema checking, and anomaly detection.
Uses synthetic test fixtures with known expected outputs.
"""

import pytest
import pandas as pd
import numpy as np
from src.validation import (
    validate_schema,
    compute_missingness,
    identify_cancellations,
    identify_duplicates,
    perform_data_audit,
    ADMIN_STOCK_CODES,
)
from src.ingestion import EXPECTED_COLUMNS


@pytest.fixture
def sample_valid_df():
    """Create a small valid dataframe representing typical transaction records."""
    data = {
        "Invoice": ["489434", "489434", "C489435", "489436", "489437", "489437"],
        "StockCode": ["85048", "79323P", "85048", "POST", "21232", "21232"],
        "Description": [
            "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
            "PINK CHERRY LIGHTS",
            "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
            "POSTAGE",
            "STRAWBERRY CERAMIC TRINKET BOX",
            "STRAWBERRY CERAMIC TRINKET BOX",
        ],
        "Quantity": [12, 6, -1, 1, 10, 10],  # 1 cancellation, 1 duplicate row
        "InvoiceDate": [
            "2009-12-01 07:45:00",
            "2009-12-01 07:45:00",
            "2009-12-01 07:46:00",
            "2009-12-01 09:00:00",
            "2009-12-01 10:00:00",
            "2009-12-01 10:00:00",
        ],
        "Price": [6.95, 6.75, 6.95, 18.00, 1.25, 1.25],
        "Customer ID": [13085.0, 13085.0, 13085.0, np.nan, 15311.0, 15311.0],
        "Country": [
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
        ],
    }
    return pd.DataFrame(data)


def test_validate_schema_success(sample_valid_df):
    is_valid, missing = validate_schema(sample_valid_df)
    assert is_valid is True
    assert missing == []


def test_validate_schema_failure():
    df_incomplete = pd.DataFrame({"Invoice": ["123"], "StockCode": ["456"]})
    is_valid, missing = validate_schema(df_incomplete)
    assert is_valid is False
    assert "Quantity" in missing
    assert "Customer ID" in missing


def test_compute_missingness(sample_valid_df):
    report = compute_missingness(sample_valid_df)
    # Total rows = 6. Customer ID has 1 NaN -> 1/6 = 16.67%
    cust_row = report[report["column"] == "Customer ID"].iloc[0]
    assert cust_row["missing_count"] == 1
    assert round(cust_row["missing_percentage"], 1) == 16.7


def test_identify_cancellations(sample_valid_df):
    cancels = identify_cancellations(sample_valid_df)
    # Invoice C489435 should be flagged
    assert cancels.sum() == 1
    assert cancels.iloc[2] is True or cancels.iloc[2] == 1
    assert cancels.iloc[0] is False or cancels.iloc[0] == 0


def test_identify_duplicates(sample_valid_df):
    # Rows 4 and 5 are identical duplicates
    dup_count = identify_duplicates(sample_valid_df)
    assert dup_count == 1


def test_perform_data_audit(sample_valid_df):
    audit = perform_data_audit(sample_valid_df)
    assert audit["total_records"] == 6
    assert audit["entities"]["unique_invoices"] == 4
    assert audit["entities"]["unique_customers"] == 2
    assert audit["missingness"]["missing_customer_id_count"] == 1
    assert audit["integrity"]["duplicate_records_count"] == 1
    assert audit["integrity"]["cancellations_count"] == 1
    assert audit["integrity"]["negative_quantity_count"] == 1
    assert audit["integrity"]["admin_stock_codes_count"] == 1  # "POST"
