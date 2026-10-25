"""
Unit tests for Data Cleaning, Sales Tier Separation, SQLite Analytical Database, and EDA Engine.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.cleaning import (
    standardize_column_names,
    clean_base_transactions,
    separate_sales_tiers,
    execute_cleaning_pipeline,
)
from src.database import AnalyticalDatabase
from src.eda import EDAEngine


@pytest.fixture
def raw_test_transactions():
    """Create synthetic raw transactions including cancellations, duplicates, admin codes, and guests."""
    data = {
        "Invoice": [
            "536365", "536365", "536365",  # standard valid items
            "C536366",                      # cancellation
            "536367",                      # admin code (POST)
            "536368",                      # zero price anomaly
            "536369",                      # guest checkout (no customer ID)
            "536365",                      # exact duplicate row
        ],
        "StockCode": [
            "85123A", "71053", "84406B",
            "85123A",
            "POST",
            "21730",
            "84029G",
            "85123A",
        ],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "CREAM CUPID HEARTS COAT HANGER",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "POSTAGE",
            "GLASS STAR FROSTED T-LIGHT HOLDER",
            "KNITTED UNION FLAG HOT WATER BOTTLE",
            "WHITE HANGING HEART T-LIGHT HOLDER",
        ],
        "Quantity": [6, 6, 8, -2, 1, 10, 4, 6],
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:28:00",
            "2010-12-01 09:00:00",
            "2010-12-01 10:00:00",
            "2010-12-02 11:30:00",
            "2010-12-01 08:26:00",
        ],
        "Price": [2.55, 3.39, 2.75, 2.55, 18.00, 0.00, 3.75, 2.55],
        "Customer ID": [17850.0, 17850.0, 17850.0, 17850.0, np.nan, 13047.0, np.nan, 17850.0],
        "Country": ["United Kingdom"] * 6 + ["Germany", "United Kingdom"],
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------
# Cleaning Tests
# -------------------------------------------------------------

def test_clean_base_transactions(raw_test_transactions):
    cleaned = clean_base_transactions(raw_test_transactions, drop_duplicates=True, filter_admin_codes=True)
    # Admin code 'POST' dropped, exact duplicate dropped
    # 8 total -> -1 duplicate -> -1 admin = 6 rows
    assert len(cleaned) == 6
    assert "line_total" in cleaned.columns
    assert "is_cancellation" in cleaned.columns
    assert "year_month" in cleaned.columns
    # Verify customer_id type
    assert str(cleaned["customer_id"].dtype) == "Int64"
    # Verify cancellation flag
    cancellation_rows = cleaned[cleaned["is_cancellation"]]
    assert len(cancellation_rows) == 1
    assert cancellation_rows.iloc[0]["invoice"] == "C536366"


def test_separate_sales_tiers(raw_test_transactions):
    cleaned = clean_base_transactions(raw_test_transactions, drop_duplicates=True, filter_admin_codes=True)
    tiers = separate_sales_tiers(cleaned)

    gross = tiers["gross_sales"]
    net = tiers["net_sales"]
    customer = tiers["customer_transactions"]

    # Gross should have no cancellations and no zero price
    assert all(gross["quantity"] > 0)
    assert all(gross["unit_price"] > 0)
    assert not any(gross["is_cancellation"])

    # Net should include valid positive sales and valid cancellations
    assert any(net["is_cancellation"])
    assert not any(net["unit_price"] <= 0)

    # Customer transactions must have non-null customer_id
    assert customer["customer_id"].notnull().all()

    # Gross revenue must exceed net revenue due to cancellations
    assert gross["line_total"].sum() > net["line_total"].sum()


def test_execute_cleaning_pipeline(raw_test_transactions):
    df_clean, tiers, stats = execute_cleaning_pipeline(raw_test_transactions)
    assert stats["raw_record_count"] == 8
    assert stats["gross_revenue"] > stats["net_revenue"]
    assert stats["unique_customers_identified"] >= 1


# -------------------------------------------------------------
# SQLite Analytical Database Tests
# -------------------------------------------------------------

def test_database_creation_and_loading(raw_test_transactions, tmp_path):
    db_file = tmp_path / "test_analytical.db"
    db = AnalyticalDatabase(db_path=db_file)
    
    cleaned = clean_base_transactions(raw_test_transactions)
    counts = db.load_clean_data(cleaned, if_exists="replace")

    assert counts["transactions"] == len(cleaned)
    assert counts["customers"] > 0
    assert counts["products"] > 0

    # Query tests
    monthly = db.get_monthly_revenue_trends()
    assert not monthly.empty
    assert "gross_revenue" in monthly.columns

    country_df = db.get_country_aggregates()
    assert not country_df.empty
    assert "country" in country_df.columns

    top_prod = db.get_top_selling_products(top_n=5)
    assert not top_prod.empty

    cancel_df = db.get_cancellation_rates_by_month()
    assert not cancel_df.empty
    assert "cancellation_rate_pct" in cancel_df.columns


# -------------------------------------------------------------
# EDA Engine Tests
# -------------------------------------------------------------

def test_eda_engine_calculations(raw_test_transactions):
    cleaned = clean_base_transactions(raw_test_transactions)
    eda = EDAEngine(cleaned)

    # KPIs
    kpis = eda.compute_summary_kpis()
    assert kpis["total_gross_revenue"] > 0
    assert kpis["total_orders"] > 0
    assert kpis["unique_customers"] >= 1

    # Monthly trends
    m_trends = eda.compute_monthly_trends()
    assert "gross_revenue" in m_trends.columns
    assert "net_revenue" in m_trends.columns
    assert "aov" in m_trends.columns

    # Country breakdown & HHI
    c_breakdown = eda.compute_country_breakdown()
    assert not c_breakdown.empty
    hhi = eda.compute_market_concentration_hhi()
    assert "hhi_index" in hhi
    assert "uk_revenue_share_pct" in hhi

    # Product Pareto
    pareto = eda.compute_product_pareto()
    assert pareto["total_products"] > 0
    assert "product_percentage_for_80_rev" in pareto

    # Unit price dynamics
    prices = eda.compute_unit_price_dynamics()
    assert prices["median_unit_price"] > 0
    assert prices["mean_unit_price"] > 0

    # Temporal patterns
    temporal = eda.compute_temporal_patterns()
    assert "by_day_of_week" in temporal
    assert "by_hour" in temporal
