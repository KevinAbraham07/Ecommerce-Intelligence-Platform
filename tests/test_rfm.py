"""
Unit tests for Module C: Customer Analytics & RFM Segmentation.
"""

from datetime import datetime
import pytest
import pandas as pd
import numpy as np

from src.rfm import (
    calculate_rfm_metrics,
    assign_rfm_scores,
    classify_customer_segment,
    segment_customers,
    CustomerAnalytics,
)


@pytest.fixture
def synthetic_customer_transactions():
    """
    Synthetic dataset containing varying customer transaction patterns:
      - Customer 101: High frequency, recent, high spend (Champion)
      - Customer 102: Recent, 1 order, medium spend (New / Recent)
      - Customer 103: Old purchase, 1 order, low spend (Hibernating / Lost)
      - Customer 104: High frequency, high spend, old purchase (At Risk)
      - Customer 105: Moderate frequency, moderate recency, moderate spend (Loyal)
    """
    records = [
        # Customer 101 (Recent: 2011-12-08, 4 orders, total spend £1200)
        {"customer_id": 101, "invoice": "INV01", "invoice_date": "2011-10-01 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 101, "invoice": "INV02", "invoice_date": "2011-11-01 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 101, "invoice": "INV03", "invoice_date": "2011-12-01 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 101, "invoice": "INV04", "invoice_date": "2011-12-08 10:00:00", "line_total": 300.0, "is_cancellation": False},
        
        # Customer 102 (Recent: 2011-12-07, 1 order, total spend £150)
        {"customer_id": 102, "invoice": "INV05", "invoice_date": "2011-12-07 11:00:00", "line_total": 150.0, "is_cancellation": False},

        # Customer 103 (Old: 2010-01-15, 1 order, total spend £25)
        {"customer_id": 103, "invoice": "INV06", "invoice_date": "2010-01-15 12:00:00", "line_total": 25.0, "is_cancellation": False},

        # Customer 104 (Old: 2010-06-10, 5 orders, total spend £1500)
        {"customer_id": 104, "invoice": "INV07", "invoice_date": "2010-02-10 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 104, "invoice": "INV08", "invoice_date": "2010-03-10 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 104, "invoice": "INV09", "invoice_date": "2010-04-10 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 104, "invoice": "INV10", "invoice_date": "2010-05-10 10:00:00", "line_total": 300.0, "is_cancellation": False},
        {"customer_id": 104, "invoice": "INV11", "invoice_date": "2010-06-10 10:00:00", "line_total": 300.0, "is_cancellation": False},

        # Customer 105 (Moderate: 2011-09-01, 3 orders, total spend £600)
        {"customer_id": 105, "invoice": "INV12", "invoice_date": "2011-05-01 10:00:00", "line_total": 200.0, "is_cancellation": False},
        {"customer_id": 105, "invoice": "INV13", "invoice_date": "2011-07-01 10:00:00", "line_total": 200.0, "is_cancellation": False},
        {"customer_id": 105, "invoice": "INV14", "invoice_date": "2011-09-01 10:00:00", "line_total": 200.0, "is_cancellation": False},
    ]
    df = pd.DataFrame(records)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_calculate_rfm_metrics(synthetic_customer_transactions):
    snapshot_date = "2011-12-10"
    rfm = calculate_rfm_metrics(synthetic_customer_transactions, snapshot_date=snapshot_date)

    assert len(rfm) == 5
    assert set(rfm.columns) >= {"customer_id", "recency", "frequency", "monetary"}

    c101 = rfm[rfm["customer_id"] == 101].iloc[0]
    assert c101["frequency"] == 4
    assert c101["monetary"] == 1200.0
    assert c101["recency"] == 2  # Dec 10 minus Dec 08 = 2 days

    c104 = rfm[rfm["customer_id"] == 104].iloc[0]
    assert c104["frequency"] == 5
    assert c104["monetary"] == 1500.0
    assert c104["recency"] > 500  # 2010 to late 2011


def test_assign_rfm_scores(synthetic_customer_transactions):
    raw_rfm = calculate_rfm_metrics(synthetic_customer_transactions, snapshot_date="2011-12-10")
    scored = assign_rfm_scores(raw_rfm, r_bins=5, f_bins=5, m_bins=5)

    assert "r_score" in scored.columns
    assert "f_score" in scored.columns
    assert "m_score" in scored.columns
    assert "rfm_score_str" in scored.columns

    # Verify score ranges [1, 5]
    assert scored["r_score"].between(1, 5).all()
    assert scored["f_score"].between(1, 5).all()
    assert scored["m_score"].between(1, 5).all()

    # Most recent customer should receive highest r_score
    min_recency_id = scored.loc[scored["recency"].idxmin(), "customer_id"]
    assert scored.loc[scored["customer_id"] == min_recency_id, "r_score"].values[0] == 5


def test_segment_classification():
    # R=5, F=5, M=5 -> Champions
    assert classify_customer_segment(5, 5, 5) == "Champions"
    # R=4, F=1, M=2 -> New / Recent Customers
    assert classify_customer_segment(4, 1, 2) == "New / Recent Customers"
    # R=1, F=4, M=5 -> Can't Lose Them
    assert classify_customer_segment(1, 4, 5) == "Can't Lose Them"
    # R=1, F=1, M=1 -> Hibernating / Lost
    assert classify_customer_segment(1, 1, 1) == "Hibernating / Lost"


def test_customer_analytics_pipeline(synthetic_customer_transactions):
    analytics = CustomerAnalytics(synthetic_customer_transactions)
    segmented, summary = analytics.run_rfm_analysis(snapshot_date="2011-12-10")

    assert len(segmented) == 5
    assert "segment" in segmented.columns
    assert summary["total_customers"] == 5
    assert summary["total_revenue"] == 3475.0

    seg_table = summary["segment_table"]
    assert not seg_table.empty
    assert "customer_share_pct" in seg_table.columns
    assert "revenue_share_pct" in seg_table.columns
    assert round(seg_table["customer_share_pct"].sum(), 0) == 100.0


def test_inter_period_transitions(synthetic_customer_transactions):
    analytics = CustomerAnalytics(synthetic_customer_transactions)
    # Split using midpoint 2011-01-01
    res = analytics.compute_inter_period_transitions(cutoff_date="2011-01-01")

    assert res["period_1_customers"] > 0
    assert res["period_2_customers"] > 0
    assert "transition_matrix" in res
