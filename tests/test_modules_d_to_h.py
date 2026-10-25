"""
Comprehensive Unit Tests for Modules D through H:
  - Module D: Cohort & Retention Analysis (src/cohort.py)
  - Module E: Market Basket Analysis (src/market_basket.py)
  - Module F: Statistical Analysis & Hypothesis Testing (src/statistical.py)
  - Module G: Predictive Modeling (src/predictive.py)
  - Module H: Business Insights Engine (src/business_insights.py)
"""

import pytest
import pandas as pd
import numpy as np

from src.cohort import CohortAnalysis
from src.market_basket import MarketBasketAnalysis
from src.statistical import StatisticalAnalysis
from src.predictive import RepeatPurchasePredictor
from src.business_insights import BusinessInsightsEngine


@pytest.fixture
def rich_transaction_data():
    """Generates synthetic dataset for multi-period testing."""
    records = []
    # Customer 101: acquired Jan 2011, repeats Feb, Mar, Apr
    for d, inv in [("2011-01-10", "I1"), ("2011-02-12", "I2"), ("2011-03-15", "I3"), ("2011-04-18", "I4")]:
        records.append({"customer_id": 101, "invoice": inv, "invoice_date": d, "line_total": 120.0, "quantity": 10, "unit_price": 12.0, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_A", "description": "Product Alpha"})
        records.append({"customer_id": 101, "invoice": inv, "invoice_date": d, "line_total": 80.0, "quantity": 8, "unit_price": 10.0, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_B", "description": "Product Beta"})

    # Customer 102: acquired Jan 2011, repeats Mar
    records.append({"customer_id": 102, "invoice": "I5", "invoice_date": "2011-01-20", "line_total": 50.0, "quantity": 5, "unit_price": 10.0, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_A", "description": "Product Alpha"})
    records.append({"customer_id": 102, "invoice": "I6", "invoice_date": "2011-03-22", "line_total": 60.0, "quantity": 6, "unit_price": 10.0, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_C", "description": "Product Gamma"})

    # Customer 103: acquired Feb 2011, international, big orders
    records.append({"customer_id": 103, "invoice": "I7", "invoice_date": "2011-02-05", "line_total": 800.0, "quantity": 40, "unit_price": 20.0, "country": "Germany", "is_cancellation": False, "stock_code": "PROD_B", "description": "Product Beta"})
    records.append({"customer_id": 103, "invoice": "I8", "invoice_date": "2011-04-05", "line_total": 950.0, "quantity": 50, "unit_price": 19.0, "country": "Germany", "is_cancellation": False, "stock_code": "PROD_A", "description": "Product Alpha"})

    # Customer 104: acquired Feb 2011, with 1 cancellation
    records.append({"customer_id": 104, "invoice": "I9", "invoice_date": "2011-02-15", "line_total": 45.0, "quantity": 3, "unit_price": 15.0, "country": "France", "is_cancellation": False, "stock_code": "PROD_C", "description": "Product Gamma"})
    records.append({"customer_id": 104, "invoice": "C10", "invoice_date": "2011-02-16", "line_total": -45.0, "quantity": -3, "unit_price": 15.0, "country": "France", "is_cancellation": True, "stock_code": "PROD_C", "description": "Product Gamma"})

    # Additional customers for robust predictive modeling (both repeats and non-repeats)
    for cid in range(201, 215):
        # 7 repeat customers and 7 non-repeat customers
        records.append({"customer_id": cid, "invoice": f"I_pre_{cid}", "invoice_date": "2011-01-15", "line_total": 100.0, "quantity": 5, "unit_price": 20.0, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_A", "description": "Product Alpha"})
        if cid % 2 == 0:
            # Repeats in observation window or prediction window
            records.append({"customer_id": cid, "invoice": f"I_post_{cid}", "invoice_date": "2011-03-25", "line_total": 150.0, "quantity": 8, "unit_price": 18.75, "country": "United Kingdom", "is_cancellation": False, "stock_code": "PROD_B", "description": "Product Beta"})

    df = pd.DataFrame(records)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


# -------------------------------------------------------------
# Module D: Cohort Tests
# -------------------------------------------------------------

def test_cohort_analysis(rich_transaction_data):
    cohort = CohortAnalysis(rich_transaction_data)
    counts, rates = cohort.compute_retention_matrix()

    assert not counts.empty
    assert not rates.empty
    # Month 0 retention must always be 100%
    assert (rates[0] == 100.0).all()

    rev_matrix, arpu_matrix = cohort.compute_revenue_matrix()
    assert not rev_matrix.empty
    assert not arpu_matrix.empty

    metrics = cohort.get_summary_metrics()
    assert metrics["total_cohorts"] >= 2


# -------------------------------------------------------------
# Module E: Market Basket Tests
# -------------------------------------------------------------

def test_market_basket_analysis(rich_transaction_data):
    mba = MarketBasketAnalysis(rich_transaction_data)
    baskets = mba.get_baskets(min_basket_size=2)
    assert len(baskets) > 0

    rules = mba.generate_association_rules(min_support=0.1, min_confidence=0.2, min_lift=0.5)
    assert not rules.empty
    assert "antecedent" in rules.columns
    assert "lift" in rules.columns
    assert (rules["lift"] > 0).all()

    # Bundle recommendations
    recs = mba.get_bundle_recommendations_for_product("PROD_A", top_n=3, min_support=0.1)
    assert isinstance(recs, list)


# -------------------------------------------------------------
# Module F: Statistical Tests
# -------------------------------------------------------------

def test_statistical_analysis(rich_transaction_data):
    stats_engine = StatisticalAnalysis(rich_transaction_data)
    
    # Welch's t-test and Mann-Whitney
    aov_res = stats_engine.compare_uk_vs_international_aov()
    assert "t_statistic" in aov_res
    assert "mann_whitney_u" in aov_res
    assert "cohens_d" in aov_res

    # Chi-square test
    chi_res = stats_engine.test_cancellation_propensity_by_geography()
    assert "chi2_statistic" in chi_res
    assert "p_value" in chi_res

    # Confidence interval
    ci_res = stats_engine.compute_aov_confidence_intervals()
    assert ci_res["ci_lower"] <= ci_res["mean_aov"] <= ci_res["ci_upper"]


# -------------------------------------------------------------
# Module G: Predictive Modeling Tests
# -------------------------------------------------------------

def test_predictive_modeling(rich_transaction_data):
    predictor = RepeatPurchasePredictor(rich_transaction_data)
    # Cutoff at 2011-03-01, window of 60 days
    X, y = predictor.build_temporal_dataset(cutoff_date="2011-03-01", prediction_window_days=60)

    assert len(X) == len(y)
    assert "recency_days" in X.columns
    assert "frequency" in X.columns
    assert "total_spend" in X.columns

    # Train and evaluate
    eval_res = predictor.train_and_evaluate(X, y, test_size=0.33, random_state=42)
    assert "best_model" in eval_res
    assert "metrics" in eval_res
    assert "Random Forest" in eval_res["metrics"]
    assert "feature_importances" in eval_res


# -------------------------------------------------------------
# Module H: Business Insights Engine Tests
# -------------------------------------------------------------

def test_business_insights_engine(rich_transaction_data):
    mba = MarketBasketAnalysis(rich_transaction_data)
    rules = mba.generate_association_rules(min_support=0.1, min_confidence=0.2)

    insights_engine = BusinessInsightsEngine(
        eda_summary={"uk_revenue_share_pct": 92.5},
        rfm_summary={"segment_table": pd.DataFrame([
            {"segment": "Champions", "customer_count": 10, "revenue_share_pct": 45.0},
            {"segment": "At Risk", "customer_count": 5, "revenue_share_pct": 12.0},
        ])},
        cohort_summary={"avg_month_1_retention_pct": 28.4},
        market_basket_rules=rules,
        statistical_findings={"uk_mean_aov": 250.0, "intl_mean_aov": 850.0},
        predictive_summary={"best_model": "Random Forest"},
    )

    insights = insights_engine.generate_executive_insights()
    assert len(insights) >= 4
    for item in insights:
        assert "category" in item
        assert "recommendation" in item
        assert "impact_potential" in item

    report = insights_engine.generate_markdown_report()
    assert "# Strategic Business Intelligence" in report
