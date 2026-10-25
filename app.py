"""
Interactive Streamlit Analytics Dashboard for E-Commerce Intelligence Platform.

Features Multi-Page Executive Analysis:
  - 1. Executive Summary & KPIs
  - 2. Exploratory Data Analysis (EDA) & Trends
  - 3. Customer Analytics & RFM Segmentation
  - 4. Cohort Retention Matrix
  - 5. Market Basket & Product Affinity
  - 6. Statistical Significance & Hypothesis Testing
  - 7. Predictive Repeat Purchase Engine
  - 8. Strategic Business Playbook & Recommendations
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ingestion import load_raw_dataset
from src.cleaning import execute_cleaning_pipeline
from src.eda import EDAEngine
from src.rfm import CustomerAnalytics
from src.cohort import CohortAnalysis
from src.market_basket import MarketBasketAnalysis
from src.statistical import StatisticalAnalysis
from src.predictive import RepeatPurchasePredictor
from src.business_insights import BusinessInsightsEngine


st.set_page_config(
    page_title="E-Commerce Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning transaction dataset...")
def get_cached_pipeline(sample_size: int = 15000):
    data_path = Path("data/raw/online_retail_II.xlsx")
    if not data_path.exists():
        return None, None, None
    raw_df, _ = load_raw_dataset(data_path, nrows=sample_size)
    df_clean, tiers, audit_stats = execute_cleaning_pipeline(raw_df)
    return df_clean, tiers, audit_stats


def main():
    st.sidebar.title("🛍️ Navigation")
    pages = [
        "Executive Overview",
        "EDA & Revenue Trends",
        "RFM Customer Segmentation",
        "Cohort & Retention Analysis",
        "Market Basket & Cross-Selling",
        "Statistical Hypothesis Testing",
        "Predictive Repeat Purchase",
        "Strategic Business Playbook",
    ]
    selected_page = st.sidebar.radio("Go to Module:", pages)

    st.sidebar.markdown("---")
    sample_choice = st.sidebar.select_slider(
        "Dataset Sample Size:",
        options=[5000, 10000, 20000, 50000, 100000],
        value=20000,
    )

    df_clean, tiers, audit_stats = get_cached_pipeline(sample_size=sample_choice)
    if df_clean is None:
        st.error("Dataset not found at `data/raw/online_retail_II.xlsx`. Please run `python scripts/download_dataset.py` first.")
        return

    gross_df = tiers["gross_sales"]
    customer_df = tiers["customer_transactions"]

    # =========================================================================
    # 1. Executive Overview
    # =========================================================================
    if selected_page == "Executive Overview":
        st.markdown('<div class="main-header">Executive Summary & Platform Health</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Holistic overview of customer transactions, revenue velocity, and accounting tiers.</div>', unsafe_allow_html=True)

        eda = EDAEngine(df_clean)
        kpis = eda.compute_summary_kpis()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Revenue", f"£{kpis['total_gross_revenue']:,.2f}")
        c2.metric("Net Revenue", f"£{kpis['total_net_revenue']:,.2f}", delta=f"-£{kpis['total_refunds']:,.2f} refunds")
        c3.metric("Total Completed Orders", f"{kpis['total_orders']:,}")
        c4.metric("Average Order Value (AOV)", f"£{kpis['avg_order_value']:.2f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Data Partitioning & Scope")
            partition_data = pd.DataFrame({
                "Scope": ["Gross Sales", "Net Sales", "Customer Verified", "Cancellations / Returns"],
                "Records": [
                    audit_stats["gross_sales_count"],
                    audit_stats["net_sales_count"],
                    audit_stats["customer_transactions_count"],
                    audit_stats["cancellations_count"],
                ],
            })
            fig_part = px.bar(partition_data, x="Scope", y="Records", text="Records", color="Scope", color_discrete_sequence=px.colors.qualitative.Prism)
            fig_part.update_layout(showlegend=False)
            st.plotly_chart(fig_part, use_container_width=True)

        with col_right:
            st.subheader("Monthly Revenue Trajectory")
            monthly_trends = eda.compute_monthly_trends()
            fig_traj = px.line(monthly_trends, x="year_month", y=["gross_revenue", "net_revenue"], markers=True, title="Gross vs Net Revenue Over Time")
            st.plotly_chart(fig_traj, use_container_width=True)

    # =========================================================================
    # 2. EDA & Trends
    # =========================================================================
    elif selected_page == "EDA & Revenue Trends":
        st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        eda = EDAEngine(df_clean)

        tab1, tab2, tab3 = st.tabs(["Geographic Breakdown", "Pareto Analysis", "Temporal Dynamics"])
        with tab1:
            country_df = eda.compute_country_breakdown(top_n=10)
            hhi = eda.compute_market_concentration_hhi()
            st.info(f"**Herfindahl-Hirschman Index (HHI)**: {hhi['hhi_index']} ({hhi['concentration_level']}). UK Revenue Share: {hhi['uk_revenue_share_pct']}%.")
            fig_country = px.bar(country_df, x="country", y="gross_revenue", text="gross_revenue", color="revenue_share_pct", title="Top 10 Markets by Gross Spend")
            st.plotly_chart(fig_country, use_container_width=True)

        with tab2:
            pareto = eda.compute_product_pareto()
            st.metric("Catalog Pareto Concentration", f"{pareto['product_percentage_for_80_rev']}% of products drive 80% revenue")
            st.write(f"Total Unique Catalog Items: {pareto['total_products']:,} | Top revenue drivers: {pareto['products_for_80_pct_revenue']:,} SKUs.")

        with tab3:
            temporal = eda.compute_temporal_patterns()
            col_d, col_h = st.columns(2)
            with col_d:
                fig_dow = px.bar(temporal["by_day_of_week"], x="day_of_week", y="gross_revenue", title="Sales by Day of Week")
                st.plotly_chart(fig_dow, use_container_width=True)
            with col_h:
                fig_hour = px.line(temporal["by_hour"], x="hour", y="gross_revenue", markers=True, title="Sales by Hour of Day")
                st.plotly_chart(fig_hour, use_container_width=True)

    # =========================================================================
    # 3. RFM Customer Segmentation
    # =========================================================================
    elif selected_page == "RFM Customer Segmentation":
        st.markdown('<div class="main-header">Customer RFM Segmentation</div>', unsafe_allow_html=True)
        analytics = CustomerAnalytics(customer_df)
        segmented, summary = analytics.run_rfm_analysis()

        seg_table = summary["segment_table"]
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie = px.pie(seg_table, names="segment", values="customer_count", title="Customer Share by Segment", hole=0.35)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_rev = px.bar(seg_table, x="segment", y="total_revenue", text="revenue_share_pct", color="segment", title="Revenue Contribution (£) by Segment")
            st.plotly_chart(fig_rev, use_container_width=True)

        st.subheader("Segment Portfolio Summary")
        st.dataframe(seg_table[["segment", "customer_count", "customer_share_pct", "total_revenue", "revenue_share_pct", "avg_recency", "avg_frequency", "avg_monetary"]], use_container_width=True)

    # =========================================================================
    # 4. Cohort & Retention Analysis
    # =========================================================================
    elif selected_page == "Cohort & Retention Analysis":
        st.markdown('<div class="main-header">Cohort & Retention Analysis</div>', unsafe_allow_html=True)
        cohort = CohortAnalysis(customer_df)
        counts, rates = cohort.compute_retention_matrix()
        summary = cohort.get_summary_metrics()

        col1, col2 = st.columns(2)
        col1.metric("Average Month 1 Retention", f"{summary['avg_month_1_retention_pct']}%")
        col2.metric("Average Month 3 Retention", f"{summary['avg_month_3_retention_pct']}%")

        if not rates.empty:
            st.subheader("Monthly Retention Heatmap (%)")
            fig_heat = px.imshow(
                rates,
                labels=dict(x="Cohort Period (Months)", y="Acquisition Cohort", color="Retention %"),
                color_continuous_scale="Viridis",
                text_auto=".1f",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # =========================================================================
    # 5. Market Basket & Cross-Selling
    # =========================================================================
    elif selected_page == "Market Basket & Cross-Selling":
        st.markdown('<div class="main-header">Market Basket Analysis & Product Affinity</div>', unsafe_allow_html=True)
        mba = MarketBasketAnalysis(gross_df)

        col_s, col_c = st.columns(2)
        min_supp = col_s.slider("Minimum Support:", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
        min_conf = col_c.slider("Minimum Confidence:", min_value=0.1, max_value=0.8, value=0.2, step=0.05)

        rules = mba.generate_association_rules(min_support=min_supp, min_confidence=min_conf, min_lift=1.2)
        st.write(f"Mined **{len(rules)} association rules** meeting specified thresholds.")

        if not rules.empty:
            fig_scatter = px.scatter(
                rules,
                x="support",
                y="confidence",
                size="lift",
                color="lift",
                hover_data=["antecedent_desc", "consequent_desc"],
                title="Association Rules: Support vs. Confidence (Size = Lift)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.dataframe(rules[["antecedent_desc", "consequent_desc", "support", "confidence", "lift", "leverage"]], use_container_width=True)

    # =========================================================================
    # 6. Statistical Hypothesis Testing
    # =========================================================================
    elif selected_page == "Statistical Hypothesis Testing":
        st.markdown('<div class="main-header">Statistical Analysis & Hypothesis Testing</div>', unsafe_allow_html=True)
        stats_engine = StatisticalAnalysis(df_clean)

        res_aov = stats_engine.compare_uk_vs_international_aov()
        st.subheader("1. UK vs. International Average Order Value (Welch's t-test)")
        c1, c2, c3 = st.columns(3)
        c1.metric("UK Mean AOV", f"£{res_aov.get('uk_mean_aov', 0):.2f}")
        c2.metric("Intl Mean AOV", f"£{res_aov.get('intl_mean_aov', 0):.2f}")
        c3.metric("Cohen's d Effect Size", f"{res_aov.get('cohens_d', 0):.3f}")

        st.write(f"**Welch's t-statistic**: `{res_aov.get('t_statistic')}` | **p-value**: `{res_aov.get('t_p_value'):.4e}` | **Statistically Significant**: `{res_aov.get('is_significant_p05')}`")
        st.info(f"**Interpretation**: {res_aov.get('practical_interpretation')}")

        st.markdown("---")
        st.subheader("2. Cancellation Propensity by Geography (Chi-Square Test)")
        res_chi = stats_engine.test_cancellation_propensity_by_geography()
        st.write(f"UK Return Rate: `{res_chi.get('uk_cancellation_rate_pct')}%` | Intl Return Rate: `{res_chi.get('intl_cancellation_rate_pct')}%`")
        st.write(f"**Chi-Square Statistic**: `{res_chi.get('chi2_statistic')}` | **p-value**: `{res_chi.get('p_value'):.4e}`")

    # =========================================================================
    # 7. Predictive Repeat Purchase
    # =========================================================================
    elif selected_page == "Predictive Repeat Purchase":
        st.markdown('<div class="main-header">Predictive Repeat Purchase Modeling</div>', unsafe_allow_html=True)
        predictor = RepeatPurchasePredictor(customer_df)

        try:
            X, y = predictor.build_temporal_dataset(cutoff_date="2011-06-01", prediction_window_days=90)
            eval_res = predictor.train_and_evaluate(X, y)

            st.write(f"Observation Cutoff: **2011-06-01** | Window: **90 Days** | Base Repeat Rate: **{eval_res['repeat_rate_base']}%**")
            st.success(f"Best Performing Architecture: **{eval_res['best_model']}**")

            metrics_df = pd.DataFrame(eval_res["metrics"]).T
            st.dataframe(metrics_df[["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score"]], use_container_width=True)

            st.subheader("Random Forest Feature Importances")
            fi_df = pd.DataFrame(list(eval_res["feature_importances"].items()), columns=["Feature", "Importance"])
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance", title="Repeat Purchase Predictive Drivers")
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Predictive dataset configuration error: {e}. Try increasing the sample size.")

    # =========================================================================
    # 8. Strategic Business Playbook
    # =========================================================================
    elif selected_page == "Strategic Business Playbook":
        st.markdown('<div class="main-header">Boardroom Business Playbook & Recommendations</div>', unsafe_allow_html=True)
        eda = EDAEngine(df_clean)
        analytics = CustomerAnalytics(customer_df)
        _, rfm_summary = analytics.run_rfm_analysis()
        cohort = CohortAnalysis(customer_df)
        mba = MarketBasketAnalysis(gross_df)
        rules = mba.generate_association_rules(min_support=0.01, min_confidence=0.2)
        stats_engine = StatisticalAnalysis(df_clean)
        predictor = RepeatPurchasePredictor(customer_df)
        try:
            X, y = predictor.build_temporal_dataset(cutoff_date="2011-06-01", prediction_window_days=90)
            pred_eval = predictor.train_and_evaluate(X, y)
        except Exception:
            pred_eval = {"best_model": "Random Forest"}

        insights_engine = BusinessInsightsEngine(
            eda_summary=eda.compute_market_concentration_hhi(),
            rfm_summary=rfm_summary,
            cohort_summary=cohort.get_summary_metrics(),
            market_basket_rules=rules,
            statistical_findings=stats_engine.compare_uk_vs_international_aov(),
            predictive_summary=pred_eval,
        )

        insights = insights_engine.generate_executive_insights()
        for idx, item in enumerate(insights, 1):
            with st.expander(f"{idx}. [{item['category']}] {item['title']}", expanded=True):
                st.markdown(f"**Diagnostic Observation**: {item['observation']}")
                st.markdown(f"**Strategic Recommendation**: {item['recommendation']}")
                st.markdown(f"**Projected Business Impact**: `{item['impact_potential']}`")


if __name__ == "__main__":
    main()
