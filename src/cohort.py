"""
Cohort and Multi-Period Retention Analysis Module.

Performs customer acquisition cohort segmentation, tracking retention,
churn, repeat-purchase rates, and cumulative revenue matrices over time:
  - Monthly acquisition cohort assignment (first purchase month).
  - Cohort period index calculation (Month 0, Month 1, ... Month N).
  - Customer retention matrix (headcount and percentage).
  - Cohort revenue & average spend per active customer matrix.
  - Cohort churn rates and repeat order cadence.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class CohortAnalysis:
    """Cohort Analysis Engine for longitudinal customer retention and spending."""

    def __init__(self, transactions_df: pd.DataFrame):
        """
        Initialize with cleaned customer transactions.
        """
        valid_df = transactions_df[transactions_df["customer_id"].notnull()].copy()
        if "is_cancellation" in valid_df.columns:
            valid_df = valid_df[~valid_df["is_cancellation"]].copy()

        if "invoice_date" in valid_df.columns and not pd.api.types.is_datetime64_any_dtype(valid_df["invoice_date"]):
            valid_df["invoice_date"] = pd.to_datetime(valid_df["invoice_date"], errors="coerce")

        self.df = valid_df

    def prepare_cohort_data(self) -> pd.DataFrame:
        """
        Assign cohort month and period index to each transaction.
        """
        df = self.df.copy()
        if df.empty:
            return pd.DataFrame(columns=["customer_id", "invoice", "invoice_date", "cohort_month", "order_month", "cohort_index"])

        # Order month (period format YYYY-MM)
        df["order_month"] = df["invoice_date"].dt.to_period("M")

        # First purchase month per customer
        customer_cohort = df.groupby("customer_id")["order_month"].min().reset_index()
        customer_cohort.rename(columns={"order_month": "cohort_month"}, inplace=True)

        df = pd.merge(df, customer_cohort, on="customer_id", how="left")

        # Period index (month difference: 0 = acquisition month, 1 = month 1, ...)
        df["cohort_index"] = (
            (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
            + (df["order_month"].dt.month - df["cohort_month"].dt.month)
        )
        return df

    def compute_retention_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate cohort customer retention matrix.
        
        Returns:
            Tuple of:
              - retention_counts: Raw count of active customers per cohort & period.
              - retention_rates: Percentage retention relative to Month 0 (acquisition).
        """
        df_cohort = self.prepare_cohort_data()
        if df_cohort.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Group by cohort_month and cohort_index -> unique active customers
        cohort_group = df_cohort.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()

        retention_counts = cohort_group.pivot(
            index="cohort_month",
            columns="cohort_index",
            values="customer_id"
        ).fillna(0).astype(int)

        # Baseline cohort size (Month 0)
        cohort_sizes = retention_counts.iloc[:, 0]
        retention_rates = retention_counts.divide(cohort_sizes, axis=0).round(4) * 100.0

        # Convert period index to string for serialization
        retention_counts.index = retention_counts.index.astype(str)
        retention_rates.index = retention_rates.index.astype(str)

        return retention_counts, retention_rates

    def compute_revenue_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate total gross revenue and average revenue per active customer per cohort period.
        
        Returns:
            Tuple of:
              - total_revenue_matrix: Total spend per cohort month and cohort index.
              - arpu_matrix: Average revenue per active customer.
        """
        df_cohort = self.prepare_cohort_data()
        if df_cohort.empty:
            return pd.DataFrame(), pd.DataFrame()

        rev_group = df_cohort.groupby(["cohort_month", "cohort_index"]).agg(
            total_revenue=("line_total", "sum"),
            active_customers=("customer_id", "nunique"),
        ).reset_index()

        rev_group["arpu"] = (rev_group["total_revenue"] / rev_group["active_customers"]).round(2)

        total_revenue_matrix = rev_group.pivot(
            index="cohort_month",
            columns="cohort_index",
            values="total_revenue"
        ).fillna(0.0).round(2)

        arpu_matrix = rev_group.pivot(
            index="cohort_month",
            columns="cohort_index",
            values="arpu"
        ).fillna(0.0).round(2)

        total_revenue_matrix.index = total_revenue_matrix.index.astype(str)
        arpu_matrix.index = arpu_matrix.index.astype(str)

        return total_revenue_matrix, arpu_matrix

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Compute headline cohort KPIs: average month-1 retention, total cohorts, acquisition rate.
        """
        counts, rates = self.compute_retention_matrix()
        if rates.empty or rates.shape[1] < 2:
            return {
                "total_cohorts": len(rates),
                "avg_month_1_retention_pct": 0.0,
                "avg_month_3_retention_pct": 0.0,
            }

        m1_rates = rates[1].dropna() if 1 in rates.columns else pd.Series(dtype=float)
        m3_rates = rates[3].dropna() if 3 in rates.columns else pd.Series(dtype=float)

        return {
            "total_cohorts": len(rates),
            "avg_month_1_retention_pct": round(float(m1_rates.mean()), 2) if not m1_rates.empty else 0.0,
            "avg_month_3_retention_pct": round(float(m3_rates.mean()), 2) if not m3_rates.empty else 0.0,
            "cohort_sizes": counts[0].to_dict() if 0 in counts.columns else {},
        }
