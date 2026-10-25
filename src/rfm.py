"""
Customer Analytics and RFM Segmentation Module.

Implements quantitative Recency, Frequency, and Monetary (RFM) modeling:
  - Reference date determination (or explicit snapshot date).
  - Recency calculation (days since last purchase).
  - Frequency calculation (count of distinct purchase orders).
  - Monetary calculation (total monetary value across orders).
  - Quantile / rank-based scoring (R_Score, F_Score, M_Score on 1-5 scale).
    Note: Lower recency -> higher score (5). Higher frequency/monetary -> higher score (5).
  - Business segment classification (Champions, Loyal Customers, Potential Loyalists,
    Recent Customers, Promising, Need Attention, About To Sleep, At Risk, Can't Lose Them, Hibernating).
  - Segment profile summaries (customer counts, total revenue, average order value, average recency).
  - Transition / snapshot matrix tracking between multi-year observation periods.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Canonical RFM segment definitions and criteria based on R_Score and F_Score/M_Score
RFM_SEGMENT_RULES = [
    {
        "segment": "Champions",
        "description": "Bought recently, buy often, and spend the most.",
        "condition": lambda r, f, m: (r >= 4) and (f >= 4),
    },
    {
        "segment": "Loyal Customers",
        "description": "Buy on a regular basis, responsive to promotions.",
        "condition": lambda r, f, m: (r >= 3) and (f >= 3),
    },
    {
        "segment": "Potential Loyalists",
        "description": "Recent buyers with average frequency, high potential.",
        "condition": lambda r, f, m: (r >= 4) and (f >= 2),
    },
    {
        "segment": "New / Recent Customers",
        "description": "Bought most recently, but not frequent buyers yet.",
        "condition": lambda r, f, m: (r >= 4) and (f == 1),
    },
    {
        "segment": "Promising",
        "description": "Recent shoppers, but haven't spent much.",
        "condition": lambda r, f, m: (r == 3) and (f <= 2),
    },
    {
        "segment": "Need Attention",
        "description": "Above average recency, frequency, and monetary values; may slip away.",
        "condition": lambda r, f, m: (r == 3) and (f >= 3),
    },
    {
        "segment": "About To Sleep",
        "description": "Below average recency and frequency. Will be lost without re-activation.",
        "condition": lambda r, f, m: (r == 2) and (f <= 2),
    },
    {
        "segment": "Can't Lose Them",
        "description": "Made huge purchases and frequent orders, but have not returned for a long time.",
        "condition": lambda r, f, m: (r == 1) and (f >= 4),
    },
    {
        "segment": "At Risk",
        "description": "Purchased often and spent big in the past, but long time since last purchase.",
        "condition": lambda r, f, m: (r <= 2) and (f >= 3),
    },
    {
        "segment": "Hibernating / Lost",
        "description": "Low frequency, low spend, and last purchase was a long time ago.",
        "condition": lambda r, f, m: True,  # Fallback default
    },
]


def calculate_rfm_metrics(
    df: pd.DataFrame,
    snapshot_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    customer_col: str = "customer_id",
    invoice_col: str = "invoice",
    date_col: str = "invoice_date",
    amount_col: str = "line_total",
) -> pd.DataFrame:
    """
    Calculate raw Recency, Frequency, and Monetary metrics for each customer.
    
    Args:
        df: Customer transactions dataframe (customer_id not null, gross or net).
        snapshot_date: Reference analysis date. If None, defaults to max(date_col) + 1 day.
        customer_col: Customer identifier column.
        invoice_col: Invoice/order identifier column.
        date_col: Date/time column of transactions.
        amount_col: Monetary value column.
        
    Returns:
        DataFrame with columns: [customer_id, recency, frequency, monetary]
    """
    # Ensure dataframe is not empty and customer_id is valid
    valid_df = df[df[customer_col].notnull()].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=[customer_col, "recency", "frequency", "monetary"])

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(valid_df[date_col]):
        valid_df[date_col] = pd.to_datetime(valid_df[date_col], errors="coerce")

    # Determine snapshot date
    if snapshot_date is None:
        ref_date = valid_df[date_col].max().normalize() + pd.Timedelta(days=1)
    else:
        ref_date = pd.to_datetime(snapshot_date)

    # Filter out cancellations for frequency (frequency = distinct completed orders)
    # If is_cancellation column exists, consider non-cancelled orders for frequency
    order_mask = ~valid_df["is_cancellation"] if "is_cancellation" in valid_df.columns else True
    orders_df = valid_df[order_mask]

    rfm = orders_df.groupby(customer_col).agg(
        last_purchase_date=(date_col, "max"),
        first_purchase_date=(date_col, "min"),
        frequency=(invoice_col, "nunique"),
        monetary=(amount_col, "sum"),
    ).reset_index()

    # Recency in calendar days
    rfm["recency"] = ((ref_date - rfm["last_purchase_date"]).dt.total_seconds() / 86400.0).apply(np.ceil).astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)

    return rfm[[customer_col, "recency", "frequency", "monetary", "first_purchase_date", "last_purchase_date"]]


def assign_rfm_scores(
    rfm_df: pd.DataFrame,
    r_bins: int = 5,
    f_bins: int = 5,
    m_bins: int = 5,
) -> pd.DataFrame:
    """
    Assign quintile scores (1 to 5) for Recency, Frequency, and Monetary metrics.
    
    - Recency: Lower days -> Higher score (inverted rank).
    - Frequency: Higher orders -> Higher score.
    - Monetary: Higher spend -> Higher score.
    
    Uses pd.qcut with rank method 'first' to gracefully handle tied values / repeated frequencies.
    """
    df = rfm_df.copy()
    if df.empty:
        return df

    # 1. Recency Score (1 = most distant, 5 = most recent)
    # Inverted: labels 5 down to 1
    df["r_score"] = pd.qcut(
        df["recency"].rank(method="first"),
        q=r_bins,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    # 2. Frequency Score (1 = least frequent, 5 = most frequent)
    df["f_score"] = pd.qcut(
        df["frequency"].rank(method="first"),
        q=f_bins,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # 3. Monetary Score (1 = lowest spend, 5 = highest spend)
    df["m_score"] = pd.qcut(
        df["monetary"].rank(method="first"),
        q=m_bins,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Combined composite strings and average score
    df["rfm_score_str"] = (
        df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
    )
    df["rfm_score_mean"] = ((df["r_score"] + df["f_score"] + df["m_score"]) / 3.0).round(2)

    return df


def classify_customer_segment(r: int, f: int, m: int) -> str:
    """
    Classify a customer based on R, F, and M quintile scores into a named business segment.
    """
    for rule in RFM_SEGMENT_RULES:
        if rule["condition"](r, f, m):
            return rule["segment"]
    return "Hibernating / Lost"


def segment_customers(rfm_scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based segmentation to scored RFM DataFrame.
    
    Returns:
        DataFrame augmented with 'segment' column.
    """
    df = rfm_scored_df.copy()
    if df.empty:
        df["segment"] = pd.Series(dtype=str)
        return df

    df["segment"] = [
        classify_customer_segment(row["r_score"], row["f_score"], row["m_score"])
        for _, row in df.iterrows()
    ]
    return df


class CustomerAnalytics:
    """
    High-level customer intelligence and RFM segmentation orchestrator.
    """

    def __init__(self, transactions_df: pd.DataFrame):
        """
        Initialize with cleaned customer transactions.
        """
        self.df = transactions_df.copy()

    def run_rfm_analysis(
        self,
        snapshot_date: Optional[Union[str, datetime]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full RFM pipeline: metrics, scoring, and segmentation.
        
        Returns:
            Tuple of (rfm_segmented_df, summary_dict)
        """
        raw_rfm = calculate_rfm_metrics(self.df, snapshot_date=snapshot_date)
        scored_rfm = assign_rfm_scores(raw_rfm)
        segmented_rfm = segment_customers(scored_rfm)

        summary = self.compute_segment_profiles(segmented_rfm)
        return segmented_rfm, summary

    def compute_segment_profiles(self, segmented_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary profile statistics for each RFM segment:
          - Customer count and percentage
          - Total monetary revenue and revenue share percentage
          - Average recency, average frequency, and average monetary value
        """
        if segmented_df.empty:
            return {"total_customers": 0, "segment_table": pd.DataFrame()}

        total_customers = len(segmented_df)
        total_revenue = segmented_df["monetary"].sum()

        grouped = segmented_df.groupby("segment").agg(
            customer_count=("customer_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            median_recency=("recency", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
        ).reset_index()

        grouped["customer_share_pct"] = (grouped["customer_count"] / total_customers * 100).round(2)
        grouped["revenue_share_pct"] = (grouped["total_revenue"] / total_revenue * 100).round(2) if total_revenue > 0 else 0.0
        grouped["avg_recency"] = grouped["avg_recency"].round(1)
        grouped["avg_frequency"] = grouped["avg_frequency"].round(2)
        grouped["avg_monetary"] = grouped["avg_monetary"].round(2)
        grouped["total_revenue"] = grouped["total_revenue"].round(2)

        # Sort by total revenue descending
        grouped = grouped.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)

        return {
            "total_customers": total_customers,
            "total_revenue": round(total_revenue, 2),
            "segment_table": grouped,
        }

    def compute_inter_period_transitions(
        self,
        cutoff_date: Union[str, datetime] = "2010-12-09",
    ) -> Dict[str, Any]:
        """
        Analyze customer movement between Year 1 (2009-2010) and Year 2 (2010-2011).
        
        Args:
            cutoff_date: Boundary separating the two periods.
            
        Returns:
            Dictionary with period 1 segments, period 2 segments, and transition matrix cross-tabulation.
        """
        cutoff = pd.to_datetime(cutoff_date)
        df_p1 = self.df[self.df["invoice_date"] < cutoff]
        df_p2 = self.df[self.df["invoice_date"] >= cutoff]

        if df_p1.empty or df_p2.empty:
            return {"transition_matrix": pd.DataFrame(), "retained_customers": 0}

        p1_rfm, _ = CustomerAnalytics(df_p1).run_rfm_analysis(snapshot_date=cutoff)
        p2_rfm, _ = CustomerAnalytics(df_p2).run_rfm_analysis()

        merged = pd.merge(
            p1_rfm[["customer_id", "segment"]].rename(columns={"segment": "period_1_segment"}),
            p2_rfm[["customer_id", "segment"]].rename(columns={"segment": "period_2_segment"}),
            on="customer_id",
            how="inner",
        )

        transition_matrix = pd.crosstab(
            merged["period_1_segment"],
            merged["period_2_segment"],
            margins=True,
            margins_name="Total",
        )

        return {
            "period_1_customers": len(p1_rfm),
            "period_2_customers": len(p2_rfm),
            "active_in_both_periods": len(merged),
            "retention_rate_pct": round(len(merged) / len(p1_rfm) * 100, 2) if len(p1_rfm) > 0 else 0.0,
            "transition_matrix": transition_matrix,
        }
