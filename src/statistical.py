"""
Statistical Analysis and Hypothesis Testing Module (Module F).

Conducts rigorous hypothesis tests, computes effect sizes, and evaluates practical business significance:
  - Two-sample t-test / Welch's t-test for revenue difference between markets (e.g. UK vs Non-UK).
  - Mann-Whitney U test for non-parametric evaluation of skewed order spend distributions.
  - Chi-square test of independence for cancellation propensity across geographic regions.
  - Cohen's d effect size calculations.
  - Confidence interval estimation for Average Order Value (AOV).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnalysis:
    """Hypothesis testing and statistical inference engine."""

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()

    def compare_uk_vs_international_aov(self) -> Dict[str, Any]:
        """
        Test whether UK orders differ significantly in order value from International orders.
        Applies both Welch's t-test (parametric) and Mann-Whitney U test (non-parametric for skewness).
        """
        # Aggregate to order (invoice) level
        valid_orders = self.df.loc[
            (~self.df["is_cancellation"]) & (self.df["line_total"] > 0)
        ].groupby(["invoice", "country"])["line_total"].sum().reset_index()

        uk_orders = valid_orders.loc[valid_orders["country"] == "United Kingdom", "line_total"].values
        intl_orders = valid_orders.loc[valid_orders["country"] != "United Kingdom", "line_total"].values

        if len(uk_orders) < 2 or len(intl_orders) < 2:
            return {"error": "Insufficient sample size for comparison."}

        # Welch's t-test
        t_stat, t_pval = stats.ttest_ind(uk_orders, intl_orders, equal_var=False)

        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(uk_orders, intl_orders, alternative="two-sided")

        # Cohen's d effect size
        n1, n2 = len(uk_orders), len(intl_orders)
        s1, s2 = np.var(uk_orders, ddof=1), np.var(intl_orders, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
        cohens_d = (np.mean(intl_orders) - np.mean(uk_orders)) / pooled_std if pooled_std > 0 else 0.0

        return {
            "test_name": "UK vs. International Order Value Comparison",
            "uk_order_count": int(n1),
            "intl_order_count": int(n2),
            "uk_mean_aov": round(float(np.mean(uk_orders)), 2),
            "intl_mean_aov": round(float(np.mean(intl_orders)), 2),
            "uk_median_aov": round(float(np.median(uk_orders)), 2),
            "intl_median_aov": round(float(np.median(intl_orders)), 2),
            "t_statistic": round(float(t_stat), 4),
            "t_p_value": float(t_pval),
            "mann_whitney_u": round(float(u_stat), 2),
            "mann_whitney_p_value": float(u_pval),
            "cohens_d": round(float(cohens_d), 4),
            "is_significant_p05": bool(t_pval < 0.05),
            "practical_interpretation": (
                "International orders exhibit significantly higher AOV with meaningful effect size."
                if (t_pval < 0.05 and abs(cohens_d) > 0.2)
                else "Difference is either statistically negligible or of small practical magnitude."
            ),
        }

    def test_cancellation_propensity_by_geography(self) -> Dict[str, Any]:
        """
        Chi-square contingency test: Is cancellation probability dependent on geography (UK vs International)?
        """
        # Order-level cancellation status
        order_status = self.df.groupby("invoice").agg(
            is_cancel=("is_cancellation", "max"),
            country=("country", "first"),
        ).reset_index()

        order_status["is_uk"] = order_status["country"] == "United Kingdom"
        order_status["region"] = order_status["is_uk"].map({True: "UK", False: "International"})
        order_status["status"] = order_status["is_cancel"].apply(lambda x: "Cancelled" if bool(x) else "Completed")

        contingency_table = pd.crosstab(
            order_status["region"],
            order_status["status"],
        )

        rates = order_status.groupby("is_uk")["is_cancel"].mean().to_dict()
        uk_cancel_rate = round(float(rates.get(True, 0.0)) * 100, 2)
        intl_cancel_rate = round(float(rates.get(False, 0.0)) * 100, 2)

        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return {
                "test_name": "Chi-Square Test of Cancellation Independence",
                "chi2_statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "uk_cancellation_rate_pct": uk_cancel_rate,
                "intl_cancellation_rate_pct": intl_cancel_rate,
                "is_significant_p05": False,
                "note": "Contingency table had insufficient categories for Chi-square computation.",
            }

        chi2, pval, dof, _ = stats.chi2_contingency(contingency_table)

        return {
            "test_name": "Chi-Square Test of Cancellation Independence",
            "chi2_statistic": round(float(chi2), 4),
            "p_value": float(pval),
            "degrees_of_freedom": int(dof),
            "uk_cancellation_rate_pct": uk_cancel_rate,
            "intl_cancellation_rate_pct": intl_cancel_rate,
            "is_significant_p05": bool(pval < 0.05),
        }

    def compute_aov_confidence_intervals(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute parametric confidence interval for the portfolio Average Order Value.
        """
        order_totals = self.df.loc[
            (~self.df["is_cancellation"]) & (self.df["line_total"] > 0)
        ].groupby("invoice")["line_total"].sum().values

        n = len(order_totals)
        if n < 2:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        mean = float(np.mean(order_totals))
        sem = stats.sem(order_totals)
        margin = sem * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)

        return {
            "sample_size": n,
            "mean_aov": round(mean, 2),
            "confidence_level": confidence,
            "ci_lower": round(max(0.0, mean - margin), 2),
            "ci_upper": round(mean + margin, 2),
            "standard_error": round(float(sem), 3),
        }
