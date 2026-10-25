"""
Exploratory Data Analysis (EDA) Engine for E-commerce Intelligence Platform.

Computes statistical distributions, business aggregations, and visual analytical summaries:
  - Monthly and Weekly revenue trends (Gross vs Net sales, refunds, volume)
  - International market & country comparisons (Revenue concentration, HHI)
  - Product sales distributions, Pareto (80/20) concentration, top/bottom performers
  - Unit price dynamics, elasticity proxies, and basket size statistics
  - Hourly and day-of-week shopping patterns
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class EDAEngine:
    """EDA computation and metrics calculation engine."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a standardized cleaned transactions dataframe (or gross/net slice).
        """
        self.df = df.copy()
        if "invoice_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["invoice_date"]):
            self.df["invoice_date"] = pd.to_datetime(self.df["invoice_date"], errors="coerce")

    def compute_summary_kpis(self) -> Dict[str, Any]:
        """Compute headline business KPIs from transactions."""
        gross_mask = (self.df["unit_price"] > 0) & (self.df["quantity"] > 0) & (~self.df["is_cancellation"])
        gross_sales = self.df[gross_mask]

        cancel_mask = self.df["is_cancellation"] | (self.df["quantity"] < 0)
        cancellations = self.df[cancel_mask]

        total_gross_revenue = float(gross_sales["line_total"].sum())
        total_refunds = float(cancellations["line_total"].abs().sum())
        total_net_revenue = float(self.df["line_total"].sum())
        total_orders = int(gross_sales["invoice"].nunique())
        total_units_sold = int(gross_sales["quantity"].sum())
        unique_customers = int(self.df["customer_id"].dropna().nunique())
        unique_products = int(gross_sales["stock_code"].nunique())
        avg_order_value = round(total_gross_revenue / total_orders, 2) if total_orders > 0 else 0.0

        return {
            "total_gross_revenue": round(total_gross_revenue, 2),
            "total_net_revenue": round(total_net_revenue, 2),
            "total_refunds": round(total_refunds, 2),
            "refund_ratio_pct": round((total_refunds / total_gross_revenue * 100), 2) if total_gross_revenue > 0 else 0.0,
            "total_orders": total_orders,
            "total_units_sold": total_units_sold,
            "unique_customers": unique_customers,
            "unique_products": unique_products,
            "avg_order_value": avg_order_value,
        }

    def compute_monthly_trends(self) -> pd.DataFrame:
        """Monthly aggregation with gross, net, refund, orders, and average basket value."""
        df = self.df.copy()
        if "year_month" not in df.columns:
            df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)

        monthly = df.groupby("year_month").apply(
            lambda g: pd.Series({
                "gross_revenue": g.loc[(g["quantity"] > 0) & (~g["is_cancellation"]), "line_total"].sum(),
                "net_revenue": g["line_total"].sum(),
                "refund_amount": g.loc[g["is_cancellation"], "line_total"].abs().sum(),
                "orders_count": g.loc[~g["is_cancellation"], "invoice"].nunique(),
                "units_sold": g.loc[g["quantity"] > 0, "quantity"].sum(),
                "active_customers": g["customer_id"].dropna().nunique(),
            }),
            include_groups=False,
        ).reset_index()

        monthly["aov"] = (monthly["gross_revenue"] / monthly["orders_count"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        monthly["gross_revenue"] = monthly["gross_revenue"].round(2)
        monthly["net_revenue"] = monthly["net_revenue"].round(2)
        monthly["refund_amount"] = monthly["refund_amount"].round(2)
        return monthly.sort_values("year_month").reset_index(drop=True)

    def compute_weekly_trends(self) -> pd.DataFrame:
        """Weekly trends for identifying short-term seasonal surges (e.g. Q4 holiday peaks)."""
        df = self.df.copy()
        df["year_week"] = df["invoice_date"].dt.strftime("%Y-W%U")

        weekly = df.groupby("year_week").apply(
            lambda g: pd.Series({
                "gross_revenue": g.loc[(g["quantity"] > 0) & (~g["is_cancellation"]), "line_total"].sum(),
                "net_revenue": g["line_total"].sum(),
                "orders_count": g.loc[~g["is_cancellation"], "invoice"].nunique(),
                "units_sold": g.loc[g["quantity"] > 0, "quantity"].sum(),
            }),
            include_groups=False,
        ).reset_index()

        weekly["gross_revenue"] = weekly["gross_revenue"].round(2)
        weekly["net_revenue"] = weekly["net_revenue"].round(2)
        return weekly.sort_values("year_week").reset_index(drop=True)

    def compute_country_breakdown(self, top_n: int = 15) -> pd.DataFrame:
        """
        Country-level distribution with gross revenue, customer share, and domestic vs international split.
        """
        gross_mask = (self.df["quantity"] > 0) & (~self.df["is_cancellation"])
        df_gross = self.df[gross_mask]

        country_stats = df_gross.groupby("country").agg(
            gross_revenue=("line_total", "sum"),
            order_count=("invoice", "nunique"),
            customer_count=("customer_id", lambda x: x.dropna().nunique()),
            units_sold=("quantity", "sum"),
        ).reset_index()

        total_rev = country_stats["gross_revenue"].sum()
        country_stats["revenue_share_pct"] = (country_stats["gross_revenue"] / total_rev * 100).round(2)
        country_stats["aov"] = (country_stats["gross_revenue"] / country_stats["order_count"]).round(2)
        country_stats = country_stats.sort_values(by="gross_revenue", ascending=False).reset_index(drop=True)

        return country_stats.head(top_n)

    def compute_market_concentration_hhi(self) -> Dict[str, float]:
        """
        Compute Herfindahl-Hirschman Index (HHI) for country revenue concentration.
        HHI = sum(market_share_percentage ^ 2).
        HHI < 1500: Competitive / Diversified.
        HHI 1500 - 2500: Moderate concentration.
        HHI > 2500: Highly concentrated.
        """
        country_stats = self.compute_country_breakdown(top_n=100)
        shares = country_stats["revenue_share_pct"]
        hhi = float((shares ** 2).sum())
        uk_share = float(country_stats.loc[country_stats["country"] == "United Kingdom", "revenue_share_pct"].sum())
        non_uk_share = round(100.0 - uk_share, 2)

        return {
            "hhi_index": round(hhi, 2),
            "uk_revenue_share_pct": round(uk_share, 2),
            "international_revenue_share_pct": non_uk_share,
            "concentration_level": "Highly Concentrated" if hhi > 2500 else ("Moderate" if hhi >= 1500 else "Diversified"),
        }

    def compute_product_pareto(self) -> Dict[str, Any]:
        """
        Evaluate Pareto 80/20 principle: percentage of products generating 80% of gross revenue.
        """
        gross_mask = (self.df["quantity"] > 0) & (~self.df["is_cancellation"])
        df_gross = self.df[gross_mask]

        prod_rev = df_gross.groupby("stock_code")["line_total"].sum().sort_values(ascending=False).reset_index()
        total_rev = prod_rev["line_total"].sum()
        prod_rev["cumulative_rev"] = prod_rev["line_total"].cumsum()
        prod_rev["cumulative_share"] = prod_rev["cumulative_rev"] / total_rev

        top_80_count = int((prod_rev["cumulative_share"] <= 0.80).sum()) + 1
        total_prods = len(prod_rev)
        top_80_pct = round(top_80_count / total_prods * 100, 2) if total_prods > 0 else 0.0

        return {
            "total_products": total_prods,
            "products_for_80_pct_revenue": top_80_count,
            "product_percentage_for_80_rev": top_80_pct,
            "pareto_holds": top_80_pct <= 25.0,  # Typically <= 20-25%
        }

    def compute_unit_price_dynamics(self) -> Dict[str, Any]:
        """
        Analyze unit price distribution, quartiles, median, skewness, and extreme price outliers.
        """
        prices = self.df.loc[self.df["unit_price"] > 0, "unit_price"]

        return {
            "mean_unit_price": round(float(prices.mean()), 2),
            "median_unit_price": round(float(prices.median()), 2),
            "std_unit_price": round(float(prices.std()), 2),
            "min_unit_price": round(float(prices.min()), 2),
            "p25_unit_price": round(float(prices.quantile(0.25)), 2),
            "p75_unit_price": round(float(prices.quantile(0.75)), 2),
            "p95_unit_price": round(float(prices.quantile(0.95)), 2),
            "p99_unit_price": round(float(prices.quantile(0.99)), 2),
            "max_unit_price": round(float(prices.max()), 2),
        }

    def compute_temporal_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze purchasing activity across day of the week and hour of the day.
        """
        gross_mask = (self.df["quantity"] > 0) & (~self.df["is_cancellation"])
        df_gross = self.df[gross_mask].copy()

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = df_gross.groupby("day_of_week").agg(
            order_count=("invoice", "nunique"),
            gross_revenue=("line_total", "sum"),
        ).reindex(day_order).fillna(0).reset_index()
        by_day["gross_revenue"] = by_day["gross_revenue"].round(2)

        by_hour = df_gross.groupby("hour").agg(
            order_count=("invoice", "nunique"),
            gross_revenue=("line_total", "sum"),
        ).reset_index().sort_values("hour")
        by_hour["gross_revenue"] = by_hour["gross_revenue"].round(2)

        return {
            "by_day_of_week": by_day,
            "by_hour": by_hour,
        }
