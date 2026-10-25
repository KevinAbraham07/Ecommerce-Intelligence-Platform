"""
Analytical SQLite Database Module for E-commerce Intelligence Platform.

Constructs and manages an indexed SQLite analytical database with optimized relational tables:
  - `transactions`: Line-level transactional data with datetime, amounts, and flags.
  - `customers`: Customer summary metrics (lifetime gross, net, first/last purchase, order counts).
  - `products`: Product catalog attributes (first/last seen, total quantity sold, average unit price).

Provides analytical query execution wrappers for order aggregations and time-series trends.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sqlite3
import pandas as pd


DATABASE_SCHEMA_SQL = """
-- 1. Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL,
    invoice_date TIMESTAMP NOT NULL,
    unit_price REAL NOT NULL,
    line_total REAL NOT NULL,
    customer_id INTEGER,
    country TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    year_month TEXT,
    day_of_week TEXT,
    hour INTEGER,
    is_cancellation INTEGER DEFAULT 0,
    is_admin_code INTEGER DEFAULT 0,
    sheet_source TEXT
);

-- 2. Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    country TEXT,
    first_purchase_date TIMESTAMP,
    last_purchase_date TIMESTAMP,
    total_orders INTEGER DEFAULT 0,
    total_gross_revenue REAL DEFAULT 0.0,
    total_net_revenue REAL DEFAULT 0.0,
    total_items_purchased INTEGER DEFAULT 0,
    is_repeat_buyer INTEGER DEFAULT 0
);

-- 3. Products Table
CREATE TABLE IF NOT EXISTS products (
    stock_code TEXT PRIMARY KEY,
    description TEXT,
    first_sold_date TIMESTAMP,
    last_sold_date TIMESTAMP,
    total_quantity_sold INTEGER DEFAULT 0,
    total_gross_revenue REAL DEFAULT 0.0,
    avg_unit_price REAL DEFAULT 0.0,
    min_unit_price REAL DEFAULT 0.0,
    max_unit_price REAL DEFAULT 0.0
);

-- Indices for rapid querying and aggregation
CREATE INDEX IF NOT EXISTS idx_trans_invoice ON transactions(invoice);
CREATE INDEX IF NOT EXISTS idx_trans_stock_code ON transactions(stock_code);
CREATE INDEX IF NOT EXISTS idx_trans_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_trans_invoice_date ON transactions(invoice_date);
CREATE INDEX IF NOT EXISTS idx_trans_year_month ON transactions(year_month);
CREATE INDEX IF NOT EXISTS idx_trans_country ON transactions(country);
CREATE INDEX IF NOT EXISTS idx_trans_cancellation ON transactions(is_cancellation);
CREATE INDEX IF NOT EXISTS idx_cust_country ON customers(country);
"""


class AnalyticalDatabase:
    """Analytical SQLite Database Manager for E-Commerce data."""

    def __init__(self, db_path: Union[str, Path] = "data/ecommerce_analytical.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row factory support."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Create database tables and indices if they do not exist."""
        with self.get_connection() as conn:
            conn.executescript(DATABASE_SCHEMA_SQL)

    def load_clean_data(
        self,
        cleaned_df: pd.DataFrame,
        if_exists: str = "replace",
    ) -> Dict[str, int]:
        """
        Populate `transactions`, `customers`, and `products` tables from cleaned DataFrame.
        
        Args:
            cleaned_df: Cleaned transactions DataFrame (from src.cleaning).
            if_exists: How to handle existing tables ('replace' or 'append').
            
        Returns:
            Dictionary with counts of records inserted into each table.
        """
        df = cleaned_df.copy()

        # Format columns for transactions schema
        df["is_cancellation"] = df["is_cancellation"].astype(int)
        df["is_admin_code"] = df["is_admin_code"].astype(int)
        if "customer_id" in df.columns:
            # Handle pandas Int64 to nullable integer in SQLite
            df["customer_id"] = df["customer_id"].astype(object).where(df["customer_id"].notnull(), None)

        trans_cols = [
            "invoice", "stock_code", "description", "quantity", "invoice_date",
            "unit_price", "line_total", "customer_id", "country", "year",
            "month", "year_month", "day_of_week", "hour", "is_cancellation",
            "is_admin_code"
        ]
        if "sheet_source" in df.columns:
            trans_cols.append("sheet_source")

        target_trans = df[[c for c in trans_cols if c in df.columns]].copy()
        if "invoice_date" in target_trans.columns:
            target_trans["invoice_date"] = target_trans["invoice_date"].astype(str)

        with self.get_connection() as conn:
            # 1. Populate transactions
            if if_exists == "replace":
                conn.execute("DELETE FROM transactions;")
                conn.execute("DELETE FROM customers;")
                conn.execute("DELETE FROM products;")

            target_trans.to_sql(
                "transactions",
                conn,
                if_exists="append",
                index=False,
                chunksize=10000,
            )

            # 2. Derive Customers table summary
            cust_sql = """
            INSERT OR REPLACE INTO customers (
                customer_id, country, first_purchase_date, last_purchase_date,
                total_orders, total_gross_revenue, total_net_revenue, total_items_purchased,
                is_repeat_buyer
            )
            SELECT
                customer_id,
                MAX(country) as country,
                MIN(invoice_date) as first_purchase_date,
                MAX(invoice_date) as last_purchase_date,
                COUNT(DISTINCT invoice) as total_orders,
                ROUND(SUM(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total ELSE 0 END), 2) as total_gross_revenue,
                ROUND(SUM(line_total), 2) as total_net_revenue,
                SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) as total_items_purchased,
                CASE WHEN COUNT(DISTINCT CASE WHEN is_cancellation = 0 THEN invoice END) > 1 THEN 1 ELSE 0 END as is_repeat_buyer
            FROM transactions
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id;
            """
            conn.execute(cust_sql)

            # 3. Derive Products table summary
            prod_sql = """
            INSERT OR REPLACE INTO products (
                stock_code, description, first_sold_date, last_sold_date,
                total_quantity_sold, total_gross_revenue, avg_unit_price,
                min_unit_price, max_unit_price
            )
            SELECT
                stock_code,
                MAX(description) as description,
                MIN(invoice_date) as first_sold_date,
                MAX(invoice_date) as last_sold_date,
                SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) as total_quantity_sold,
                ROUND(SUM(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total ELSE 0 END), 2) as total_gross_revenue,
                ROUND(AVG(CASE WHEN unit_price > 0 THEN unit_price END), 2) as avg_unit_price,
                ROUND(MIN(CASE WHEN unit_price > 0 THEN unit_price END), 2) as min_unit_price,
                ROUND(MAX(CASE WHEN unit_price > 0 THEN unit_price END), 2) as max_unit_price
            FROM transactions
            GROUP BY stock_code;
            """
            conn.execute(prod_sql)

            trans_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            cust_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            prod_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

        return {
            "transactions": trans_count,
            "customers": cust_count,
            "products": prod_count,
        }

    def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> pd.DataFrame:
        """Execute a raw SQL query and return results as a Pandas DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # -------------------------------------------------------------
    # Analytical Query Helpers
    # -------------------------------------------------------------

    def get_monthly_revenue_trends(self) -> pd.DataFrame:
        """Monthly gross revenue, net revenue, order volume, and active customer counts."""
        query = """
        SELECT
            year_month,
            COUNT(DISTINCT invoice) AS order_count,
            COUNT(DISTINCT customer_id) AS active_customers,
            ROUND(SUM(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total ELSE 0 END), 2) AS gross_revenue,
            ROUND(SUM(line_total), 2) AS net_revenue,
            ROUND(SUM(CASE WHEN is_cancellation = 1 THEN ABS(line_total) ELSE 0 END), 2) AS refund_amount,
            SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) AS items_sold
        FROM transactions
        GROUP BY year_month
        ORDER BY year_month ASC;
        """
        return self.execute_query(query)

    def get_weekly_revenue_trends(self) -> pd.DataFrame:
        """Weekly aggregated revenue, order count, and units sold."""
        query = """
        SELECT
            strftime('%Y-W%W', invoice_date) AS year_week,
            COUNT(DISTINCT invoice) AS order_count,
            COUNT(DISTINCT customer_id) AS active_customers,
            ROUND(SUM(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total ELSE 0 END), 2) AS gross_revenue,
            ROUND(SUM(line_total), 2) AS net_revenue,
            SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) AS items_sold
        FROM transactions
        GROUP BY year_week
        ORDER BY year_week ASC;
        """
        return self.execute_query(query)

    def get_country_aggregates(self, top_n: int = 15) -> pd.DataFrame:
        """Aggregate sales, orders, and customer counts grouped by country."""
        query = f"""
        SELECT
            country,
            COUNT(DISTINCT invoice) AS order_count,
            COUNT(DISTINCT customer_id) AS customer_count,
            ROUND(SUM(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total ELSE 0 END), 2) AS gross_revenue,
            ROUND(SUM(line_total), 2) AS net_revenue,
            ROUND(AVG(CASE WHEN is_cancellation = 0 AND line_total > 0 THEN line_total END), 2) AS avg_item_value
        FROM transactions
        GROUP BY country
        ORDER BY gross_revenue DESC
        LIMIT {int(top_n)};
        """
        return self.execute_query(query)

    def get_top_selling_products(self, top_n: int = 20) -> pd.DataFrame:
        """Retrieve top products by gross revenue, volume sold, and order frequency."""
        query = f"""
        SELECT
            stock_code,
            description,
            total_quantity_sold,
            total_gross_revenue,
            avg_unit_price
        FROM products
        ORDER BY total_gross_revenue DESC
        LIMIT {int(top_n)};
        """
        return self.execute_query(query)

    def get_cancellation_rates_by_month(self) -> pd.DataFrame:
        """Calculate cancellation and refund rates across months."""
        query = """
        SELECT
            year_month,
            COUNT(DISTINCT invoice) AS total_invoices,
            COUNT(DISTINCT CASE WHEN is_cancellation = 1 THEN invoice END) AS cancelled_invoices,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN is_cancellation = 1 THEN invoice END) / COUNT(DISTINCT invoice),
                2
            ) AS cancellation_rate_pct,
            ROUND(SUM(CASE WHEN is_cancellation = 1 THEN ABS(line_total) ELSE 0 END), 2) AS total_refunded_value
        FROM transactions
        GROUP BY year_month
        ORDER BY year_month ASC;
        """
        return self.execute_query(query)

    def get_order_size_distribution(self) -> pd.DataFrame:
        """Compute order-level metrics: line count, total quantity, total order amount."""
        query = """
        SELECT
            invoice,
            customer_id,
            country,
            is_cancellation,
            COUNT(stock_code) AS items_in_order,
            SUM(quantity) AS total_order_units,
            ROUND(SUM(line_total), 2) AS order_net_total
        FROM transactions
        GROUP BY invoice
        HAVING is_cancellation = 0;
        """
        return self.execute_query(query)
