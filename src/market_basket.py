"""
Market Basket Analysis (Module E).

Implements frequent itemset mining and association rule generation:
  - Basket compilation (grouping items by invoice).
  - Frequent 1-itemsets and 2-itemsets (Apriori candidate generation).
  - Metrics computation: Support, Confidence, Lift, Conviction, Leverage.
  - Product association queries for cross-selling and product bundling.
"""

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd


class MarketBasketAnalysis:
    """Market Basket Analysis Engine for association rules and cross-selling recommendations."""

    def __init__(self, transactions_df: pd.DataFrame):
        """
        Initialize with transaction DataFrame (gross product purchases).
        """
        # Ensure only valid product sales are analyzed
        valid = transactions_df.copy()
        if "quantity" in valid.columns:
            valid = valid[valid["quantity"] > 0]
        if "is_cancellation" in valid.columns:
            valid = valid[~valid["is_cancellation"]]
        if "unit_price" in valid.columns:
            valid = valid[valid["unit_price"] > 0]

        self.df = valid

        # Cache product code to description mapping
        self.code_to_desc = {}
        if "description" in valid.columns and "stock_code" in valid.columns:
            desc_df = valid.dropna(subset=["description"]).drop_duplicates(subset=["stock_code"])
            self.code_to_desc = dict(zip(desc_df["stock_code"], desc_df["description"]))

    def get_baskets(self, min_basket_size: int = 2) -> List[Set[str]]:
        """
        Compile list of unique itemsets (baskets) per invoice.
        """
        baskets_series = self.df.groupby("invoice")["stock_code"].apply(lambda s: set(s.dropna().unique()))
        if min_basket_size > 1:
            baskets_series = baskets_series[baskets_series.apply(len) >= min_basket_size]
        return baskets_series.tolist()

    def generate_association_rules(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.20,
        min_lift: float = 1.0,
        max_rules: int = 100,
    ) -> pd.DataFrame:
        """
        Mine 2-item association rules: A -> B.
        
        Args:
            min_support: Minimum percentage of baskets containing itemset.
            min_confidence: P(B|A) minimum conditional probability.
            min_lift: Lift threshold (>1 indicates positive correlation).
            max_rules: Maximum number of rules to return.
            
        Returns:
            DataFrame of mined association rules.
        """
        baskets = self.get_baskets(min_basket_size=2)
        total_baskets = len(baskets)
        if total_baskets == 0:
            return pd.DataFrame(columns=[
                "antecedent", "consequent", "antecedent_desc", "consequent_desc",
                "support", "confidence", "lift", "leverage"
            ])

        # 1. Count individual item frequencies
        item_counts = Counter()
        for b in baskets:
            for item in b:
                item_counts[item] += 1

        # Filter items meeting minimum support
        min_item_count = max(1, int(min_support * total_baskets))
        frequent_items = {item for item, cnt in item_counts.items() if cnt >= min_item_count}

        # 2. Count pair co-occurrences for frequent items
        pair_counts = Counter()
        for b in baskets:
            # Filter basket to frequent items
            frequent_in_b = sorted([item for item in b if item in frequent_items])
            for item_a, item_b in combinations(frequent_in_b, 2):
                pair_counts[(item_a, item_b)] += 1

        # 3. Generate rules (A -> B and B -> A)
        rules = []
        for (item_a, item_b), pair_cnt in pair_counts.items():
            pair_support = pair_cnt / total_baskets
            if pair_support < min_support:
                continue

            count_a = item_counts[item_a]
            count_b = item_counts[item_b]
            supp_a = count_a / total_baskets
            supp_b = count_b / total_baskets

            # Direction 1: A -> B
            conf_a_b = pair_cnt / count_a
            lift_a_b = conf_a_b / supp_b if supp_b > 0 else 0.0
            leverage_a_b = pair_support - (supp_a * supp_b)

            if conf_a_b >= min_confidence and lift_a_b >= min_lift:
                rules.append({
                    "antecedent": item_a,
                    "consequent": item_b,
                    "antecedent_desc": self.code_to_desc.get(item_a, item_a),
                    "consequent_desc": self.code_to_desc.get(item_b, item_b),
                    "support": round(pair_support, 4),
                    "confidence": round(conf_a_b, 4),
                    "lift": round(lift_a_b, 3),
                    "leverage": round(leverage_a_b, 4),
                })

            # Direction 2: B -> A
            conf_b_a = pair_cnt / count_b
            lift_b_a = conf_b_a / supp_a if supp_a > 0 else 0.0
            leverage_b_a = pair_support - (supp_a * supp_b)

            if conf_b_a >= min_confidence and lift_b_a >= min_lift:
                rules.append({
                    "antecedent": item_b,
                    "consequent": item_a,
                    "antecedent_desc": self.code_to_desc.get(item_b, item_b),
                    "consequent_desc": self.code_to_desc.get(item_a, item_a),
                    "support": round(pair_support, 4),
                    "confidence": round(conf_b_a, 4),
                    "lift": round(lift_b_a, 3),
                    "leverage": round(leverage_b_a, 4),
                })

        df_rules = pd.DataFrame(rules)
        if not df_rules.empty:
            df_rules = df_rules.sort_values(by=["lift", "confidence"], ascending=False).head(max_rules).reset_index(drop=True)

        return df_rules

    def get_bundle_recommendations_for_product(
        self,
        stock_code: str,
        top_n: int = 5,
        min_support: float = 0.005,
    ) -> List[Dict[str, Any]]:
        """Find the top recommended bundle products for a given stock code."""
        rules = self.generate_association_rules(min_support=min_support, min_confidence=0.1, min_lift=1.0, max_rules=500)
        if rules.empty:
            return []

        matches = rules[rules["antecedent"] == stock_code].sort_values(by="lift", ascending=False).head(top_n)
        return matches.to_dict(orient="records")
