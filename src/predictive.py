"""
Predictive Modeling Module (Module G).

Constructs temporal leakage-free customer repeat-purchase prediction models:
  - Temporal cutoff splitting (Observation window vs. Prediction window) to guarantee NO lookahead bias.
  - Feature engineering on observation period:
      * recency, frequency, monetary total, average basket size, cancellation ratio,
      * tenure days, item diversity (unique stock codes).
  - Target label: Will the customer make at least one purchase in the prediction window? (Binary 0/1).
  - Multiple model architectures: Logistic Regression, Random Forest Classifier, Gradient Boosting.
  - Evaluation metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1-score, Confusion Matrix.
  - Feature importances and customer churn / repeat risk scoring.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


class RepeatPurchasePredictor:
    """Predictive Modeling Pipeline for customer repeat purchase propensity without temporal leakage."""

    def __init__(self, transactions_df: pd.DataFrame):
        df = transactions_df[transactions_df["customer_id"].notnull()].copy()
        if "invoice_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
            df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        self.df = df

        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.evaluation_results = {}

    def build_temporal_dataset(
        self,
        cutoff_date: Union[str, datetime] = "2011-06-01",
        prediction_window_days: int = 90,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract observation features strictly BEFORE cutoff_date,
        and target labels in [cutoff_date, cutoff_date + prediction_window_days].
        Guarantees zero data leakage.
        """
        cutoff = pd.to_datetime(cutoff_date)
        window_end = cutoff + pd.Timedelta(days=prediction_window_days)

        # 1. Observation window: transactions strictly before cutoff
        obs_df = self.df[self.df["invoice_date"] < cutoff].copy()
        # 2. Prediction window: transactions in [cutoff, window_end]
        pred_df = self.df[
            (self.df["invoice_date"] >= cutoff) &
            (self.df["invoice_date"] <= window_end)
        ].copy()

        # Target label: customer made at least one valid purchase in prediction window
        if "is_cancellation" in pred_df.columns:
            pred_orders = pred_df[~pred_df["is_cancellation"]]
        else:
            pred_orders = pred_df[pred_df["quantity"] > 0]
        repeat_customers = set(pred_orders["customer_id"].dropna().unique())

        # Build feature set for all customers who were active in observation window
        if obs_df.empty:
            raise ValueError(f"No transactions found in observation window before {cutoff}.")

        # Feature Engineering on observation window
        valid_obs_orders = obs_df[~obs_df["is_cancellation"]] if "is_cancellation" in obs_df.columns else obs_df

        features = obs_df.groupby("customer_id").agg(
            first_date=("invoice_date", "min"),
            last_date=("invoice_date", "max"),
            total_spend=("line_total", "sum"),
            unique_products=("stock_code", "nunique"),
            total_items=("quantity", lambda x: max(0, x.sum())),
        ).reset_index()

        # Order-level aggregations
        order_counts = valid_obs_orders.groupby("customer_id")["invoice"].nunique().reset_index()
        order_counts.rename(columns={"invoice": "frequency"}, inplace=True)

        features = pd.merge(features, order_counts, on="customer_id", how="left").fillna({"frequency": 0})

        # Recency & Tenure
        features["recency_days"] = (cutoff - features["last_date"]).dt.days
        features["tenure_days"] = (cutoff - features["first_date"]).dt.days
        features["avg_order_value"] = (features["total_spend"] / features["frequency"]).replace([np.inf, -np.inf], 0).fillna(0.0)

        # Cancellation propensity in observation window
        if "is_cancellation" in obs_df.columns:
            cancel_counts = obs_df.groupby("customer_id")["is_cancellation"].mean().reset_index()
            cancel_counts.rename(columns={"is_cancellation": "cancellation_ratio"}, inplace=True)
            features = pd.merge(features, cancel_counts, on="customer_id", how="left").fillna({"cancellation_ratio": 0.0})
        else:
            features["cancellation_ratio"] = 0.0

        # Construct target variable: 1 if customer purchased in prediction window, else 0
        features["target_repeat"] = features["customer_id"].apply(lambda cid: 1 if cid in repeat_customers else 0)

        feature_cols = [
            "recency_days", "tenure_days", "frequency", "total_spend",
            "avg_order_value", "unique_products", "total_items", "cancellation_ratio"
        ]
        self.feature_names = feature_cols

        X = features[feature_cols].copy()
        y = features["target_repeat"].copy()

        return X, y

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train multiple classification architectures and compare performance.
        """
        from sklearn.model_selection import train_test_split

        # Stratified train/test split if both classes have at least 2 samples
        class_counts = y.value_counts()
        can_stratify = (len(class_counts) > 1) and (class_counts.min() >= 2)
        stratify_param = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        candidate_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state),
        }

        results = {}
        best_model_name = None
        best_auc = -1.0

        for name, clf in candidate_models.items():
            if name == "Logistic Regression":
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5
            pr_auc = average_precision_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5

            results[name] = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "roc_auc": round(float(auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            self.models[name] = clf
            if auc > best_auc:
                best_auc = auc
                best_model_name = name

        # Feature importances from Random Forest
        rf_model = self.models.get("Random Forest")
        importances = {}
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            for col, imp in zip(self.feature_names, rf_model.feature_importances_):
                importances[col] = round(float(imp), 4)
            importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        summary = {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "repeat_rate_base": round(float(y.mean() * 100), 2),
            "best_model": best_model_name,
            "metrics": results,
            "feature_importances": importances,
        }
        self.evaluation_results = summary
        return summary
