"""Reusable predictive-modeling pipeline helpers.

Pure orchestration logic extracted from the predictive-modeling demo: building
the standard model zoo, training/evaluating them, comparing performance, and
the data-prep guards (leakage removal, target selection, missing-value
imputation). Importable and unit-testable; the demo script narrates around
these rather than re-implementing them.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .prediction_models import (
    CatBoostModel,
    EnsembleModel,
    LightGBMModel,
    LinearRegressionModel,
    MLPModel,
    RandomForestModel,
    RidgeModel,
    SVRModel,
    XGBoostModel,
)

logger = logging.getLogger(__name__)

# Feature-name fragments that indicate a column was derived from the target and
# would leak it back into the model.
_LEAKAGE_PATTERNS = ("_rolling_", "_lag_", "_scaled", "_rate", "_trend", "_diff")


def find_leakage_features(feature_cols: List[str], target_col: str) -> List[str]:
    """Return the target-derived (leaking) features without removing them."""
    target_base = target_col.replace("_scaled", "")
    return [
        col
        for col in feature_cols
        if target_base in col
        and col != target_col
        and any(pattern in col for pattern in _LEAKAGE_PATTERNS)
    ]


def remove_data_leakage_features(feature_cols: List[str], target_col: str) -> List[str]:
    """Drop features derived from the target variable (data leakage).

    Args:
        feature_cols: Candidate feature column names.
        target_col: The target column name.

    Returns:
        ``feature_cols`` with target-derived columns removed.
    """
    leakage = set(find_leakage_features(feature_cols, target_col))
    return [col for col in feature_cols if col not in leakage]


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Return ``target_col`` if present, else the best available fallback.

    Fallback priority: engagement columns, then like columns, then comment
    columns, then any non-date numeric column.

    Raises:
        ValueError: if no suitable numeric/target column exists.
    """
    if target_col in df.columns:
        return target_col

    candidates: List[str] = []
    candidates += [c for c in df.columns if "engagement" in c.lower()]
    candidates += [c for c in df.columns if "like" in c.lower()]
    candidates += [c for c in df.columns if "comment" in c.lower()]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates += [
        c
        for c in numeric
        if not any(x in c.lower() for x in ("date", "day", "month", "year", "week", "quarter"))
    ]

    seen: set = set()
    for col in candidates:
        if col not in seen:
            seen.add(col)
            return col

    raise ValueError(
        f"No suitable target column found in data. Available columns: {list(df.columns)}"
    )


def prepare_modeling_data(
    df: pd.DataFrame, target_col: str = "num_likes_linkedin_no_video"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split a feature frame into (X, y, feature_cols) ready for modeling.

    Resolves the target column, removes leakage features, and imputes missing
    values (median for numeric, mode/empty-string for categorical).

    Returns:
        Tuple of (features DataFrame, target Series, feature column names).
    """
    target_col = resolve_target_column(df, target_col)

    feature_cols = [col for col in df.columns if col not in ("date", target_col)]
    feature_cols = remove_data_leakage_features(feature_cols, target_col)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    for col in X.columns:
        if X[col].dtype in ("int64", "float64"):
            median_val = X[col].median()
            X[col] = X[col].fillna(0 if pd.isna(median_val) else median_val)
        else:
            mode_val = X[col].mode()
            if len(mode_val) > 0 and not pd.isna(mode_val.iloc[0]):
                X[col] = X[col].fillna(mode_val.iloc[0])
            else:
                X[col] = X[col].fillna("")

    y_median = y.median()
    y = y.fillna(0 if pd.isna(y_median) else y_median)

    return X, y, feature_cols


def create_all_models(target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
    """Instantiate the standard model zoo (tree, linear, MLP, SVR, ensemble)."""
    models: Dict[str, Any] = {
        "random_forest": RandomForestModel(
            target_column=target_col,
            feature_columns=feature_cols,
            n_estimators=100,
            max_depth=10,
            random_state=42,
        ),
        "xgboost": XGBoostModel(
            target_column=target_col,
            feature_columns=feature_cols,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        ),
        "lightgbm": LightGBMModel(
            target_column=target_col,
            feature_columns=feature_cols,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        ),
        "catboost": CatBoostModel(
            target_column=target_col,
            feature_columns=feature_cols,
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
        ),
        "linear_regression": LinearRegressionModel(
            target_column=target_col, feature_columns=feature_cols
        ),
        "ridge": RidgeModel(
            target_column=target_col,
            feature_columns=feature_cols,
            alpha=1.0,
            random_state=42,
        ),
        "mlp": MLPModel(
            target_column=target_col,
            feature_columns=feature_cols,
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
        ),
        "svr": SVRModel(
            target_column=target_col,
            feature_columns=feature_cols,
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
        ),
    }

    ensemble = EnsembleModel(target_column=target_col, feature_columns=feature_cols)
    for model_name in ("random_forest", "xgboost", "lightgbm"):
        if model_name in models:
            ensemble.add_model(models[model_name], weight=1.0)
    models["ensemble"] = ensemble

    return models


def train_all_models(
    models: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, Any], List[str]]:
    """Fit every model, returning (trained_models, failed_model_names)."""
    trained: Dict[str, Any] = {}
    failed: List[str] = []
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained[name] = model
        except Exception as exc:  # noqa: BLE001 - one bad model shouldn't abort
            logger.warning("Error training %s: %s", name, exc)
            failed.append(name)
    return trained, failed


def evaluate_all_models(
    models: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Evaluate each trained model, returning {name: metrics}."""
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        try:
            results[name] = model.evaluate(X, y)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error evaluating %s: %s", name, exc)
    return results


def compare_model_performance(
    evaluation_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Build a per-model comparison frame, sorted best-first.

    Sorts by R² descending when available, else by MAE ascending. Returns an
    empty DataFrame when there are no results.
    """
    if not evaluation_results:
        return pd.DataFrame()

    rows = [{"Model": name, **metrics} for name, metrics in evaluation_results.items()]
    comparison = pd.DataFrame(rows)

    if "r2" in comparison.columns:
        comparison = comparison.sort_values("r2", ascending=False)
    elif "mae" in comparison.columns:
        comparison = comparison.sort_values("mae", ascending=True)

    return comparison.reset_index(drop=True)
