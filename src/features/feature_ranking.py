"""Data-driven feature ranking utilities.

Pure analytical helpers extracted from the feature-engineering demo so the
ranking math can be imported, unit-tested, and reused independently of the
educational walkthrough that narrates it.

Each function takes a DataFrame plus the feature columns to consider and
returns plain Python data structures (lists of dicts / lists of strings).
None of them print — formatting and explanation live in the demo layer.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Weights for the composite score, in the order
# (variance, completeness, distribution, complexity). Must sum to 1.0.
COMPOSITE_WEIGHTS = (0.3, 0.25, 0.25, 0.2)

# Substrings that mark a feature as "engineered" / more complex.
_COMPLEXITY_PATTERNS = {
    2: ("lag", "rolling", "trend", "ratio", "rate"),
    1: ("sin", "cos", "cyclical", "scaled", "normalized"),
}


def _numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Return the subset of ``feature_cols`` that are numeric in ``df``."""
    return [
        col
        for col in feature_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def variance_analysis(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Rank numeric features by variance (higher = more information).

    Returns a list of dicts with ``feature``, ``variance``, ``std``, ``mean``,
    and ``cv`` (coefficient of variation), sorted by variance descending.
    """
    data: List[Dict[str, Any]] = []
    for col in _numeric_features(df, feature_cols):
        std = df[col].std()
        mean = df[col].mean()
        data.append(
            {
                "feature": col,
                "variance": df[col].var(),
                "std": std,
                "mean": mean,
                "cv": std / mean if mean != 0 else 0,
            }
        )
    data.sort(key=lambda x: x["variance"], reverse=True)
    return data


def missing_value_analysis(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Rank features by data completeness (fewer missing = more reliable).

    Returns dicts with ``feature``, ``missing_count``, ``missing_pct``,
    sorted by missing percentage ascending.
    """
    n = len(df)
    data: List[Dict[str, Any]] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        missing_count = int(df[col].isnull().sum())
        data.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_pct": (missing_count / n) * 100 if n else 0.0,
            }
        )
    data.sort(key=lambda x: x["missing_pct"])
    return data


def distribution_analysis(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Assess distribution quality of numeric features via IQR outliers.

    Returns dicts with ``feature``, ``outlier_count``, ``outlier_pct``,
    ``zero_variance``, ``unique_values`` (input order preserved).
    """
    n = len(df)
    data: List[Dict[str, Any]] = []
    for col in _numeric_features(df, feature_cols):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = len(df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)])
        data.append(
            {
                "feature": col,
                "outlier_count": outlier_count,
                "outlier_pct": (outlier_count / n) * 100 if n else 0.0,
                "zero_variance": bool(df[col].var() == 0),
                "unique_values": int(df[col].nunique()),
            }
        )
    return data


def complexity_score(feature: str) -> int:
    """Score a feature's engineering complexity from its name (0 = simplest)."""
    name = feature.lower()
    score = 0
    for points, patterns in _COMPLEXITY_PATTERNS.items():
        if any(word in name for word in patterns):
            score += points
    return score


def complexity_level(score: int) -> str:
    """Map a complexity score to a human-readable level."""
    if score == 0:
        return "Simple"
    if score <= 2:
        return "Medium"
    return "Complex"


def complexity_analysis(feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Rank features by name-derived complexity (simpler first).

    Returns dicts with ``feature``, ``complexity_score``, ``complexity_level``,
    sorted by score ascending.
    """
    data = [
        {
            "feature": col,
            "complexity_score": complexity_score(col),
            "complexity_level": complexity_level(complexity_score(col)),
        }
        for col in feature_cols
    ]
    data.sort(key=lambda x: x["complexity_score"])
    return data


def composite_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    weights: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """Compute a normalized, weighted composite ranking score per feature.

    Combines the four analyses (variance, completeness, distribution,
    complexity), normalizing each metric to a 0-1 scale (1 = best), then
    applies ``weights`` (defaults to :data:`COMPOSITE_WEIGHTS`).

    Returns dicts with the composite score, the normalized components, and the
    raw inputs, sorted by composite score descending.
    """
    weights = weights or COMPOSITE_WEIGHTS

    variance_data = variance_analysis(df, feature_cols)
    missing_data = missing_value_analysis(df, feature_cols)
    distribution_data = distribution_analysis(df, feature_cols)
    complexity_data = complexity_analysis(feature_cols)

    if not variance_data or not distribution_data:
        return []

    max_var = max(x["variance"] for x in variance_data)
    max_missing = max((x["missing_pct"] for x in missing_data), default=0)
    max_outliers = max(x["outlier_pct"] for x in distribution_data)
    max_complexity = max(x["complexity_score"] for x in complexity_data)

    by_feature_var = {x["feature"]: x for x in variance_data}
    by_feature_missing = {x["feature"]: x for x in missing_data}
    by_feature_dist = {x["feature"]: x for x in distribution_data}
    by_feature_comp = {x["feature"]: x for x in complexity_data}

    scores: List[Dict[str, Any]] = []
    for col in feature_cols:
        var_data = by_feature_var.get(col)
        missing_item = by_feature_missing.get(col)
        dist_data = by_feature_dist.get(col)
        comp_data = by_feature_comp.get(col)

        if not all([var_data, missing_item, dist_data, comp_data]):
            continue

        norm_variance = var_data["variance"] / max_var if max_var > 0 else 0
        norm_completeness = (
            1 - (missing_item["missing_pct"] / max_missing) if max_missing > 0 else 1
        )
        norm_distribution = (
            1 - (dist_data["outlier_pct"] / max_outliers) if max_outliers > 0 else 1
        )
        norm_complexity = (
            1 - (comp_data["complexity_score"] / max_complexity)
            if max_complexity > 0
            else 1
        )

        composite = (
            norm_variance * weights[0]
            + norm_completeness * weights[1]
            + norm_distribution * weights[2]
            + norm_complexity * weights[3]
        )

        scores.append(
            {
                "feature": col,
                "composite_score": composite,
                "norm_variance": norm_variance,
                "norm_completeness": norm_completeness,
                "norm_distribution": norm_distribution,
                "norm_complexity": norm_complexity,
                "raw_variance": var_data["variance"],
                "raw_missing_pct": missing_item["missing_pct"],
                "raw_outlier_pct": dist_data["outlier_pct"],
                "raw_complexity": comp_data["complexity_score"],
            }
        )

    scores.sort(key=lambda x: x["composite_score"], reverse=True)
    return scores


def target_correlations(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> "pd.Series":
    """Absolute-sorted correlations of numeric features with ``target_col``.

    Returns an empty Series if the target is missing/non-numeric or there are
    no numeric features.
    """
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        return pd.Series(dtype=float)

    numeric = df[[c for c in feature_cols if c in df.columns]].select_dtypes(
        include=[np.number]
    )
    if numeric.empty:
        return pd.Series(dtype=float)

    # ``DataFrame.corrwith(Series)`` correlates each column against the target;
    # ``Series.corr`` only accepts another Series, so use corrwith here.
    correlations = numeric.corrwith(df[target_col])
    return correlations.sort_values(key=abs, ascending=False)
