"""Tests for the data-driven feature ranking utilities."""

import numpy as np
import pandas as pd

from src.features.feature_ranking import (
    COMPOSITE_WEIGHTS,
    complexity_analysis,
    complexity_level,
    complexity_score,
    composite_scores,
    distribution_analysis,
    missing_value_analysis,
    target_correlations,
    variance_analysis,
)


def _frame():
    """Small deterministic frame with a high- and low-variance feature."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6),
            "high_var": [0, 100, 0, 100, 0, 100],
            "low_var": [5, 5, 5, 5, 5, 5],
            "with_missing": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
            "engagement_linkedin_no_video": [10, 20, 30, 40, 50, 60],
        }
    )


class TestVarianceAnalysis:
    def test_sorted_by_variance_desc(self):
        df = _frame()
        result = variance_analysis(df, ["high_var", "low_var"])
        assert [r["feature"] for r in result] == ["high_var", "low_var"]
        assert result[0]["variance"] > result[1]["variance"]

    def test_skips_non_numeric(self):
        df = _frame()
        df["text"] = ["a", "b", "c", "d", "e", "f"]
        result = variance_analysis(df, ["text", "high_var"])
        assert [r["feature"] for r in result] == ["high_var"]


class TestMissingValueAnalysis:
    def test_completeness_sorted_ascending(self):
        df = _frame()
        result = missing_value_analysis(df, ["with_missing", "high_var"])
        # high_var has 0% missing so it comes first.
        assert result[0]["feature"] == "high_var"
        assert result[0]["missing_pct"] == 0.0
        missing = next(r for r in result if r["feature"] == "with_missing")
        assert missing["missing_count"] == 2
        assert abs(missing["missing_pct"] - (2 / 6 * 100)) < 1e-9


class TestDistributionAnalysis:
    def test_returns_outlier_metrics(self):
        df = _frame()
        result = distribution_analysis(df, ["engagement_linkedin_no_video"])
        assert len(result) == 1
        entry = result[0]
        assert "outlier_pct" in entry and "zero_variance" in entry
        assert entry["zero_variance"] is False


class TestComplexity:
    def test_score_levels(self):
        assert complexity_score("num_followers_linkedin") == 0
        assert complexity_score("engagement_lag_1") == 2  # lag pattern
        assert complexity_score("engagement_lag_1_scaled") == 3  # lag + scaled
        assert complexity_level(0) == "Simple"
        assert complexity_level(2) == "Medium"
        assert complexity_level(3) == "Complex"

    def test_analysis_sorted_simple_first(self):
        result = complexity_analysis(["a_lag_1", "plain", "b_rolling_mean"])
        assert result[0]["feature"] == "plain"
        assert result[0]["complexity_score"] == 0


class TestCompositeScores:
    def test_weights_sum_to_one(self):
        assert abs(sum(COMPOSITE_WEIGHTS) - 1.0) < 1e-9

    def test_scores_sorted_desc_and_in_range(self):
        df = _frame()
        cols = ["high_var", "low_var", "with_missing", "engagement_linkedin_no_video"]
        scores = composite_scores(df, cols)
        assert scores, "expected at least one scored feature"
        values = [s["composite_score"] for s in scores]
        assert values == sorted(values, reverse=True)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_empty_when_no_numeric(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        assert composite_scores(df, ["text"]) == []


class TestTargetCorrelations:
    def test_perfectly_correlated_feature_ranks_first(self):
        df = _frame()
        # engagement is a perfect linear ramp; build a feature equal to it.
        df["copy"] = df["engagement_linkedin_no_video"]
        result = target_correlations(
            df, "engagement_linkedin_no_video", ["copy", "low_var"]
        )
        assert result.index[0] == "copy"
        assert abs(result.iloc[0] - 1.0) < 1e-9

    def test_empty_for_non_numeric_target(self):
        df = _frame()
        df["cat"] = ["a", "b", "c", "d", "e", "f"]
        result = target_correlations(df, "cat", ["high_var"])
        assert result.empty
