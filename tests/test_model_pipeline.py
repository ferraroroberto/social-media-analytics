"""Tests for the reusable predictive-modeling pipeline helpers."""

import numpy as np
import pandas as pd
import pytest

from src.models.model_pipeline import (
    compare_model_performance,
    create_all_models,
    find_leakage_features,
    prepare_modeling_data,
    remove_data_leakage_features,
    resolve_target_column,
    train_all_models,
)


class TestLeakageRemoval:
    def test_removes_target_derived_features(self):
        target = "num_likes_linkedin_no_video"
        cols = [
            target,
            "num_likes_linkedin_no_video_lag_1",
            "num_likes_linkedin_no_video_rolling_mean_7",
            "day_of_week",
            "num_followers_linkedin",
        ]
        cleaned = remove_data_leakage_features(cols, target)
        assert "num_likes_linkedin_no_video_lag_1" not in cleaned
        assert "num_likes_linkedin_no_video_rolling_mean_7" not in cleaned
        # Unrelated and the target itself are preserved.
        assert "day_of_week" in cleaned
        assert target in cleaned

    def test_find_leakage_matches_removal(self):
        target = "engagement_linkedin_no_video"
        cols = ["engagement_linkedin_no_video_lag_3", "day_of_week"]
        leaking = find_leakage_features(cols, target)
        assert leaking == ["engagement_linkedin_no_video_lag_3"]


class TestResolveTarget:
    def test_returns_present_target(self):
        df = pd.DataFrame({"date": [1], "num_likes_linkedin_no_video": [5]})
        assert resolve_target_column(df, "num_likes_linkedin_no_video") == (
            "num_likes_linkedin_no_video"
        )

    def test_falls_back_to_engagement(self):
        df = pd.DataFrame({"date": [1], "engagement_linkedin_no_video": [5]})
        assert resolve_target_column(df, "missing_col") == "engagement_linkedin_no_video"

    def test_raises_when_no_numeric(self):
        df = pd.DataFrame({"date": [1], "label": ["x"]})
        with pytest.raises(ValueError):
            resolve_target_column(df, "missing_col")


class TestPrepareModelingData:
    def test_splits_and_imputes(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4),
                "num_likes_linkedin_no_video": [10, 20, np.nan, 40],
                "num_likes_linkedin_no_video_lag_1": [1, 2, 3, 4],  # leakage
                "num_followers_linkedin": [100, np.nan, 300, 400],
            }
        )
        X, y, feature_cols = prepare_modeling_data(df)
        # Leakage feature dropped, date excluded.
        assert "num_likes_linkedin_no_video_lag_1" not in feature_cols
        assert "date" not in feature_cols
        assert "num_followers_linkedin" in feature_cols
        # No missing values remain after imputation.
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0


class TestCreateAndTrainModels:
    def test_creates_full_zoo(self):
        models = create_all_models("target", ["f1", "f2"])
        for name in (
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "linear_regression",
            "ridge",
            "mlp",
            "svr",
            "ensemble",
        ):
            assert name in models

    def test_train_reports_failures_without_aborting(self):
        # A model whose .fit raises should land in failed, not crash the run.
        class _Boom:
            def fit(self, X, y):
                raise RuntimeError("boom")

        class _Ok:
            def fit(self, X, y):
                self.fitted = True

        trained, failed = train_all_models(
            {"bad": _Boom(), "good": _Ok()}, pd.DataFrame({"a": [1]}), pd.Series([1])
        )
        assert "good" in trained
        assert failed == ["bad"]


class TestCompareModelPerformance:
    def test_sorts_by_r2_desc(self):
        results = {
            "a": {"r2": 0.5, "mae": 3.0},
            "b": {"r2": 0.9, "mae": 1.0},
        }
        df = compare_model_performance(results)
        assert list(df["Model"]) == ["b", "a"]

    def test_empty_results_returns_empty_frame(self):
        assert compare_model_performance({}).empty
