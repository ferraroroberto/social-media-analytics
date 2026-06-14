"""Predictive Modeling Demo - educational walkthrough.

A thin narration layer over ``src.models.model_pipeline``: it loads data,
applies feature engineering, then trains, evaluates, and compares the standard
model zoo, printing a step-by-step commentary. The reusable logic lives in the
pipeline module and is unit-tested in ``tests/test_model_pipeline.py``; this
script only orchestrates and explains.

Usage:
    python tests/demo_predictive_modeling.py

Data sources:
- Sample data: realistic fake data (recommended for learning)
- Real data: loaded from Supabase (for production testing)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path so we can import our modules.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from features.feature_engineering import FeatureEngineer
from models.model_pipeline import (
    compare_model_performance,
    create_all_models,
    evaluate_all_models,
    find_leakage_features,
    prepare_modeling_data,
    train_all_models,
)
from tests._sample_data import (
    choose_data_source,
    create_sample_data,
    load_real_data_from_supabase,
)


def get_sample_size():
    """Prompt for the desired sample size (rows). Defaults to 90."""
    print("📏 SAMPLE SIZE SELECTION")
    print("=" * 30)
    print("How many rows (most recent) would you like to analyze?")
    print("   - 30-50: fast, good for testing")
    print("   - 60-100: balanced")
    print("   - 100+: better performance, slower training")
    print()

    while True:
        try:
            user_input = input("Enter number of rows (default 90): ").strip()
            if not user_input:
                return 90
            sample_size = int(user_input)
            if sample_size < 10 or sample_size > 500:
                warn = "very small" if sample_size < 10 else "large"
                print(f"⚠️  Warning: {warn} sample size.")
                if input("Continue anyway? (y/n): ").strip().lower() != "y":
                    continue
            print(f"✅ Using sample size: {sample_size} rows\n")
            return sample_size
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Test cancelled by user")
            exit(0)


def show_data_quality_info(df, stage_name):
    """Print a compact data-quality snapshot for a stage."""
    print(f"📊 Data Quality Check - {stage_name}")
    print(f"   Shape: {df.shape}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"   ⚠️  Columns with missing values: {len(missing)}")
        for col, count in missing.nlargest(5).items():
            print(f"      {col}: {count} ({count / len(df) * 100:.1f}%)")
    else:
        print("   ✅ No missing values found")
    print()


def apply_feature_engineering_silently(df, fe):
    """Run the feature-engineering pipeline with terse progress output."""
    print("🔧 Applying feature engineering...")

    steps = [
        ("temporal", lambda d: fe.create_temporal_features(d, "date")),
        (
            "lag",
            lambda d: fe.create_lag_features(
                d,
                [c for c in ("engagement_linkedin_no_video", "num_likes_linkedin_no_video") if c in d.columns],
                lags=[1, 3, 7],
            ),
        ),
        ("engagement", lambda d: fe.create_engagement_features(d, platforms=["linkedin", "instagram"])),
        ("cross-platform", fe.create_cross_platform_features),
        ("trend", lambda d: fe.create_trend_features(d, window_sizes=[7, 14])),
    ]
    for name, fn in steps:
        try:
            df = fn(df)
            print(f"   ✅ {name} features")
        except Exception as exc:
            print(f"   ⚠️  {name}: {exc}")

    try:
        exclude = {"day_of_week", "month", "quarter", "year", "day_of_year", "week_of_year"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]
        if numeric_cols:
            df = fe.scale_features(df, numeric_cols)
            print(f"   ✅ scaled {len(numeric_cols)} numeric features")
    except Exception as exc:
        print(f"   ⚠️  scaling: {exc}")

    print(f"✅ Feature engineering done. Shape: {df.shape}\n")
    show_data_quality_info(df, "After Feature Engineering")
    return df


def diagnose_data_issues(df, target_col="num_likes_linkedin_no_video"):
    """Print target distribution, top correlations, and overfitting warnings."""
    print("🔍 DIAGNOSING POTENTIAL DATA ISSUES")
    print("=" * 50)

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return

    target = df[target_col]
    print(f"🎯 Target '{target_col}': mean={target.mean():.2f} std={target.std():.2f} "
          f"min={target.min():.2f} max={target.max():.2f} unique={target.nunique()}")

    if target.nunique() <= 1:
        print("   ❌ CRITICAL: target has only 1 unique value!")
        return
    if target.std() < 0.01:
        print("   ⚠️  WARNING: target has very low variance")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(target)
        if not pd.isna(corr):
            correlations.append((col, abs(corr)))
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 Feature-Target Correlations (top 10):")
    for i, (col, corr) in enumerate(correlations[:10], 1):
        print(f"   {i:2d}. {col[:40]:40} {corr:.4f}")

    leakage = [col for col, corr in correlations if corr > 0.99]
    if leakage:
        print("\n   ⚠️  Perfect correlations (possible leakage):")
        for col in leakage:
            print(f"      {col}")

    ratio = len(numeric_cols) / len(df) if len(df) else 0
    print(f"\n📏 Rows: {len(df)} | Features: {len(numeric_cols)} | ratio: {ratio:.2f}")
    if len(numeric_cols) >= len(df):
        print("   ❌ CRITICAL: more features than samples (severe overfitting risk)")
    elif ratio > 0.1:
        print("   ⚠️  WARNING: high feature-to-sample ratio")
    print()


def demonstrate_predictions(models, X, feature_cols, y=None):
    """Run sample predictions and flag identical/divergent outputs."""
    print("🎯 DEMONSTRATING PREDICTIONS")
    print("=" * 50)

    if not models or len(X) == 0:
        print("⚠️  No models or no data available for predictions")
        return

    sample_indices = [0, len(X) // 2, len(X) - 1] if len(X) >= 3 else list(range(len(X)))
    sample_data = X.iloc[sample_indices]
    print(f"📊 Sample predictions for {len(sample_data)} data points:")

    for i, (idx, row) in enumerate(sample_data.iterrows()):
        print(f"\n📊 Sample {i + 1} (Row {idx}):")
        if y is not None:
            actual = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            print(f"   🎯 Actual: {actual:.2f}")
        for name, model in models.items():
            try:
                prediction = model.predict(row.to_frame().T)
                pred_value = prediction[0] if hasattr(prediction, "__len__") else prediction
                if y is not None:
                    actual = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
                    print(f"      {name}: {pred_value:.2f} (diff: {pred_value - actual:+.2f})")
                else:
                    print(f"      {name}: {pred_value:.2f}")
            except Exception as exc:
                print(f"      {name}: Error - {exc}")

    print("\n💡 Identical predictions across samples suggest overfitting / leakage;")
    print("   large divergence between models suggests instability.\n")


def show_feature_importance(models, feature_cols):
    """Print top-10 feature importances from the first tree model that has them."""
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    for model_name in ("random_forest", "xgboost", "lightgbm", "catboost"):
        model = models.get(model_name)
        if model is None or not hasattr(model, "get_feature_importance"):
            continue
        try:
            importance = model.get_feature_importance()
            if importance:
                print(f"🔍 {model_name.upper()} Feature Importance (top 10):")
                ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
                for i, (feature, score) in enumerate(ranked[:10], 1):
                    print(f"   {i:2d}. {feature:30} {score:8.4f}")
                print()
                return
        except Exception as exc:
            print(f"⚠️  {model_name}: {exc}")

    print("⚠️  No feature importance available (use a tree-based model)\n")


def run_comprehensive_test():
    """Run the full predictive-modeling demo end to end."""
    print("🚀 PREDICTIVE MODELING DEMO")
    print("=" * 60)

    try:
        data_source = choose_data_source()
        sample_size = get_sample_size() if data_source == "real" else 90
        df = (
            create_sample_data(sample_size)
            if data_source == "sample"
            else load_real_data_from_supabase(sample_size)
        )

        print("\n📊 Columns available for modeling:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        print()
        show_data_quality_info(df, "Initial Data Load")

        fe = FeatureEngineer()
        df_with_features = apply_feature_engineering_silently(df, fe)
        diagnose_data_issues(df_with_features, target_col="num_likes_linkedin_no_video")

        # Reusable pipeline logic (leakage removal + imputation + target resolution).
        target_guess = "num_likes_linkedin_no_video"
        raw_feature_cols = [c for c in df_with_features.columns if c not in ("date", target_guess)]
        leakage = find_leakage_features(raw_feature_cols, target_guess)
        if leakage:
            print(f"🚫 Removing {len(leakage)} data-leakage features "
                  f"(e.g. {', '.join(leakage[:3])})\n")

        X, y, feature_cols = prepare_modeling_data(df_with_features, target_col=target_guess)
        print(f"📊 Prepared X={X.shape}, target='{y.name}', features={len(feature_cols)}\n")
        show_data_quality_info(X, "After Data Preparation (Features)")

        models = create_all_models(y.name, feature_cols)
        print(f"🏗️  Created {len(models)} models: {list(models.keys())}\n")

        trained_models, failed = train_all_models(models, X, y)
        print(f"✅ Trained {len(trained_models)} models" +
              (f"; failed: {failed}" if failed else "") + "\n")

        evaluation_results = evaluate_all_models(trained_models, X, y)
        comparison_df = compare_model_performance(evaluation_results)
        if not comparison_df.empty:
            print("🏆 Model Performance Ranking:")
            for i, row in comparison_df.iterrows():
                if "r2" in row:
                    print(f"   {i + 1}. {row['Model']}: R² = {row['r2']:.4f}")
                elif "mae" in row:
                    print(f"   {i + 1}. {row['Model']}: MAE = {row['mae']:.2f}")
            print()

        demonstrate_predictions(trained_models, X, feature_cols, y)
        show_feature_importance(trained_models, feature_cols)

        print("🎉 PREDICTIVE MODELING DEMO COMPLETED")
        print("=" * 60)
        print(f"📊 Final shape: {df_with_features.shape} | features: {len(feature_cols)} "
              f"| models: {len(trained_models)}")
        print(f"📅 Data covers: {df['date'].min()} to {df['date'].max()}")
        return True

    except Exception as exc:
        print(f"❌ Error during demo: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧠 Predictive Modeling Demo\n")
    success = run_comprehensive_test()
    print("\n🎯 Demo completed." if success else "\n❌ Demo failed; see errors above.")
