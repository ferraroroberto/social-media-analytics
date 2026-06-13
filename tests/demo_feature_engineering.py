"""Feature Engineering Demo - educational walkthrough.

A thin narration layer over ``src.features.feature_engineering`` and
``src.features.feature_ranking``: it applies each feature-engineering method in
turn and explains the result, then ranks the resulting features using the pure
math in ``feature_ranking`` (unit-tested in ``tests/test_feature_ranking.py``).
This script only orchestrates and explains; the reusable logic lives in ``src``.

Usage:
    python tests/demo_feature_engineering.py

Data sources:
- Sample data: realistic fake data (recommended for learning)
- Real data: loaded from Supabase (for production testing)
"""

import os
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path so we can import our modules.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from features.feature_engineering import FeatureEngineer
from features.feature_ranking import (
    complexity_analysis,
    composite_scores,
    distribution_analysis,
    missing_value_analysis,
    target_correlations,
    variance_analysis,
)
from tests._sample_data import (
    choose_data_source,
    create_sample_data,
    load_real_data_from_supabase,
)


def _show_new_columns(df_before, df_after, label):
    """Print the columns added by a feature-engineering step."""
    new_cols = [c for c in df_after.columns if c not in df_before.columns]
    print(f"✨ {label}: added {len(new_cols)} features")
    for col in new_cols:
        print(f"   - {col}")
    print()


def demonstrate_temporal_features(df, fe):
    """STEP 1: temporal features (dates → ML-friendly cyclical encodings)."""
    print("🕐 STEP 1: Temporal Features")
    print("=" * 50)
    result = fe.create_temporal_features(df, "date")
    _show_new_columns(df, result, "Temporal features")
    print("💡 Sin/Cos encoding keeps month 1 and month 12 close (cyclical).\n")
    return result


def demonstrate_lag_features(df, fe):
    """STEP 2: lag features (past values predict future performance)."""
    print("📈 STEP 2: Lag Features")
    print("=" * 50)
    target_col = "engagement_linkedin_no_video"
    if target_col not in df.columns:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "date"]
        if not numeric:
            print("❌ No numeric column available for lag features\n")
            return df
        target_col = numeric[0]
    print(f"📊 Using column: {target_col}")
    result = fe.create_lag_features(df, target_cols=[target_col], lags=[1, 3, 7])
    _show_new_columns(df, result, "Lag features")
    print("💡 lag_N = value N days ago; rolling_mean smooths recent history.\n")
    return result


def demonstrate_engagement_features(df, fe):
    """STEP 3: engagement features (rates and interaction ratios)."""
    print("🎯 STEP 3: Engagement Features")
    print("=" * 50)
    result = fe.create_engagement_features(df, platforms=["linkedin", "instagram"])
    _show_new_columns(df, result, "Engagement features")
    print("💡 engagement_rate normalizes by followers; ratios measure quality.\n")
    return result


def demonstrate_cross_platform_features(df, fe):
    """STEP 4: cross-platform features (combine/compare across platforms)."""
    print("🌐 STEP 4: Cross-Platform Features")
    print("=" * 50)
    result = fe.create_cross_platform_features(df)
    _show_new_columns(df, result, "Cross-platform features")
    print("💡 total/avg engagement and per-platform share across platforms.\n")
    return result


def demonstrate_content_features(df, fe):
    """STEP 5: content features (NLP over any text column)."""
    print("📝 STEP 5: Content Features")
    print("=" * 50)
    text_columns = [c for c in df.columns if "text" in c.lower() or "content" in c.lower()]
    if not text_columns:
        print("⚠️  No text columns found; content features require a text column.\n")
        return df
    text_col = text_columns[0]
    print(f"📊 Using text column: {text_col}")
    result = fe.create_content_features(df, text_col=text_col)
    _show_new_columns(df, result, "Content features")
    print("💡 length, word_count, sentiment, hashtag/mention counts.\n")
    return result


def demonstrate_interaction_features(df, fe):
    """STEP 6: interaction features (combine likes/comments/reshares)."""
    print("🤝 STEP 6: Interaction Features")
    print("=" * 50)
    result = fe.create_interaction_features(df)
    _show_new_columns(df, result, "Interaction features")
    print("💡 comment/like and share/like ratios, total interactions.\n")
    return result


def demonstrate_trend_features(df, fe):
    """STEP 7: trend features (momentum over rolling windows)."""
    print("📈 STEP 7: Trend Features")
    print("=" * 50)
    target_col = "engagement_linkedin_no_video"
    if target_col not in df.columns:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "date"]
        if not numeric:
            print("❌ No numeric column available for trend features\n")
            return df
        target_col = numeric[0]
    print(f"📊 Using column: {target_col}")
    result = fe.create_trend_features(df, window_sizes=[7, 14])
    _show_new_columns(df, result, "Trend features")
    print("💡 trend_N = direction over N days; trend_strength = magnitude.\n")
    return result


def demonstrate_feature_scaling(df, fe):
    """STEP 8: feature scaling (standardize numeric features)."""
    print("⚖️  STEP 8: Feature Scaling")
    print("=" * 50)
    numerical_cols = [
        c
        for c in ("engagement_linkedin_no_video", "num_followers_linkedin", "day_of_week", "month")
        if c in df.columns
    ]
    if not numerical_cols:
        numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "date"][:4]
    print(f"📊 Scaling: {numerical_cols}")
    result = fe.scale_features(df, numerical_cols, scaler_type="standard")
    scaled_cols = [c for c in result.columns if c.endswith("_scaled")]
    print(f"✨ Added {len(scaled_cols)} scaled features (mean≈0, std≈1)\n")
    return result


def _select_target_variable(df, feature_cols: List[str]) -> str:
    """Interactively choose a numeric target column (default suggested)."""
    default_target = "num_likes_linkedin_no_video"
    if not (default_target in feature_cols and pd.api.types.is_numeric_dtype(df[default_target])):
        numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        default_target = numeric[0] if numeric else None

    good_targets = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
        and any(w in c.lower() for w in ("likes", "engagement", "followers", "comments", "shares"))
    ]
    if good_targets:
        print("🎯 Good target options:", ", ".join(good_targets[:5]))

    while True:
        try:
            prompt = (
                f"Enter target variable (Enter for '{default_target}'): "
                if default_target
                else "Enter target variable name: "
            )
            choice = input(prompt).strip()
            if not choice and default_target:
                return default_target
            if choice in feature_cols and pd.api.types.is_numeric_dtype(df[choice]):
                return choice
            print(f"❌ '{choice}' is not a valid numeric column. Try again.")
        except KeyboardInterrupt:
            print("\n👋 Using default target")
            if default_target:
                return default_target
            numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            return numeric[0] if numeric else feature_cols[0]


def demonstrate_feature_importance(df, fe):
    """STEP 9: target selection + data-driven feature ranking.

    Delegates the ranking math to ``src.features.feature_ranking``; this
    function only narrates the results.
    """
    print("🎯 STEP 9: Feature Importance & Target Selection")
    print("=" * 60)

    text_columns = [c for c in df.columns if "text" in c.lower() or "content" in c.lower()]
    feature_cols = [c for c in df.columns if c not in (["date"] + text_columns)]
    print(f"📊 {len(feature_cols)} features available for prediction\n")

    target_variable = _select_target_variable(df, feature_cols)
    print(f"✅ Selected target: {target_variable}\n")
    if target_variable in feature_cols:
        feature_cols.remove(target_variable)

    # Correlations with the target (reusable math from feature_ranking).
    correlations = target_correlations(df, target_variable, feature_cols)
    if not correlations.empty:
        print("🔍 Top correlations with target:")
        for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.5 else "Weak"
            print(f"   {i:2d}. {feature:<30} {corr:>6.3f} ({strength})")
        print()

    # Data-driven analyses (all delegated to feature_ranking).
    top = feature_cols[:15]
    variance_data = variance_analysis(df, top)
    missing_data = missing_value_analysis(df, top)
    distribution_data = distribution_analysis(df, feature_cols[:10])
    complexity_data = complexity_analysis(top)
    scores = composite_scores(df, top)

    print("🔍 Top features by variance (most informative):")
    for i, data in enumerate(variance_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} Var: {data['variance']:>10.2f} | CV: {data['cv']:>6.3f}")
    print()

    print("🔍 Top features by data completeness:")
    for i, data in enumerate(missing_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} Missing: {data['missing_pct']:>5.1f}%")
    print()

    print("🔍 Distribution quality (lower outlier % = better):")
    for i, data in enumerate(distribution_data, 1):
        quality = "Good" if data["outlier_pct"] < 5 else "Fair" if data["outlier_pct"] < 15 else "Poor"
        print(f"   {i:2d}. {data['feature']:<30} Outliers: {data['outlier_pct']:>5.1f}% ({quality})")
    print()

    print("🔍 Complexity (simpler = more reliable):")
    for i, data in enumerate(complexity_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} {data['complexity_level']}")
    print()

    if scores:
        print("🏆 Composite ranking")
        print("   Score = 0.3×Variance + 0.25×Completeness + 0.25×Distribution + 0.2×Complexity")
        for i, s in enumerate(scores[:10], 1):
            print(f"   {i:2d}. {s['feature']:<30} Score: {s['composite_score']:>6.3f}")
        print()

    return target_variable


def run_comprehensive_test():
    """Run the full feature-engineering demo end to end."""
    print("🚀 FEATURE ENGINEERING DEMO")
    print("=" * 60)

    try:
        data_source = choose_data_source()
        df = create_sample_data() if data_source == "sample" else load_real_data_from_supabase()

        print("\n📊 Columns available for feature engineering:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        print()

        fe = FeatureEngineer()
        df = demonstrate_temporal_features(df, fe)
        df = demonstrate_lag_features(df, fe)
        df = demonstrate_engagement_features(df, fe)
        df = demonstrate_cross_platform_features(df, fe)
        df = demonstrate_content_features(df, fe)
        df = demonstrate_interaction_features(df, fe)
        df = demonstrate_trend_features(df, fe)
        df = demonstrate_feature_scaling(df, fe)
        demonstrate_feature_importance(df, fe)

        print("🎉 FEATURE ENGINEERING DEMO COMPLETED")
        print("=" * 60)
        print(f"📊 Final shape: {df.shape} | total features: {len(df.columns)}")
        print(f"📅 Data covers: {df['date'].min()} to {df['date'].max()}")
        return True

    except Exception as exc:
        print(f"❌ Error during demo: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Feature Engineering Demo\n")
    success = run_comprehensive_test()
    print("\n🎯 Demo completed." if success else "\n❌ Demo failed; see errors above.")
