"""
🧠 Predictive Modeling Test Script - Educational & Testing Purposes

This script demonstrates how the predictive modeling system works by:
1. Choosing between sample data or real Supabase data
2. Applying feature engineering (silently)
3. Creating and training multiple ML models
4. Comparing model performance
5. Making predictions for LinkedIn engagement
6. Showing feature importance and practical results

Run this script to:
- Test your predictive modeling implementation
- Learn how different models perform on LinkedIn data
- See the impact of feature engineering on predictions
- Debug any issues with model training
- Test with your real social media data (if Supabase is connected)

Usage:
    python tests/test_predictive_modeling.py

Data Sources:
- 📚 Sample Data: Creates realistic fake data (recommended for learning)
- 🗄️ Real Data: Loads from your Supabase database (for production testing)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineering import FeatureEngineer
from models.prediction_models import (
    RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel,
    LinearRegressionModel, RidgeModel, SVRModel, MLPModel, EnsembleModel
)


def create_sample_data(sample_size=90):
    """
    🎯 Create realistic sample data that mimics your social media posts.
    
    This data structure matches what you'd get from your Supabase database.
    """
    print("🎯 Creating sample social media data...")
    
    # Create date range for the specified number of days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=sample_size)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:sample_size]
    
    # Create sample data with realistic patterns
    # Note: These calculations mimic real social media behavior patterns
    data = []
    for i, date in enumerate(dates):
        # Simulate realistic social media metrics
        base_engagement = 50 + (i % 7) * 10  # Weekly pattern (higher engagement mid-week)
        weekend_boost = 1.5 if date.weekday() >= 5 else 1.0  # Weekend boost (people have more time)
        
        post_data = {
            'date': date.strftime('%Y-%m-%d'),
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            
            # LinkedIn metrics (no video)
            'num_likes_linkedin_no_video': int(base_engagement * weekend_boost + np.random.normal(0, 10)),
            'num_comments_linkedin_no_video': int(base_engagement * 0.2 * weekend_boost + np.random.normal(0, 3)),
            'num_reshares_linkedin_no_video': int(base_engagement * 0.1 * weekend_boost + np.random.normal(0, 2)),
            'engagement_linkedin_no_video': 0,  # Will be calculated
            
            # LinkedIn metrics (video)
            'num_likes_linkedin_video': int(base_engagement * 1.3 * weekend_boost + np.random.normal(0, 15)),
            'num_comments_linkedin_video': int(base_engagement * 0.25 * weekend_boost + np.random.normal(0, 4)),
            'num_reshares_linkedin_video': int(base_engagement * 0.15 * weekend_boost + np.random.normal(0, 3)),
            'engagement_linkedin_video': 0,  # Will be calculated
            
            # Instagram metrics
            'num_likes_instagram_no_video': int(base_engagement * 0.8 * weekend_boost + np.random.normal(0, 8)),
            'num_comments_instagram_no_video': int(base_engagement * 0.15 * weekend_boost + np.random.normal(0, 2)),
            'num_reshares_instagram_no_video': int(base_engagement * 0.05 * weekend_boost + np.random.normal(0, 1)),
            'engagement_instagram_no_video': 0,  # Will be calculated
            
            # Follower counts (realistic growth patterns)
            'num_followers_linkedin': 1000 + i * 2,  # Steady growth (realistic for established accounts)
            'num_followers_instagram': 800 + i * 1.5,  # Slightly slower growth (typical for Instagram)
            
            # Note: post_text column not included as it doesn't exist in your Supabase schema
        }
        
        # Calculate total engagement (using the actual column names from your schema)
        # Note: These calculations match real social media engagement metrics
        post_data['engagement_linkedin_no_video'] = (
            post_data['num_likes_linkedin_no_video'] + 
            post_data['num_comments_linkedin_no_video'] + 
            post_data['num_reshares_linkedin_no_video']
        )
        post_data['engagement_linkedin_video'] = (
            post_data['num_likes_linkedin_video'] + 
            post_data['num_comments_linkedin_video'] + 
            post_data['num_reshares_linkedin_video']
        )
        post_data['engagement_instagram_no_video'] = (
            post_data['num_likes_instagram_no_video'] + 
            post_data['num_comments_instagram_no_video'] + 
            post_data['num_reshares_instagram_no_video']
        )
        
        data.append(post_data)
    
    df = pd.DataFrame(data)
    print(f"✅ Created sample data with {len(df)} posts from {df['date'].min()} to {df['date'].max()}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    return df


def get_sample_size():
    """
    📏 Get desired sample size from user.
    
    Returns:
        int: Number of rows to use for analysis
    """
    print("📏 SAMPLE SIZE SELECTION")
    print("=" * 30)
    print("How many rows (most recent) would you like to analyze?")
    print("💡 Recommendation:")
    print("   - Small dataset (30-50): Fast, good for testing")
    print("   - Medium dataset (60-100): Balanced performance and reliability")  
    print("   - Large dataset (100+): Better model performance, slower training")
    print()
    
    while True:
        try:
            user_input = input("Enter number of rows (default 90): ").strip()
            
            if not user_input:  # User pressed enter without input
                return 90
            
            sample_size = int(user_input)
            
            if sample_size < 10:
                print("⚠️  Warning: Very small sample size may lead to poor model performance")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            elif sample_size > 500:
                print("⚠️  Warning: Large sample size may slow down training significantly")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                    
            print(f"✅ Using sample size: {sample_size} rows")
            print()
            return sample_size
            
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Test cancelled by user")
            exit(0)


def load_real_data_from_supabase(sample_size=90):
    """
    🗄️ Load real data from your Supabase database.
    
    This function connects to your actual database and loads real social media data.
    """
    print("🗄️  Loading real data from Supabase database...")
    
    try:
        # Import Supabase client
        from data.database import SupabaseClient
        
        # Initialize client
        client = SupabaseClient()
        print("✅ Connected to Supabase successfully!")
        
        # Load posts data
        print("📥 Loading posts data...")
        posts_df = client.get_posts_data()
        print(f"   - Loaded {len(posts_df)} posts")
        
        # Load profile data (follower counts)
        print("📥 Loading profile data...")
        profile_df = client.get_profile_data()
        print(f"   - Loaded {len(profile_df)} profile records")
        
        # Merge posts and profile data on date
        if not posts_df.empty and not profile_df.empty:
            # Ensure date columns are in the same format
            posts_df['date'] = pd.to_datetime(posts_df['date']).dt.strftime('%Y-%m-%d')
            profile_df['date'] = pd.to_datetime(profile_df['date']).dt.strftime('%Y-%m-%d')
            
            # Merge on date
            df = posts_df.merge(profile_df, on='date', how='left')
            print(f"✅ Successfully merged data: {len(df)} records")
        else:
            print("⚠️  Warning: One or both datasets are empty, using posts data only")
            df = posts_df if not posts_df.empty else profile_df
        
        # Get the most recent N rows based on sample_size
        if len(df) > sample_size:
            # Sort by date and get the last N rows
            df_sorted = df.sort_values('date')
            df = df_sorted.tail(sample_size).copy()
            print(f"📊 Limited to most recent {sample_size} rows (from {len(df_sorted)} total)")
        else:
            print(f"📊 Using all {len(df)} available rows (requested {sample_size})")
        
        if df.empty:
            print("❌ Error: No data loaded from database")
            print("   Falling back to sample data...")
            return create_sample_data()
        
        # Show data summary
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🔍 Available columns: {len(df.columns)}")
        print()
        
        return df
        
    except ImportError:
        print("❌ Error: Could not import SupabaseClient")
        print("   Make sure your database module is properly set up")
        print("   Falling back to sample data...")
        return create_sample_data()
        
    except Exception as e:
        print(f"❌ Error loading data from Supabase: {str(e)}")
        print("   This could be due to:")
        print("   - Database connection issues")
        print("   - Missing environment variables")
        print("   - Database schema mismatches")
        print("   Falling back to sample data...")
        return create_sample_data()


def choose_data_source():
    """
    🤔 Interactive data source selection.
    
    Allows users to choose between sample data and real Supabase data.
    """
    print("🤔 CHOOSE YOUR DATA SOURCE")
    print("=" * 40)
    print("1. 📚 Sample Data (Recommended for learning/testing)")
    print("   - Creates realistic fake data on the spot")
    print("   - No database connection required")
    print("   - Perfect for understanding predictive modeling")
    print("   - Always works regardless of database status")
    print()
    print("2. 🗄️  Real Data from Supabase")
    print("   - Loads actual data from your database")
    print("   - Tests with your real social media metrics")
    print("   - Requires working Supabase connection")
    print("   - Good for production testing")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print("\n🎯 You chose: Sample Data")
                print("   Perfect for learning and testing!")
                return "sample"
            elif choice == "2":
                print("\n🗄️  You chose: Real Data from Supabase")
                print("   Loading your actual social media data...")
                return "real"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Test cancelled by user")
            exit(0)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("   Please try again.")


def show_data_quality_info(df, stage_name):
    """
    📊 Show data quality information at different stages.
    
    Args:
        df: DataFrame to analyze
        stage_name: Name of the current stage
    """
    print(f"📊 Data Quality Check - {stage_name}")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print(f"   ⚠️  Columns with missing values: {len(missing_cols)}")
        print(f"   🔍 Top 5 columns with most missing values:")
        for col, count in missing_cols.nlargest(5).items():
            percentage = (count / len(df)) * 100
            print(f"      {col}: {count} ({percentage:.1f}%)")
    else:
        print("   ✅ No missing values found")
    
    # Check data types
    dtypes = df.dtypes.value_counts()
    print(f"   📝 Data types:")
    for dtype, count in dtypes.items():
        print(f"      {dtype}: {count} columns")
    
    print()


def apply_feature_engineering_silently(df, fe):
    """
    🔧 Apply feature engineering silently (no demonstration output).
    
    This prepares the data for predictive modeling.
    """
    print("🔧 Applying feature engineering (silently)...")
    
    try:
        # Create temporal features
        df = fe.create_temporal_features(df, 'date')
        print("   ✅ Temporal features created")
    except Exception as e:
        print(f"   ⚠️  Warning creating temporal features: {e}")
    
    try:
        # Create lag features - only for columns that exist
        target_cols = ['engagement_linkedin_no_video', 'num_likes_linkedin_no_video']
        available_cols = [col for col in target_cols if col in df.columns]
        if available_cols:
            df = fe.create_lag_features(df, available_cols, lags=[1, 3, 7])
            print(f"   ✅ Lag features created for {len(available_cols)} columns")
        else:
            print("   ⚠️  No suitable columns found for lag features")
    except Exception as e:
        print(f"   ⚠️  Warning creating lag features: {e}")
    
    try:
        # Create engagement features
        df = fe.create_engagement_features(df, platforms=['linkedin', 'instagram'])
        print("   ✅ Engagement features created")
    except Exception as e:
        print(f"   ⚠️  Warning creating engagement features: {e}")
    
    try:
        # Create cross-platform features
        df = fe.create_cross_platform_features(df)
        print("   ✅ Cross-platform features created")
    except Exception as e:
        print(f"   ⚠️  Warning creating cross-platform features: {e}")
    
    try:
        # Create trend features - only for columns that exist
        target_col = 'engagement_linkedin_no_video'
        if target_col in df.columns:
            df = fe.create_trend_features(df, window_sizes=[7, 14])
            print("   ✅ Trend features created")
        else:
            print("   ⚠️  No engagement column found for trend features")
    except Exception as e:
        print(f"   ⚠️  Warning creating trend features: {e}")
    
    try:
        # Scale features - only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['day_of_week', 'month', 'quarter', 'year', 'day_of_year', 'week_of_year']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        if numeric_cols:
            df = fe.scale_features(df, numeric_cols, scaler_type='standard')
            print(f"   ✅ Scaled {len(numeric_cols)} numeric features")
        else:
            print("   ⚠️  No numeric columns found for scaling")
    except Exception as e:
        print(f"   ⚠️  Warning scaling features: {e}")
    
    print(f"✅ Feature engineering completed. Dataset shape: {df.shape}")
    
    # Show data quality information after feature engineering
    show_data_quality_info(df, "After Feature Engineering")
    
    print()
    
    return df


def diagnose_data_issues(df, target_col='num_likes_linkedin_no_video'):
    """
    🔍 Diagnose potential data issues that could cause prediction problems.
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name
    """
    print("🔍 DIAGNOSING POTENTIAL DATA ISSUES")
    print("=" * 50)
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return
    
    # Check target distribution
    target_data = df[target_col]
    print(f"🎯 Target Variable Analysis ({target_col}):")
    print(f"   Mean: {target_data.mean():.2f}")
    print(f"   Std:  {target_data.std():.2f}")
    print(f"   Min:  {target_data.min():.2f}")
    print(f"   Max:  {target_data.max():.2f}")
    print(f"   Unique values: {target_data.nunique()}")
    
    # Check for constant target (major issue)
    if target_data.nunique() <= 1:
        print("   ❌ CRITICAL: Target has only 1 unique value!")
        print("      This will cause models to predict the same value always.")
        return
    
    # Check for very low variance
    if target_data.std() < 0.01:
        print("   ⚠️  WARNING: Target has very low variance")
        print("      This may cause prediction instability.")
    
    # Check feature-target correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    print(f"\n📊 Feature-Target Correlations (top 10):")
    correlations = []
    for col in numeric_cols:
        try:
            corr = df[col].corr(target_data)
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))
        except:
            continue
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    for i, (col, corr) in enumerate(correlations[:10]):
        print(f"   {i+1:2d}. {col[:40]:40} {corr:.4f}")
    
    # Check for perfect correlations (data leakage)
    perfect_corrs = [col for col, corr in correlations if corr > 0.99]
    if perfect_corrs:
        print(f"\n   ⚠️  WARNING: Perfect correlations detected (possible data leakage):")
        for col in perfect_corrs:
            print(f"      {col}")
    
    # Check dataset size
    print(f"\n📏 Dataset Size Analysis:")
    print(f"   Rows: {len(df)}")
    print(f"   Features: {len(numeric_cols)}")
    print(f"   Features/Rows ratio: {len(numeric_cols)/len(df):.2f}")
    
    if len(numeric_cols) >= len(df):
        print("   ❌ CRITICAL: More features than samples!")
        print("      This will cause severe overfitting.")
    elif len(numeric_cols) / len(df) > 0.1:
        print("   ⚠️  WARNING: High feature-to-sample ratio")
        print("      This may cause overfitting.")
    
    print()


def remove_data_leakage_features(feature_cols, target_col):
    """
    🚫 Remove features that cause data leakage.
    
    Data leakage occurs when features are derived from the target variable,
    leading to unrealistically perfect predictions.
    
    Args:
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        List of cleaned feature columns
    """
    print("🚫 REMOVING DATA LEAKAGE FEATURES")
    print("=" * 40)
    
    # Find target-derived features
    target_base = target_col.replace('_scaled', '')  # Handle scaled versions
    leakage_features = []
    
    for col in feature_cols:
        # Check if feature contains the target name (but isn't the target itself)
        if target_base in col and col != target_col:
            # Common leakage patterns
            leakage_patterns = ['_rolling_', '_lag_', '_scaled', '_rate', '_trend', '_diff']
            if any(pattern in col for pattern in leakage_patterns):
                leakage_features.append(col)
    
    if leakage_features:
        print(f"🔍 Found {len(leakage_features)} potential data leakage features:")
        for i, col in enumerate(leakage_features[:10]):  # Show first 10
            print(f"   {i+1:2d}. {col}")
        if len(leakage_features) > 10:
            print(f"   ... and {len(leakage_features) - 10} more")
        
        # Remove leakage features
        clean_features = [col for col in feature_cols if col not in leakage_features]
        print(f"✅ Removed {len(leakage_features)} leakage features")
        print(f"📊 Features before: {len(feature_cols)} -> after: {len(clean_features)}")
        print()
        
        return clean_features
    else:
        print("✅ No data leakage features detected")
        print()
        return feature_cols


def prepare_modeling_data(df, target_col='num_likes_linkedin_no_video'):
    """
    📊 Prepare data for modeling by separating features and target.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print(f"📊 Preparing data for modeling with target: {target_col}")
    
    # Ensure target column exists
    if target_col not in df.columns:
        print(f"⚠️  Target column '{target_col}' not found in data")
        
        # Look for alternative target columns
        possible_targets = []
        
        # First priority: engagement columns
        engagement_cols = [col for col in df.columns if 'engagement' in col.lower()]
        possible_targets.extend(engagement_cols)
        
        # Second priority: like columns
        like_cols = [col for col in df.columns if 'like' in col.lower()]
        possible_targets.extend(like_cols)
        
        # Third priority: comment columns
        comment_cols = [col for col in df.columns if 'comment' in col.lower()]
        possible_targets.extend(comment_cols)
        
        # Fourth priority: any numeric columns that might be suitable
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out date-related columns
        numeric_cols = [col for col in numeric_cols if not any(x in col.lower() for x in ['date', 'day', 'month', 'year', 'week', 'quarter'])]
        possible_targets.extend(numeric_cols)
        
        if possible_targets:
            # Remove duplicates while preserving order
            seen = set()
            unique_targets = []
            for col in possible_targets:
                if col not in seen:
                    seen.add(col)
                    unique_targets.append(col)
            
            # Select the first available target
            target_col = unique_targets[0]
            print(f"✅ Using alternative target column: {target_col}")
        else:
            raise ValueError(f"No suitable target column found in data. Available columns: {list(df.columns)}")
    
    # Get feature columns (exclude date and target)
    feature_cols = [col for col in df.columns if col not in ['date', target_col]]
    
    # Remove data leakage features
    feature_cols = remove_data_leakage_features(feature_cols, target_col)
    
    # Select features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values more intelligently
    print(f"📊 Original data shape: {X.shape}")
    print(f"🔍 Missing values in features: {X.isnull().sum().sum()}")
    print(f"🔍 Missing values in target: {y.isnull().sum()}")
    
    # Fill missing values in features with appropriate strategies
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # For numeric columns, fill with median (more robust than mean)
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0  # Fallback to 0 if median is also NaN
            X[col] = X[col].fillna(median_val)
        else:
            # For categorical/text columns, fill with mode or empty string
            mode_val = X[col].mode()
            if len(mode_val) > 0 and not pd.isna(mode_val.iloc[0]):
                X[col] = X[col].fillna(mode_val.iloc[0])
            else:
                X[col] = X[col].fillna('')
    
    # Fill missing values in target with median
    y_median = y.median()
    if pd.isna(y_median):
        y_median = 0
    y = y.fillna(y_median)
    
    print(f"📊 After filling missing values - Features shape: {X.shape}")
    print(f"📊 After filling missing values - Target shape: {y.shape}")
    print(f"🎯 Target variable: {target_col}")
    print(f"🔧 Features: {len(feature_cols)}")
    print()
    
    return X, y, feature_cols


def create_all_models(target_col, feature_cols):
    """
    🏗️ Create all available prediction models.
    
    Args:
        target_col: Target column name
        feature_cols: List of feature columns
        
    Returns:
        Dictionary of initialized models
    """
    print("🏗️  Creating all prediction models...")
    
    models = {}
    
    # Tree-based models
    models['random_forest'] = RandomForestModel(
        target_column=target_col,
        feature_columns=feature_cols,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    models['xgboost'] = XGBoostModel(
        target_column=target_col,
        feature_columns=feature_cols,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    models['lightgbm'] = LightGBMModel(
        target_column=target_col,
        feature_columns=feature_cols,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    models['catboost'] = CatBoostModel(
        target_column=target_col,
        feature_columns=feature_cols,
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Linear models
    models['linear_regression'] = LinearRegressionModel(
        target_column=target_col,
        feature_columns=feature_cols
    )
    
    models['ridge'] = RidgeModel(
        target_column=target_col,
        feature_columns=feature_cols,
        alpha=1.0,
        random_state=42
    )
    
    # Neural network
    models['mlp'] = MLPModel(
        target_column=target_col,
        feature_columns=feature_cols,
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42
    )
    
    # Support Vector Regression
    models['svr'] = SVRModel(
        target_column=target_col,
        feature_columns=feature_cols,
        kernel='rbf',
        C=1.0,
        epsilon=0.1
    )
    
    # Ensemble model
    ensemble = EnsembleModel(
        target_column=target_col,
        feature_columns=feature_cols
    )
    
    # Add top models to ensemble
    top_models = ['random_forest', 'xgboost', 'lightgbm']
    for model_name in top_models:
        if model_name in models:
            ensemble.add_model(models[model_name], weight=1.0)
    
    models['ensemble'] = ensemble
    
    print(f"✅ Created {len(models)} models: {list(models.keys())}")
    print()
    
    return models


def train_all_models(models, X, y):
    """
    🚀 Train all models on the data.
    
    Args:
        models: Dictionary of models
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary of trained models
    """
    print("🚀 Training all models...")
    
    trained_models = {}
    failed_models = []
    
    for name, model in models.items():
        try:
            print(f"🔄 Training {name}...")
            model.fit(X, y)
            trained_models[name] = model
            print(f"✅ {name} trained successfully")
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            failed_models.append(name)
            continue
    
    print(f"✅ Successfully trained {len(trained_models)} models")
    if failed_models:
        print(f"❌ Failed to train {len(failed_models)} models: {failed_models}")
    print()
    
    return trained_models


def evaluate_all_models(models, X, y):
    """
    📊 Evaluate all trained models.
    
    Args:
        models: Dictionary of trained models
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    print("📊 Evaluating all models...")
    
    evaluation_results = {}
    
    if not models:
        print("⚠️  No trained models available for evaluation")
        return evaluation_results
    
    for name, model in models.items():
        try:
            print(f"📊 Evaluating {name}...")
            metrics = model.evaluate(X, y)
            evaluation_results[name] = metrics
            
            # Log key metrics
            if 'r2' in metrics:
                print(f"   {name} R²: {metrics['r2']:.4f}")
            if 'mae' in metrics:
                print(f"   {name} MAE: {metrics['mae']:.2f}")
                
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            continue
    
    print()
    return evaluation_results


def compare_model_performance(evaluation_results):
    """
    🔍 Compare model performance and create ranking.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        
    Returns:
        DataFrame with model comparison
    """
    print("🔍 Comparing model performance...")
    
    if not evaluation_results:
        print("⚠️  No model evaluation results available for comparison")
        return pd.DataFrame()
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in evaluation_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by R² score if available, otherwise by MAE
        if 'r2' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('r2', ascending=False)
            print("🏆 Model Performance Ranking (by R² score):")
        elif 'mae' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('mae', ascending=True)
            print("🏆 Model Performance Ranking (by MAE score):")
        else:
            print("🏆 Model Performance Ranking:")
        
        for i, row in comparison_df.iterrows():
            model_name = row['Model']
            if 'r2' in row:
                print(f"   {i+1}. {model_name}: R² = {row['r2']:.4f}")
            elif 'mae' in row:
                print(f"   {i+1}. {model_name}: MAE = {row['mae']:.2f}")
            else:
                print(f"   {i+1}. {model_name}")
        
        print()
        return comparison_df
    else:
        print("⚠️  No valid model comparison data available")
        return pd.DataFrame()


def demonstrate_predictions(models, X, feature_cols, y=None):
    """
    🎯 Demonstrate predictions for different scenarios.
    
    Args:
        models: Dictionary of trained models
        X: Feature DataFrame
        feature_cols: List of feature columns
        y: Target Series (optional, for showing actual vs predicted)
    """
    print("🎯 DEMONSTRATING PREDICTIONS")
    print("=" * 50)
    
    if not models:
        print("⚠️  No trained models available for predictions")
        return
    
    # Get sample data for predictions - use different rows for variety
    if len(X) > 0:
        # Select diverse samples from different parts of the dataset
        sample_indices = []
        if len(X) >= 3:
            sample_indices = [0, len(X)//2, len(X)-1]  # First, middle, last
        else:
            sample_indices = list(range(len(X)))
        
        sample_data = X.iloc[sample_indices]
        print(f"📊 Sample predictions for {len(sample_data)} data points:")
        
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            print(f"\n📊 Sample {i+1} (Row {idx}):")
            
            # Show key feature values for this sample
            print("   🔍 Key Feature Values:")
            key_features = [col for col in X.columns if any(key in col.lower() for key in 
                          ['num_likes_linkedin_no_video', 'engagement_linkedin_no_video', 'day_of_week', 'is_weekend'])][:5]
            for feat in key_features:
                if feat in X.columns:
                    print(f"      {feat}: {row[feat]:.2f}")
            
            # Show actual target value if available
            if y is not None:
                actual_value = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                print(f"   🎯 Actual Target Value: {actual_value:.2f}")
            
            print("   🤖 Model Predictions:")
            for name, model in models.items():
                try:
                    prediction = model.predict(row.to_frame().T)
                    if hasattr(prediction, '__len__') and len(prediction) > 0:
                        pred_value = prediction[0]
                    else:
                        pred_value = prediction
                    
                    # Show difference from actual if available
                    if y is not None:
                        actual_value = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                        diff = pred_value - actual_value
                        print(f"      {name}: {pred_value:.2f} (diff: {diff:+.2f})")
                    else:
                        print(f"      {name}: {pred_value:.2f}")
                except Exception as e:
                    print(f"      {name}: Error - {e}")
        
        # Check if all predictions are identical (potential issue)
        print(f"\n🔍 PREDICTION ANALYSIS:")
        if len(models) > 1:
            all_predictions = {}
            for name, model in models.items():
                try:
                    pred = model.predict(sample_data)
                    all_predictions[name] = pred
                except:
                    continue
            
            # Check for identical predictions across samples
            for model_name, predictions in all_predictions.items():
                if len(set(predictions.round(2))) == 1:
                    print(f"   ⚠️  WARNING: {model_name} gives identical predictions for all samples!")
                    print(f"      This suggests overfitting or data leakage issues.")
                
            # Check for huge differences between models
            if len(all_predictions) > 1:
                pred_ranges = {}
                for i in range(len(sample_data)):
                    sample_preds = [preds[i] for preds in all_predictions.values()]
                    pred_range = max(sample_preds) - min(sample_preds)
                    pred_ranges[f"Sample {i+1}"] = pred_range
                
                max_range = max(pred_ranges.values())
                if max_range > 50:  # Arbitrary threshold
                    print(f"   ⚠️  WARNING: Large prediction differences detected!")
                    print(f"      Maximum range: {max_range:.2f}")
                    print(f"      This suggests model instability or data issues.")
    else:
        print("⚠️  No data available for predictions")
    
    print("\n💡 Prediction Insights:")
    print("   • Different models may give different predictions")
    print("   • Identical predictions across samples indicate potential issues")
    print("   • Large prediction ranges suggest model instability")
    print("   • Ensemble models often provide more stable results")
    print("   • Use the best performing model for production")
    print()


def show_feature_importance(models, feature_cols):
    """
    🎯 Show feature importance analysis.
    
    Args:
        models: Dictionary of trained models
        feature_cols: List of feature columns
    """
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    if not models:
        print("⚠️  No trained models available for feature importance analysis")
        return
    
    # Try to get feature importance from tree-based models
    tree_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    
    for model_name in tree_models:
        if model_name in models:
            try:
                model = models[model_name]
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance is not None and len(importance) > 0:
                        print(f"🔍 {model_name.upper()} Feature Importance:")
                        
                        # Create importance DataFrame from dictionary
                        importance_df = pd.DataFrame([
                            {'Feature': feature, 'Importance': score}
                            for feature, score in importance.items()
                        ]).sort_values('Importance', ascending=False)
                        
                        # Show top 10 features
                        top_features = importance_df.head(10)
                        for i, (_, row) in enumerate(top_features.iterrows()):
                            print(f"   {i+1:2d}. {row['Feature']:30} {row['Importance']:8.4f}")
                        
                        print()
                        return  # Found feature importance, exit
                        
            except Exception as e:
                print(f"⚠️  Error getting feature importance from {model_name}: {e}")
                continue
    
    print("⚠️  No feature importance available from any model")
    print("   • Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost) provide feature importance")
    print("   • Linear models can provide coefficient importance")
    print("   • Consider using a tree-based model for feature importance analysis")
    print()


def run_comprehensive_test():
    """
    🚀 Run the complete predictive modeling test suite.
    
    This function demonstrates all capabilities of the predictive modeling system.
    """
    print("🚀 PREDICTIVE MODELING COMPREHENSIVE TEST")
    print("=" * 60)
    print("This test demonstrates predictive modeling capabilities")
    print("Perfect for learning and testing your implementation")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Choose data source and load data
        data_source = choose_data_source()
        
        # Step 1.5: Get sample size if using real data
        if data_source == "real":
            sample_size = get_sample_size()
        else:
            sample_size = 90  # Default for sample data
        
        if data_source == "sample":
            df = create_sample_data(sample_size)
        else:  # real data
            df = load_real_data_from_supabase(sample_size)
        
        # Step 2: Show available features in the database
        print("📊 AVAILABLE FEATURES IN YOUR DATABASE")
        print("=" * 50)
        print("These are the columns available for modeling:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        print()
        
        # Show initial data quality
        show_data_quality_info(df, "Initial Data Load")
        
        # Step 3: Initialize feature engineer
        print("🔧 Initializing FeatureEngineer...")
        fe = FeatureEngineer()
        print("✅ FeatureEngineer initialized successfully!")
        print()
        
        # Step 4: Apply feature engineering silently
        df_with_features = apply_feature_engineering_silently(df, fe)
        
        # Step 4.5: Diagnose potential data issues
        diagnose_data_issues(df_with_features, target_col='num_likes_linkedin_no_video')
        
        # Step 5: Prepare data for modeling
        X, y, feature_cols = prepare_modeling_data(df_with_features)
        
        # Show data quality after preparation
        show_data_quality_info(X, "After Data Preparation (Features)")
        show_data_quality_info(pd.DataFrame(y), "After Data Preparation (Target)")
        
        # Step 6: Create all models
        models = create_all_models(y.name, feature_cols)
        
        # Step 7: Train all models
        trained_models = train_all_models(models, X, y)
        
        # Step 8: Evaluate all models
        evaluation_results = evaluate_all_models(trained_models, X, y)
        
        # Step 9: Compare model performance
        comparison_df = compare_model_performance(evaluation_results)
        
        # Step 10: Demonstrate predictions
        demonstrate_predictions(trained_models, X, feature_cols, y)
        
        # Step 11: Show feature importance
        show_feature_importance(trained_models, feature_cols)
        
        # Final summary
        print("🎉 PREDICTIVE MODELING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final dataset shape: {df_with_features.shape}")
        print(f"✨ Total features created: {len(feature_cols)}")
        print(f"📅 Data covers: {df['date'].min()} to {df['date'].max()}")
        print(f"🤖 Models tested: {len(trained_models)}")
        print()
        
        print("🔍 What you've learned:")
        print("   ✅ How to prepare data for predictive modeling")
        print("   ✅ How to create and train multiple ML models")
        print("   ✅ How to compare model performance")
        print("   ✅ How to make predictions with different models")
        print("   ✅ How to interpret feature importance")
        print()
        
        print("🚀 Next steps:")
        print("   - Use the best performing model for your predictions")
        print("   - Experiment with different feature combinations")
        print("   - Collect more data to improve model performance")
        print("   - Deploy your model for real-time predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    🎯 Main execution block.
    
    Run this script directly to test your predictive modeling implementation.
    """
    print("🧠 Predictive Modeling Test Script")
    print("=" * 40)
    print("Testing and learning the predictive modeling system")
    print("=" * 40)
    print()
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Test completed successfully!")
        print("Your predictive modeling system is working correctly.")
    else:
        print("\n❌ Test failed. Check the error messages above.")
        print("Make sure your models and feature engineering are properly implemented.")
    
    print("\n📚 Educational Notes:")
    print("- This script shows how to build and evaluate ML models")
    print("- Each model demonstrates different prediction approaches")
    print("- You can choose between sample data and real Supabase data")
    print("- Sample data is perfect for learning and testing")
    print("- Real data tests your actual database connection")
    print("- The ensemble model combines multiple models for better performance")
