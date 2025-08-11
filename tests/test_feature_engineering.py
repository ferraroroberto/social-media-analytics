"""
🧪 Feature Engineering Test Script - Educational & Testing Purposes

This script demonstrates how the FeatureEngineer class works by:
1. Choosing between sample data or real Supabase data
2. Applying each feature engineering method step by step
3. Showing the before/after state of the data
4. Explaining what each feature represents
5. Testing all functionality of the FeatureEngineer class

Run this script to:
- Test your FeatureEngineer implementation
- Learn how each feature is created
- See the impact of feature engineering on your data
- Debug any issues with feature creation
- Test with your real social media data (if Supabase is connected)

Usage:
    python tests/test_feature_engineering.py

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


def create_sample_data():
    """
    🎯 Create realistic sample data that mimics your social media posts.
    
    This data structure matches what you'd get from your Supabase database.
    """
    print("🎯 Creating sample social media data...")
    
    # Create date range for the last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
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


def load_real_data_from_supabase():
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
    print("   - Perfect for understanding feature engineering")
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


def demonstrate_temporal_features(df, fe):
    """
    🕐 Demonstrate temporal feature creation.
    
    Shows how dates are transformed into useful features for machine learning.
    """
    print("🕐 STEP 1: Creating Temporal Features")
    print("=" * 50)
    
    print("📅 Original date column (first 5 rows):")
    print(df[['date']].head())
    print()
    
    # Create temporal features
    df_with_temporal = fe.create_temporal_features(df, 'date')
    
    print("✨ New temporal features created:")
    temporal_cols = [col for col in df_with_temporal.columns if col not in df.columns]
    print(f"   Added {len(temporal_cols)} new features:")
    
    for col in temporal_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of temporal features (first 5 rows):")
    temporal_sample = df_with_temporal[['date', 'day_of_week', 'month', 'quarter', 'is_weekend', 'month_sin', 'month_cos']].head()
    print(temporal_sample)
    print()
    
    # Show cyclical encoding explanation
    print("🔄 Cyclical Encoding Explanation:")
    print("   - Regular encoding: month 1 and month 12 are far apart")
    print("   - Sin/Cos encoding: month 1 and month 12 are close (circle)")
    print("   - This helps models understand seasonal patterns better")
    print()
    
    return df_with_temporal


def demonstrate_lag_features(df, fe):
    """
    📈 Demonstrate lag feature creation.
    
    Shows how previous values help predict future performance.
    """
    print("📈 STEP 2: Creating Lag Features")
    print("=" * 50)
    
    # Use the actual column name from your Supabase schema
    target_col = 'engagement_linkedin_no_video'
    
    if target_col not in df.columns:
        print(f"⚠️  Column '{target_col}' not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Using first available numeric column instead.")
        # Fallback to any numeric column that's not date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'date']
        if numeric_cols:
            target_col = numeric_cols[0]
        else:
            print("❌ No suitable columns found for lag features")
            return df
    
    print(f"📊 Using column: {target_col}")
    print("📊 Original engagement data (first 10 rows):")
    print(df[['date', target_col]].head(10))
    print()
    
    # Create lag features
    df_with_lags = fe.create_lag_features(
        df, 
        target_cols=[target_col], 
        lags=[1, 3, 7]
    )
    
    print("✨ New lag features created:")
    lag_cols = [col for col in df_with_lags.columns if col not in df.columns]
    print(f"   Added {len(lag_cols)} new features:")
    
    for col in lag_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of lag features (first 10 rows):")
    lag_sample_cols = ['date', target_col]
    # Add available lag columns
    for lag_col in [f'{target_col}_lag_1', f'{target_col}_lag_3', f'{target_col}_rolling_mean_7']:
        if lag_col in df_with_lags.columns:
            lag_sample_cols.append(lag_col)
    
    lag_sample = df_with_lags[lag_sample_cols].head(10)
    print(lag_sample)
    print()
    
    print("💡 Lag Feature Explanation:")
    print("   - lag_1: Yesterday's engagement (immediate trend)")
    print("   - lag_3: 3 days ago engagement (short-term pattern)")
    print("   - lag_7: Week ago engagement (weekly pattern)")
    print("   - rolling_mean: Average over the last N days (smoothing)")
    print()
    
    return df_with_lags


def demonstrate_engagement_features(df, fe):
    """
    🎯 Demonstrate engagement feature creation.
    
    Shows how engagement metrics are transformed into rates and ratios.
    """
    print("🎯 STEP 3: Creating Engagement Features")
    print("=" * 50)
    
    # Use actual column names from your Supabase schema
    engagement_cols = ['engagement_linkedin_no_video', 'num_followers_linkedin']
    available_cols = [col for col in engagement_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  Some expected columns not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Proceeding with available columns...")
    
    print("📊 Original engagement and follower data (first 5 rows):")
    print(df[['date'] + available_cols].head())
    print()
    
    # Create engagement features
    df_with_engagement = fe.create_engagement_features(df, platforms=['linkedin', 'instagram'])
    
    print("✨ New engagement features created:")
    engagement_new_cols = [col for col in df_with_engagement.columns if col not in df.columns]
    print(f"   Added {len(engagement_new_cols)} new features:")
    
    for col in engagement_new_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of engagement features (first 5 rows):")
    engagement_sample_cols = ['date'] + available_cols
    # Add any new features that were created
    for col in ['engagement_linkedin_no_video_rate', 'linkedin_no_video_comment_ratio']:
        if col in df_with_engagement.columns:
            engagement_sample_cols.append(col)
    
    available_sample_cols = [col for col in engagement_sample_cols if col in df_with_engagement.columns]
    print(df_with_engagement[available_sample_cols].head())
    print()
    
    print("💡 Engagement Feature Explanation:")
    print("   - engagement_rate: Engagement per follower (normalized)")
    print("   - comment_ratio: Comments per like (interaction quality)")
    print("   - share_ratio: Shares per like (virality measure)")
    print()
    
    return df_with_engagement


def demonstrate_cross_platform_features(df, fe):
    """
    🌐 Demonstrate cross-platform feature creation.
    
    Shows how data from multiple platforms is combined and compared.
    """
    print("🌐 STEP 4: Creating Cross-Platform Features")
    print("=" * 50)
    
    # Use actual column names from your Supabase schema
    platform_cols = ['engagement_linkedin_no_video', 'engagement_instagram_no_video', 
                     'num_followers_linkedin', 'num_followers_instagram']
    available_cols = [col for col in platform_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  Some expected columns not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Proceeding with available columns...")
    
    print("📊 Original platform-specific data (first 5 rows):")
    print(df[['date'] + available_cols].head())
    print()
    
    # Create cross-platform features
    df_with_cross = fe.create_cross_platform_features(df)
    
    print("✨ New cross-platform features created:")
    cross_new_cols = [col for col in df_with_cross.columns if col not in df.columns]
    print(f"   Added {len(cross_new_cols)} new features:")
    
    for col in cross_new_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of cross-platform features (first 5 rows):")
    cross_sample_cols = ['date'] + available_cols
    # Add any new features that were created
    for col in ['total_engagement_no_video', 'avg_engagement_no_video', 'total_followers', 'linkedin_share_no_video']:
        if col in df_with_cross.columns:
            cross_sample_cols.append(col)
    
    available_sample_cols = [col for col in cross_sample_cols if col in df_with_cross.columns]
    print(df_with_cross[available_sample_cols].head())
    print()
    
    print("💡 Cross-Platform Feature Explanation:")
    print("   - total_engagement: Sum across all platforms")
    print("   - avg_engagement: Average across all platforms")
    print("   - platform_share: Percentage of total engagement per platform")
    print("   - total_followers: Combined audience size")
    print()
    
    return df_with_cross


def demonstrate_content_features(df, fe):
    """
    📝 Demonstrate content feature creation.
    
    Shows how text content is analyzed for NLP features.
    """
    print("📝 STEP 5: Creating Content Features")
    print("=" * 50)
    
    # Check if we have any text columns for content analysis
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
    
    if not text_columns:
        print("⚠️  No text content columns found in your data")
        print("   Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print()
        print("💡 Content Feature Explanation:")
        print("   - Content features require text columns (e.g., post_text, content, description)")
        print("   - Your current data has engagement metrics and follower counts")
        print("   - To use content features, add text columns to your Supabase schema")
        print()
        return df
    
    # Use the first available text column
    text_col = text_columns[0]
    print(f"📊 Using text column: {text_col}")
    print("📊 Original text content (first 5 rows):")
    print(df[['date', text_col]].head())
    print()
    
    # Create content features
    df_with_content = fe.create_content_features(df, text_col=text_col)
    
    print("✨ New content features created:")
    content_new_cols = [col for col in df_with_content.columns if col not in df.columns]
    print(f"   Added {len(content_new_cols)} new features:")
    
    for col in content_new_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of content features (first 5 rows):")
    content_sample_cols = ['date', text_col]
    # Add any new features that were created
    for col in [f'{text_col}_length', f'{text_col}_word_count', f'{text_col}_sentiment', f'{text_col}_hashtag_count']:
        if col in df_with_content.columns:
            content_sample_cols.append(col)
    
    available_cols = [col for col in content_sample_cols if col in df_with_content.columns]
    print(df_with_content[available_cols].head())
    print()
    
    print("💡 Content Feature Explanation:")
    print("   - text_length: Character count (engagement correlation)")
    print("   - word_count: Word count (readability measure)")
    print("   - sentiment: Positive/negative tone (-1 to +1)")
    print("   - hashtag_count: Number of hashtags (discoverability)")
    print("   - mention_count: Number of @mentions (networking)")
    print()
    
    return df_with_content


def demonstrate_interaction_features(df, fe):
    """
    🤝 Demonstrate interaction feature creation.
    
    Shows how different types of interactions are combined and analyzed.
    """
    print("🤝 STEP 6: Creating Interaction Features")
    print("=" * 50)
    
    # Use actual column names from your Supabase schema
    interaction_cols = ['num_likes_linkedin_no_video', 'num_comments_linkedin_no_video', 'num_reshares_linkedin_no_video']
    available_cols = [col for col in interaction_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  Some expected columns not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Proceeding with available columns...")
    
    print("📊 Original interaction data (first 5 rows):")
    print(df[['date'] + available_cols].head())
    print()
    
    # Create interaction features
    df_with_interactions = fe.create_interaction_features(df)
    
    print("✨ New interaction features created:")
    interaction_new_cols = [col for col in df_with_interactions.columns if col not in df.columns]
    print(f"   Added {len(interaction_new_cols)} new features:")
    
    for col in interaction_new_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of interaction features (first 5 rows):")
    interaction_sample_cols = ['date'] + available_cols
    # Add any new features that were created
    for col in ['linkedin_no_video_comment_like_ratio', 'linkedin_no_video_total_interactions']:
        if col in df_with_interactions.columns:
            interaction_sample_cols.append(col)
    
    available_sample_cols = [col for col in interaction_sample_cols if col in df_with_interactions.columns]
    print(df_with_interactions[available_sample_cols].head())
    print()
    
    print("💡 Interaction Feature Explanation:")
    print("   - comment_like_ratio: Comments per like (engagement quality)")
    print("   - share_like_ratio: Shares per like (virality measure)")
    print("   - total_interactions: Sum of all interaction types")
    print()
    
    return df_with_interactions


def demonstrate_trend_features(df, fe):
    """
    📈 Demonstrate trend feature creation.
    
    Shows how time-based trends and patterns are captured.
    """
    print("📈 STEP 7: Creating Trend Features")
    print("=" * 50)
    
    # Use actual column name from your Supabase schema
    target_col = 'engagement_linkedin_no_video'
    
    if target_col not in df.columns:
        print(f"⚠️  Column '{target_col}' not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Using first available numeric column instead.")
        # Fallback to any numeric column that's not date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'date']
        if numeric_cols:
            target_col = numeric_cols[0]
        else:
            print("❌ No suitable columns found for trend features")
            return df
    
    print(f"📊 Using column: {target_col}")
    print("📊 Original engagement data for trend analysis (first 10 rows):")
    print(df[['date', target_col]].head(10))
    print()
    
    # Create trend features
    df_with_trends = fe.create_trend_features(df, window_sizes=[7, 14])
    
    print("✨ New trend features created:")
    trend_new_cols = [col for col in df_with_trends.columns if col not in df.columns]
    print(f"   Added {len(trend_new_cols)} new features:")
    
    for col in trend_new_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Sample of trend features (first 10 rows):")
    trend_sample_cols = ['date', target_col]
    # Add any new features that were created
    for col in [f'{target_col}_trend_7', f'{target_col}_trend_strength_7']:
        if col in df_with_trends.columns:
            trend_sample_cols.append(col)
    
    available_sample_cols = [col for col in trend_sample_cols if col in df_with_trends.columns]
    print(df_with_trends[available_sample_cols].head(10))
    print()
    
    print("💡 Trend Feature Explanation:")
    print("   - trend_N: Direction of change over N days (positive/negative)")
    print("   - trend_strength_N: Magnitude of change relative to mean")
    print("   - Helps models understand momentum and seasonality")
    print()
    
    return df_with_trends


def demonstrate_feature_scaling(df, fe):
    """
    ⚖️ Demonstrate feature scaling.
    
    Shows how numerical features are normalized for machine learning.
    """
    print("⚖️  STEP 8: Feature Scaling")
    print("=" * 50)
    
    # Select numerical features for scaling (using actual column names from your schema)
    numerical_cols = ['engagement_linkedin_no_video', 'num_followers_linkedin', 'day_of_week', 'month']
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  Some expected columns not found. Available columns:")
        print(f"   {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more")
        print("   Using available numeric columns instead.")
        # Fallback to any numeric columns that exist
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = [col for col in available_cols if col != 'date'][:4]  # Take first 4
    
    print(f"📊 Features to scale: {available_cols}")
    print("Original feature values (first 5 rows):")
    print(df[available_cols].head())
    print()
    
    # Show statistics before scaling
    print("📈 Feature statistics before scaling:")
    print(df[available_cols].describe())
    print()
    
    # Scale features
    df_scaled = fe.scale_features(df, available_cols, scaler_type='standard')
    
    print("✨ Scaled features created:")
    scaled_cols = [col for col in df_scaled.columns if col.endswith('_scaled')]
    print(f"   Added {len(scaled_cols)} scaled features:")
    
    for col in scaled_cols:
        print(f"   - {col}")
    
    print()
    print("📊 Scaled feature values (first 5 rows):")
    print(df_scaled[scaled_cols].head())
    print()
    
    print("📈 Feature statistics after scaling:")
    print(df_scaled[scaled_cols].describe())
    print()
    
    print("💡 Scaling Explanation:")
    print("   - Standard scaling: Mean=0, Standard deviation=1")
    print("   - Helps models converge faster and perform better")
    print("   - Prevents features with large values from dominating")
    print()
    
    return df_scaled


def demonstrate_feature_importance(df, fe):
    """
    🎯 Demonstrate feature importance ranking and target variable selection.
    
    Shows how to choose what to predict and which features are most important.
    """
    print("🎯 STEP 9: Feature Importance & Target Variable Selection")
    print("=" * 60)
    
    # Get all feature columns (excluding date and any text columns that might not exist)
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
    exclude_cols = ['date'] + text_columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"📊 Total features available: {len(feature_cols)}")
    print("Available features for prediction:")
    for i, col in enumerate(feature_cols[:15]):
        print(f"   {i+1:2d}. {col}")
    if len(feature_cols) > 15:
        print(f"   ... and {len(feature_cols) - 15} more")
    print()
    
    # Interactive target variable selection
    print("🎯 CHOOSE YOUR TARGET VARIABLE (What do you want to predict?)")
    print("=" * 50)
    print("The target variable is what your machine learning model will try to predict.")
    print("Choose a column that represents a business outcome you care about.")
    print()
    
    # Default target variable
    default_target = 'num_likes_linkedin_no_video'
    
    # Ensure default target exists and is numeric
    if default_target in feature_cols and pd.api.types.is_numeric_dtype(df[default_target]):
        print(f"💡 Default suggestion: {default_target}")
        print("   This represents the number of likes on LinkedIn posts (no video)")
        print("   It's a good target because it's directly measurable and actionable")
        print()
    else:
        # Find a better default target
        numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_features:
            default_target = numeric_features[0]
            print(f"💡 Default suggestion: {default_target}")
            print("   This is a numeric column suitable for prediction")
            print()
        else:
            default_target = None
            print("⚠️  No suitable numeric columns found for target variable")
            print()
    
    # Show some good target variable options (only numeric ones)
    good_targets = [col for col in feature_cols 
                   if pd.api.types.is_numeric_dtype(df[col]) and 
                   any(word in col.lower() for word in ['likes', 'engagement', 'followers', 'comments', 'shares'])]
    if good_targets:
        print("🎯 Good target variable options:")
        for i, col in enumerate(good_targets[:5]):
            print(f"   {i+1}. {col}")
        print()
    
    # Interactive selection
    while True:
        try:
            if default_target and default_target in feature_cols:
                choice = input(f"Enter target variable name (or press Enter for default '{default_target}'): ").strip()
                if not choice:
                    target_variable = default_target
                    break
            else:
                choice = input("Enter target variable name: ").strip()
            
            if choice in feature_cols:
                # Ensure the chosen target is numeric
                if pd.api.types.is_numeric_dtype(df[choice]):
                    target_variable = choice
                    break
                else:
                    print(f"⚠️  '{choice}' is not numeric. Please choose a numeric column.")
                    print("   Available numeric columns:")
                    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
                    for i, col in enumerate(numeric_cols[:10]):
                        print(f"   {i+1}. {col}")
                    if len(numeric_cols) > 10:
                        print(f"   ... and {len(numeric_cols) - 10} more")
                    print()
            else:
                print(f"❌ '{choice}' not found in your data. Available options:")
                print(f"   {list(feature_cols[:10])}")
                if len(feature_cols) > 10:
                    print(f"   ... and {len(feature_cols) - 10} more")
                print()
        except KeyboardInterrupt:
            print("\n\n👋 Target selection cancelled, using default")
            if default_target and default_target in feature_cols:
                target_variable = default_target
            else:
                # Find any numeric column as fallback
                numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
                target_variable = numeric_cols[0] if numeric_cols else feature_cols[0]
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("   Please try again.")
    
    print(f"✅ Selected target variable: {target_variable}")
    print()
    
    # Remove target variable from feature columns (we don't want to predict it using itself!)
    if target_variable in feature_cols:
        feature_cols.remove(target_variable)
        print(f"⚠️  Removed '{target_variable}' from features (can't predict using itself!)")
        print()
    
    # Show correlation analysis with target variable
    print("🔍 CORRELATION ANALYSIS WITH TARGET VARIABLE")
    print("=" * 50)
    print(f"Let's see which features correlate most with '{target_variable}':")
    print()
    
    # Calculate correlations (only for numeric columns)
    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    if not numeric_features.empty:
        try:
            # Ensure target variable is numeric
            if pd.api.types.is_numeric_dtype(df[target_variable]):
                correlations = df[target_variable].corr(numeric_features)
                correlations = correlations.sort_values(key=abs, ascending=False)
                
                print("📊 Feature correlations with target (absolute values):")
                for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
                    print(f"   {i:2d}. {feature}: {corr:.3f}")
                print()
                
                # Show what correlations mean
                print("💡 What correlations mean:")
                print("   - Close to +1.0: Strong positive relationship (higher feature = higher target)")
                print("   - Close to -1.0: Strong negative relationship (higher feature = lower target)")
                print("   - Close to 0.0: No linear relationship")
                print("   - |correlation| > 0.7: Strong relationship")
                print("   - |correlation| > 0.5: Moderate relationship")
                print("   - |correlation| > 0.3: Weak relationship")
                print()
            else:
                print(f"⚠️  Target variable '{target_variable}' is not numeric, skipping correlation analysis")
                print()
        except Exception as e:
            print(f"⚠️  Error calculating correlations: {str(e)}")
            print("   This can happen with non-numeric data or missing values")
            print("   Continuing with feature importance ranking...")
            print()
    else:
        print("⚠️  No numeric features found for correlation analysis")
        print()
    
    # Get feature importance ranking
    print("🏆 FEATURE IMPORTANCE RANKING")
    print("=" * 50)
    print("Based on domain knowledge and correlation analysis:")
    print()
    
    try:
        ranked_features = fe.get_feature_importance_ranking(feature_cols)
        
        print("🏆 Feature importance ranking (top 20):")
        for i, feature in enumerate(ranked_features[:20]):
            print(f"   {i+1:2d}. {feature}")
        print()
    except Exception as e:
        print(f"⚠️  Error getting feature importance ranking: {str(e)}")
        print("   Using simple alphabetical ranking instead...")
        print()
        ranked_features = sorted(feature_cols)
        
        print("🏆 Simple feature ranking (alphabetical):")
        for i, feature in enumerate(ranked_features[:20]):
            print(f"   {i+1:2d}. {feature}")
        print()
    
    # Add hard data analysis for feature ranking
    print("📊 HARD DATA ANALYSIS FOR FEATURE RANKING")
    print("=" * 60)
    print("Let's analyze why features are ranked in this order using actual data:")
    print()
    
    # 1. Variance analysis (higher variance = more information)
    print("🔍 1. FEATURE VARIANCE ANALYSIS")
    print("-" * 40)
    print("Features with higher variance contain more information:")
    print()
    
    variance_data = []
    for col in feature_cols[:15]:  # Top 15 features
        if pd.api.types.is_numeric_dtype(df[col]):
            variance = df[col].var()
            std = df[col].std()
            mean = df[col].mean()
            cv = std / mean if mean != 0 else 0  # Coefficient of variation
            variance_data.append({
                'feature': col,
                'variance': variance,
                'std': std,
                'mean': mean,
                'cv': cv
            })
    
    # Sort by variance
    variance_data.sort(key=lambda x: x['variance'], reverse=True)
    
    print("📊 Top 10 features by variance (most informative):")
    for i, data in enumerate(variance_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} Var: {data['variance']:>10.2f} | CV: {data['cv']:>6.3f}")
    print()
    
    # 2. Missing value analysis
    print("🔍 2. MISSING VALUE ANALYSIS")
    print("-" * 40)
    print("Features with fewer missing values are more reliable:")
    print()
    
    missing_data = []
    for col in feature_cols[:15]:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_data.append({
            'feature': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
    
    # Sort by missing percentage (ascending)
    missing_data.sort(key=lambda x: x['missing_pct'])
    
    print("📊 Top 10 features by data completeness:")
    for i, data in enumerate(missing_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} Missing: {data['missing_count']:>3d} ({data['missing_pct']:>5.1f}%)")
    print()
    
    # 3. Correlation with target (if we have correlations)
    if 'correlations' in locals() and not correlations.empty:
        print("🔍 3. CORRELATION WITH TARGET ANALYSIS")
        print("-" * 40)
        print("Features with higher absolute correlation are more predictive:")
        print()
        
        print("📊 Top 10 features by absolute correlation with target:")
        for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.5 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            print(f"   {i:2d}. {feature:<30} {corr:>6.3f} ({strength} {direction})")
        print()
    
    # 4. Feature distribution analysis
    print("🔍 4. FEATURE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    print("Features with better distributions are more suitable for ML:")
    print()
    
    distribution_data = []
    for col in feature_cols[:10]:  # Top 10 features
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
            outlier_pct = (outlier_count / len(df)) * 100
            
            # Check for zero variance
            zero_var = df[col].var() == 0
            
            distribution_data.append({
                'feature': col,
                'outlier_count': outlier_count,
                'outlier_pct': outlier_pct,
                'zero_variance': zero_var,
                'unique_values': df[col].nunique()
            })
    
    print("📊 Feature distribution quality (lower outlier % = better):")
    for i, data in enumerate(distribution_data, 1):
        quality = "Good" if data['outlier_pct'] < 5 else "Fair" if data['outlier_pct'] < 15 else "Poor"
        print(f"   {i:2d}. {data['feature']:<30} Outliers: {data['outlier_pct']:>5.1f}% | Quality: {quality}")
    print()
    
    # 5. Feature engineering complexity
    print("🔍 5. FEATURE ENGINEERING COMPLEXITY")
    print("-" * 40)
    print("Simple features are often more reliable than complex ones:")
    print()
    
    complexity_data = []
    for col in feature_cols[:15]:
        # Determine complexity based on feature name patterns
        complexity_score = 0
        if any(word in col.lower() for word in ['lag', 'rolling', 'trend', 'ratio', 'rate']):
            complexity_score += 2  # Engineered features
        if any(word in col.lower() for word in ['sin', 'cos', 'cyclical']):
            complexity_score += 1  # Cyclical encoding
        if any(word in col.lower() for word in ['scaled', 'normalized']):
            complexity_score += 1  # Scaled features
        
        complexity_level = "Simple" if complexity_score == 0 else "Medium" if complexity_score <= 2 else "Complex"
        complexity_data.append({
            'feature': col,
            'complexity_score': complexity_score,
            'complexity_level': complexity_level
        })
    
    # Sort by complexity (ascending - simple first)
    complexity_data.sort(key=lambda x: x['complexity_score'])
    
    print("📊 Feature complexity ranking (simpler = more reliable):")
    for i, data in enumerate(complexity_data[:10], 1):
        print(f"   {i:2d}. {data['feature']:<30} Complexity: {data['complexity_level']:<8} (Score: {data['complexity_score']})")
    print()
    
    # 6. Final ranking explanation
    print("🔍 6. FINAL RANKING EXPLANATION")
    print("=" * 60)
    print("Based on the hard data above, here's why features are ranked this way:")
    print()
    
    print("🏆 TOP FEATURES (Why they're ranked first):")
    if variance_data:
        top_feature = variance_data[0]['feature']
        print(f"   • {top_feature}: Highest variance ({variance_data[0]['variance']:.2f}) = most informative")
    
    if missing_data:
        most_complete = missing_data[0]['feature']
        print(f"   • {most_complete}: Most complete data ({missing_data[0]['missing_pct']:.1f}% missing)")
    
    if 'correlations' in locals() and not correlations.empty:
        most_correlated = correlations.index[0]
        corr_value = correlations.iloc[0]
        print(f"   • {most_correlated}: Highest correlation with target ({corr_value:.3f})")
    
    if complexity_data:
        simplest = complexity_data[0]['feature']
        print(f"   • {simplest}: Simplest feature (Complexity score: {complexity_data[0]['complexity_score']})")
    
    print()
    print("📊 RANKING FACTORS (in order of importance):")
    print("   1. Data completeness (fewer missing values)")
    print("   2. Correlation with target variable")
    print("   3. Feature variance (information content)")
    print("   4. Distribution quality (outlier percentage)")
    print("   5. Feature complexity (simpler = better)")
    print()
    
    # 7. Mathematical ranking calculation
    print("🔍 7. MATHEMATICAL RANKING CALCULATION")
    print("=" * 60)
    print("Here's the exact mathematical formula used for ranking:")
    print()
    
    # Calculate composite score for each feature
    composite_scores = []
    for col in feature_cols[:15]:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Get all the metrics for this feature
            var_data = next((x for x in variance_data if x['feature'] == col), None)
            missing_data_item = next((x for x in missing_data if x['feature'] == col), None)
            dist_data = next((x for x in distribution_data if x['feature'] == col), None)
            comp_data = next((x for x in complexity_data if x['feature'] == col), None)
            
            if all([var_data, missing_data_item, dist_data, comp_data]):
                # Normalize each metric to 0-1 scale
                # Variance: higher is better, normalize by max variance
                max_var = max(x['variance'] for x in variance_data)
                norm_variance = var_data['variance'] / max_var if max_var > 0 else 0
                
                # Missing data: lower is better, invert and normalize
                max_missing = max(x['missing_pct'] for x in missing_data)
                norm_completeness = 1 - (missing_data_item['missing_pct'] / max_missing) if max_missing > 0 else 1
                
                # Distribution: lower outlier % is better, invert and normalize
                max_outliers = max(x['outlier_pct'] for x in distribution_data)
                norm_distribution = 1 - (dist_data['outlier_pct'] / max_outliers) if max_outliers > 0 else 1
                
                # Complexity: lower is better, invert and normalize
                max_complexity = max(x['complexity_score'] for x in complexity_data)
                norm_complexity = 1 - (comp_data['complexity_score'] / max_complexity) if max_complexity > 0 else 1
                
                # Calculate composite score (weighted average)
                weights = [0.3, 0.25, 0.25, 0.2]  # Weights for each factor
                composite_score = (
                    norm_variance * weights[0] +
                    norm_completeness * weights[1] +
                    norm_distribution * weights[2] +
                    norm_complexity * weights[3]
                )
                
                composite_scores.append({
                    'feature': col,
                    'composite_score': composite_score,
                    'norm_variance': norm_variance,
                    'norm_completeness': norm_completeness,
                    'norm_distribution': norm_distribution,
                    'norm_complexity': norm_complexity,
                    'raw_variance': var_data['variance'],
                    'raw_missing_pct': missing_data_item['missing_pct'],
                    'raw_outlier_pct': dist_data['outlier_pct'],
                    'raw_complexity': comp_data['complexity_score']
                })
    
    # Sort by composite score
    composite_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    
    print("📊 COMPOSITE SCORES (mathematical ranking):")
    print("Formula: Score = 0.3×Variance + 0.25×Completeness + 0.25×Distribution + 0.2×Complexity")
    print()
    
    print("🏆 Top 10 features by composite score:")
    for i, score_data in enumerate(composite_scores[:10], 1):
        print(f"   {i:2d}. {score_data['feature']:<30} Score: {score_data['composite_score']:>6.3f}")
        print(f"       Variance: {score_data['norm_variance']:>6.3f} | Completeness: {score_data['norm_completeness']:>6.3f} | Distribution: {score_data['norm_distribution']:>6.3f} | Complexity: {score_data['norm_complexity']:>6.3f}")
        print(f"       Raw: Var={score_data['raw_variance']:>8.2f}, Missing={score_data['raw_missing_pct']:>5.1f}%, Outliers={score_data['raw_outlier_pct']:>5.1f}%, Complexity={score_data['raw_complexity']}")
        print()
    
    print("💡 MATHEMATICAL EXPLANATION:")
    print("   • Each metric is normalized to 0-1 scale (0=worst, 1=best)")
    print("   • Weights reflect relative importance of each factor")
    print("   • Higher composite score = better feature for machine learning")
    print("   • This is purely data-driven, no domain knowledge involved")
    print()
    
    # 8. Compare mathematical vs domain knowledge ranking
    print("🔍 8. MATHEMATICAL vs DOMAIN KNOWLEDGE COMPARISON")
    print("=" * 60)
    print("Let's see how the data-driven ranking differs from domain knowledge:")
    print()
    
    if composite_scores and 'ranked_features' in locals():
        print("📊 COMPARISON OF RANKING METHODS:")
        print()
        
        # Get top 10 from each method
        math_top_10 = [score['feature'] for score in composite_scores[:10]]
        domain_top_10 = ranked_features[:10]
        
        print("🏆 TOP 10 BY MATHEMATICAL SCORE (Data-driven):")
        for i, feature in enumerate(math_top_10, 1):
            print(f"   {i:2d}. {feature}")
        print()
        
        print("🏆 TOP 10 BY DOMAIN KNOWLEDGE (Expert opinion):")
        for i, feature in enumerate(domain_top_10, 1):
            print(f"   {i:2d}. {feature}")
        print()
        
        # Find common features in top 10
        common_features = set(math_top_10) & set(domain_top_10)
        math_only = set(math_top_10) - set(domain_top_10)
        domain_only = set(domain_top_10) - set(math_top_10)
        
        print("📊 ANALYSIS OF DIFFERENCES:")
        print(f"   • Features in BOTH top 10s: {len(common_features)}")
        print(f"   • Only in mathematical top 10: {len(math_only)}")
        print(f"   • Only in domain knowledge top 10: {len(domain_only)}")
        print()
        
        if common_features:
            print("✅ FEATURES AGREED UPON BY BOTH METHODS:")
            for feature in sorted(common_features):
                print(f"   • {feature}")
            print()
        
        if math_only:
            print("🔢 FEATURES RANKED HIGH BY DATA (but not domain knowledge):")
            for feature in math_only:
                # Find the mathematical score for this feature
                score_data = next((s for s in composite_scores if s['feature'] == feature), None)
                if score_data:
                    print(f"   • {feature} (Score: {score_data['composite_score']:.3f})")
            print()
        
        if domain_only:
            print("🧠 FEATURES RANKED HIGH BY DOMAIN KNOWLEDGE (but not data):")
            for feature in domain_only:
                print(f"   • {feature}")
            print()
        
        print("💡 KEY INSIGHTS:")
        print("   • Mathematical ranking focuses on data quality and statistical properties")
        print("   • Domain knowledge ranking considers business relevance and interpretability")
        print("   • Features in both lists are your 'golden features' - use these first!")
        print("   • Features only in mathematical ranking might be good but less interpretable")
        print("   • Features only in domain ranking might be important but have data quality issues")
        print()
    
    # Enhanced explanation
    print("💡 FEATURE IMPORTANCE EXPLANATION")
    print("=" * 50)
    print("Feature importance helps you understand:")
    print("   🎯 Which features matter most for predicting your target")
    print("   🔍 Which features to focus on in your analysis")
    print("   🚀 Which features to include in your machine learning model")
    print("   💰 Which features might be worth collecting more data for")
    print()
    
    print("📊 IMPORTANCE LEVELS:")
    print("   🔴 HIGH IMPORTANCE: Strong influence on predictions")
    print("     - These are your 'golden features'")
    print("     - Focus your data collection efforts here")
    print("     - These often represent core business drivers")
    print()
    print("   🟡 MEDIUM IMPORTANCE: Moderate influence on predictions")
    print("     - Good supporting features")
    print("     - Worth including in your model")
    print("     - May provide incremental improvements")
    print()
    print("   🟢 LOW IMPORTANCE: Weak influence on predictions")
    print("     - These might be noise or redundant")
    print("     - Consider removing to simplify your model")
    print("     - But don't discard without testing!")
    print()
    
    print("🚀 NEXT STEPS FOR MACHINE LEARNING:")
    print("   1. Use the top 10-15 features to start your model")
    print("   2. Test different feature combinations")
    print("   3. Monitor which features actually improve performance")
    print("   4. Iterate and refine based on results")
    print()
    
    return ranked_features, target_variable


def run_comprehensive_test():
    """
    🚀 Run the complete feature engineering test suite.
    
    This function demonstrates all capabilities of the FeatureEngineer class.
    """
    print("🚀 FEATURE ENGINEERING COMPREHENSIVE TEST")
    print("=" * 60)
    print("This test demonstrates all feature engineering capabilities")
    print("Perfect for learning and testing your implementation")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Choose data source and load data
        data_source = choose_data_source()
        
        if data_source == "sample":
            df = create_sample_data()
        else:  # real data
            df = load_real_data_from_supabase()
        
        # Step 2: Show available features in the database
        print("📊 AVAILABLE FEATURES IN YOUR DATABASE")
        print("=" * 50)
        print("These are the columns available for feature engineering:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        print()
        
        # Step 3: Initialize feature engineer
        print("🔧 Initializing FeatureEngineer...")
        fe = FeatureEngineer()
        print("✅ FeatureEngineer initialized successfully!")
        print()
        
        # Step 4: Apply all feature engineering methods
        df = demonstrate_temporal_features(df, fe)
        df = demonstrate_lag_features(df, fe)
        df = demonstrate_engagement_features(df, fe)
        df = demonstrate_cross_platform_features(df, fe)
        df = demonstrate_content_features(df, fe)
        df = demonstrate_interaction_features(df, fe)
        df = demonstrate_trend_features(df, fe)
        df = demonstrate_feature_scaling(df, fe)
        ranked_features, target_variable = demonstrate_feature_importance(df, fe)
        
        # Final summary
        print("🎉 FEATURE ENGINEERING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"✨ Total features created: {len(df.columns)}")
        print(f"📅 Data covers: {df['date'].min()} to {df['date'].max()}")
        print()
        
        print("🔍 Feature breakdown:")
        # Define original columns based on what actually exists
        original_cols = ['date', 'engagement_linkedin_no_video', 'num_followers_linkedin']
        # Add any text columns if they exist
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
        original_cols.extend(text_columns)
        engineered_cols = [col for col in df.columns if col not in original_cols]
        
        print(f"   - Original columns: {len(original_cols)}")
        print(f"   - Engineered features: {len(engineered_cols)}")
        print()
        
        print("💡 What you've learned:")
        print("   ✅ Temporal features (time patterns)")
        print("   ✅ Lag features (historical trends)")
        print("   ✅ Engagement features (normalized metrics)")
        print("   ✅ Cross-platform features (unified view)")
        print("   ✅ Content features (NLP analysis)")
        print("   ✅ Interaction features (engagement ratios)")
        print("   ✅ Trend features (momentum patterns)")
        print("   ✅ Feature scaling (normalization)")
        print("   ✅ Feature importance (prioritization)")
        print()
        
        print("🚀 Next steps:")
        print("   - Use these features to train machine learning models")
        print("   - Experiment with different feature combinations")
        print("   - Analyze which features improve model performance")
        print("   - Customize features for your specific use case")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    🎯 Main execution block.
    
    Run this script directly to test your FeatureEngineer implementation.
    """
    print("🧪 Feature Engineering Test Script")
    print("=" * 40)
    print("Testing and learning the FeatureEngineer class")
    print("=" * 40)
    print()
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Test completed successfully!")
        print("Your FeatureEngineer class is working correctly.")
    else:
        print("\n❌ Test failed. Check the error messages above.")
        print("Make sure your FeatureEngineer class is properly implemented.")
    
    print("\n📚 Educational Notes:")
    print("- This script shows how each feature is created step by step")
    print("- Each method demonstrates a different aspect of feature engineering")
    print("- You can choose between sample data and real Supabase data")
    print("- Sample data is perfect for learning and testing")
    print("- Real data tests your actual database connection")
    print("- Modify the sample data to test different scenarios")
