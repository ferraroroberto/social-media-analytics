"""
Shared test-data helpers used by demo_feature_engineering.py and demo_predictive_modeling.py.

Extracted from both test scripts to eliminate duplication (issue #5).
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# Add src to path so callers don't have to repeat this
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_sample_data(sample_size: int = 60) -> pd.DataFrame:
    """
    Create realistic sample data that mimics social media posts.

    This data structure matches what you'd get from your Supabase database.

    Args:
        sample_size: Number of days of data to generate (default 60).

    Returns:
        DataFrame with sample social media metrics.
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

            # Note: post_text column not included as it doesn't exist in the Supabase schema
        }

        # Calculate total engagement (using the actual column names from the schema)
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


def load_real_data_from_supabase(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load real data from the Supabase database.

    Connects to the actual database and loads real social media data.
    Falls back to sample data on any error.

    Args:
        sample_size: If provided, limit to the most recent N rows.

    Returns:
        DataFrame with social media metrics (real or sample).
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

        # Optionally limit to the most recent N rows
        if sample_size is not None and len(df) > sample_size:
            df_sorted = df.sort_values('date')
            df = df_sorted.tail(sample_size).copy()
            print(f"📊 Limited to most recent {sample_size} rows (from {len(df_sorted)} total)")
        else:
            if sample_size is not None:
                print(f"📊 Using all {len(df)} available rows (requested {sample_size})")

        if df.empty:
            print("❌ Error: No data loaded from database")
            print("   Falling back to sample data...")
            return create_sample_data(sample_size or 60)

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
        return create_sample_data(sample_size or 60)

    except Exception as e:
        print(f"❌ Error loading data from Supabase: {str(e)}")
        print("   This could be due to:")
        print("   - Database connection issues")
        print("   - Missing environment variables")
        print("   - Database schema mismatches")
        print("   Falling back to sample data...")
        return create_sample_data(sample_size or 60)


def choose_data_source() -> str:
    """
    Interactive data source selection.

    Allows users to choose between sample data and real Supabase data.

    Returns:
        "sample" or "real"
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
