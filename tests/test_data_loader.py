"""Tests for data loader module."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data.data_loader import DataLoader
from src.data.database import SupabaseClient


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_supabase_client = Mock(spec=SupabaseClient)
        self.data_loader = DataLoader(self.mock_supabase_client)
    
    def test_init(self):
        """Test DataLoader initialization."""
        assert self.data_loader.supabase_client == self.mock_supabase_client
        assert self.data_loader.platforms == ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        assert self.data_loader.content_types == ['no_video', 'video']
    
    def test_load_consolidated_data(self):
        """Test loading consolidated data."""
        # Mock data
        mock_posts_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'engagement_linkedin_no_video': [100, 150]
        })
        mock_profile_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_followers_linkedin': [1000, 1050]
        })
        
        self.mock_supabase_client.get_posts_data.return_value = mock_posts_df
        self.mock_supabase_client.get_profile_data.return_value = mock_profile_df
        
        # Test
        posts_df, profile_df = self.data_loader.load_consolidated_data()
        
        assert len(posts_df) == 2
        assert len(profile_df) == 2
        self.mock_supabase_client.get_posts_data.assert_called_once()
        self.mock_supabase_client.get_profile_data.assert_called_once()
    
    def test_preprocess_posts_data(self):
        """Test posts data preprocessing."""
        # Mock data
        posts_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_likes_linkedin_no_video': [50, 75],
            'num_comments_linkedin_no_video': [10, 15],
            'num_reshares_linkedin_no_video': [5, 8]
        })

        # Test
        processed_df = self.data_loader.preprocess_posts_data(posts_df)

        assert 'engagement_linkedin_no_video' in processed_df.columns
        assert 'day_of_week' in processed_df.columns
        assert 'is_weekend' in processed_df.columns
        assert processed_df['engagement_linkedin_no_video'].iloc[0] == 65  # 50 + 10 + 5

    def test_preprocess_posts_data_no_nan_from_missing_columns(self):
        """Regression: pd.Series(0) default causes index-alignment NaN for all
        rows except index 0. Using a scalar default (or column-guard) must
        produce zero-NaN engagement columns even with a multi-row frame where
        comments/reshares columns are absent."""
        posts_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'num_likes_linkedin_no_video': [10, 20, 30, 40, 50],
            # comments and reshares columns deliberately absent
        })

        processed_df = self.data_loader.preprocess_posts_data(posts_df)

        assert 'engagement_linkedin_no_video' in processed_df.columns
        # Every row must be non-NaN (the old pd.Series(0) default blanked rows 1-4)
        assert processed_df['engagement_linkedin_no_video'].isna().sum() == 0
        # Values must equal likes only (no comments/reshares to add)
        assert list(processed_df['engagement_linkedin_no_video']) == [10, 20, 30, 40, 50]

    def test_preprocess_posts_data_engagement_rate_no_nan_without_followers(self):
        """Regression: pd.Series(1) default for missing follower column causes
        engagement_rate to be NaN for every row except index 0. When the
        followers column is absent the engagement_rate column must not be
        computed at all (not silently all-zero / all-NaN via a stale default)."""
        posts_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=4),
            'num_likes_linkedin_no_video': [10, 20, 30, 40],
            'num_comments_linkedin_no_video': [1, 2, 3, 4],
            'num_reshares_linkedin_no_video': [0, 0, 0, 0],
            # num_followers_linkedin deliberately absent
        })

        processed_df = self.data_loader.preprocess_posts_data(posts_df)

        # engagement column must be present and fully populated
        assert 'engagement_linkedin_no_video' in processed_df.columns
        assert processed_df['engagement_linkedin_no_video'].isna().sum() == 0
        # engagement_rate must NOT be computed when followers are absent
        assert 'engagement_rate_linkedin_no_video' not in processed_df.columns
    
    def test_preprocess_profile_data(self):
        """Test profile data preprocessing."""
        # Mock data
        profile_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_followers_linkedin': [1000, 1050]
        })
        
        # Test
        processed_df = self.data_loader.preprocess_profile_data(profile_df)
        
        assert 'follower_growth_linkedin' in processed_df.columns
        assert 'follower_growth_rate_linkedin' in processed_df.columns
        assert 'day_of_week' in processed_df.columns
    
    def test_create_platform_dataset(self):
        """Test platform dataset creation.

        ``create_platform_dataset`` consumes *preprocessed* frames (it selects
        the temporal columns ``day_of_week``/``month``/``quarter``/``year``/
        ``is_weekend`` and the ``follower_growth_*`` columns that only
        ``preprocess_*`` produce), exactly as the real pipeline does in
        ``cli.py``. Feed it raw frames through the preprocessing step rather
        than hand-rolling half-preprocessed mocks.
        """
        # Raw mock data, as it would arrive from Supabase
        posts_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_likes_linkedin_no_video': [85, 130],
            'num_comments_linkedin_no_video': [10, 15],
            'num_reshares_linkedin_no_video': [5, 5]
        })
        profile_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_followers_linkedin': [1000, 1050]
        })

        # Preprocess first, mirroring the real pipeline contract
        posts_df = self.data_loader.preprocess_posts_data(posts_df)
        profile_df = self.data_loader.preprocess_profile_data(profile_df)

        # Test
        platform_df = self.data_loader.create_platform_dataset('linkedin', posts_df, profile_df)

        assert len(platform_df) == 2
        assert 'engagement_linkedin_no_video' in platform_df.columns
        assert 'num_followers_linkedin' in platform_df.columns
    
    def test_create_cross_platform_dataset(self):
        """Test cross-platform dataset creation."""
        # Mock data
        posts_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'engagement_linkedin_no_video': [100, 150],
            'engagement_instagram_no_video': [80, 120]
        })
        profile_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'num_followers_linkedin': [1000, 1050],
            'num_followers_instagram': [800, 850]
        })
        
        # Test
        cross_platform_df = self.data_loader.create_cross_platform_dataset(posts_df, profile_df)
        
        assert 'total_engagement_no_video' in cross_platform_df.columns
        assert 'avg_engagement_no_video' in cross_platform_df.columns
        assert 'max_engagement_no_video' in cross_platform_df.columns
    
    def test_get_feature_columns(self):
        """Test feature columns retrieval."""
        # Test with platform
        feature_cols = self.data_loader.get_feature_columns('linkedin')
        assert 'day_of_week' in feature_cols
        assert 'engagement_linkedin_no_video' in feature_cols or 'engagement_linkedin_video' in feature_cols
        
        # Test without platform
        feature_cols = self.data_loader.get_feature_columns()
        assert 'day_of_week' in feature_cols
        assert 'total_engagement_no_video' in feature_cols or 'total_engagement_video' in feature_cols