"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from .database import SupabaseClient
from ..constants import PLATFORMS, CONTENT_TYPES
from ..features.feature_engineering import add_cross_platform_engagement_features

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for content performance prediction."""

    def __init__(self, supabase_client: Optional[SupabaseClient] = None):
        """Initialize data loader."""
        self.supabase_client = supabase_client or SupabaseClient()
        self.platforms = list(PLATFORMS)
        self.content_types = list(CONTENT_TYPES)
        
    def load_consolidated_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load consolidated posts and profile data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of (posts_df, profile_df)
        """
        posts_df = self.supabase_client.get_posts_data(start_date, end_date)
        profile_df = self.supabase_client.get_profile_data(start_date, end_date)
        
        return posts_df, profile_df
    
    def preprocess_posts_data(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess posts data for analysis.
        
        Args:
            posts_df: Raw posts DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = posts_df.copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Create platform-specific engagement metrics
        for platform in self.platforms:
            for content_type in self.content_types:
                # Likes
                likes_col = f'num_likes_{platform}_{content_type}'
                if likes_col in df.columns:
                    comments_col = f'num_comments_{platform}_{content_type}'
                    reshares_col = f'num_reshares_{platform}_{content_type}'
                    df[f'engagement_{platform}_{content_type}'] = (
                        df[likes_col].fillna(0) +
                        (df[comments_col].fillna(0) if comments_col in df.columns else 0) +
                        (df[reshares_col].fillna(0) if reshares_col in df.columns else 0)
                    )

                # Engagement rate (computed here only when follower data is
                # already in the posts frame; the normal path is to compute it
                # after the profile merge in create_platform_dataset).
                engagement_col = f'engagement_{platform}_{content_type}'
                followers_col = f'num_followers_{platform}'
                if engagement_col in df.columns and followers_col in df.columns:
                    df[f'engagement_rate_{platform}_{content_type}'] = (
                        df[engagement_col] /
                        df[followers_col].replace(0, 1).fillna(1)
                    )
        
        # Create aggregated metrics
        for platform in self.platforms:
            platform_engagement_cols = [col for col in df.columns if col.startswith(f'engagement_{platform}_')]
            if platform_engagement_cols:
                df[f'total_engagement_{platform}'] = df[platform_engagement_cols].sum(axis=1)
        
        # Add temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Preprocessed {len(df)} posts records")
        return df
    
    def preprocess_profile_data(self, profile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess profile data for analysis.
        
        Args:
            profile_df: Raw profile DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = profile_df.copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate follower growth
        for platform in self.platforms:
            followers_col = f'num_followers_{platform}'
            if followers_col in df.columns:
                df[f'follower_growth_{platform}'] = df[followers_col].diff()
                df[f'follower_growth_rate_{platform}'] = df[followers_col].pct_change()
        
        # Add temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        logger.info(f"Preprocessed {len(df)} profile records")
        return df
    
    def create_platform_dataset(self, platform: str, posts_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataset for a specific platform.
        
        Args:
            platform: Platform name
            posts_df: Preprocessed posts DataFrame
            profile_df: Preprocessed profile DataFrame
            
        Returns:
            Platform-specific dataset
        """
        # Merge posts and profile data
        df = posts_df.merge(profile_df, on='date', how='left', suffixes=('', '_profile'))

        # Compute engagement_rate after the merge so follower data is available.
        # preprocess_posts_data only sets it when followers are already in the
        # posts frame (rare); the standard path computes it here.
        followers_col = f'num_followers_{platform}'
        for content_type in self.content_types:
            engagement_col = f'engagement_{platform}_{content_type}'
            rate_col = f'engagement_rate_{platform}_{content_type}'
            if engagement_col in df.columns and followers_col in df.columns and rate_col not in df.columns:
                df[rate_col] = df[engagement_col] / df[followers_col].replace(0, 1).fillna(1)

        # Select platform-specific columns (only those that actually exist)
        platform_cols = ['date', 'day_of_week', 'month', 'quarter', 'year', 'is_weekend']

        for content_type in self.content_types:
            engagement_col = f'engagement_{platform}_{content_type}'
            if engagement_col in df.columns:
                for col in [
                    engagement_col,
                    f'engagement_rate_{platform}_{content_type}',
                    f'num_likes_{platform}_{content_type}',
                    f'num_comments_{platform}_{content_type}',
                    f'num_reshares_{platform}_{content_type}',
                ]:
                    if col in df.columns:
                        platform_cols.append(col)

        # Profile metrics (only those that actually exist)
        if followers_col in df.columns:
            for col in [
                followers_col,
                f'follower_growth_{platform}',
                f'follower_growth_rate_{platform}',
            ]:
                if col in df.columns:
                    platform_cols.append(col)

        platform_df = df[platform_cols].copy()
        platform_df = platform_df.dropna(subset=[col for col in platform_cols if col.startswith(f'engagement_{platform}')])
        
        logger.info(f"Created {platform} dataset with {len(platform_df)} records")
        return platform_df
    
    def create_cross_platform_dataset(self, posts_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a cross-platform dataset for analysis.
        
        Args:
            posts_df: Preprocessed posts DataFrame
            profile_df: Preprocessed profile DataFrame
            
        Returns:
            Cross-platform dataset
        """
        df = posts_df.merge(profile_df, on='date', how='left', suffixes=('', '_profile'))

        # The cross-platform aggregation math (total/avg/max/std + per-platform
        # share) lives in exactly one place — the shared helper in
        # feature_engineering — so the two entry points cannot drift.
        df = add_cross_platform_engagement_features(
            df, platforms=self.platforms, content_types=self.content_types
        )

        logger.info(f"Created cross-platform dataset with {len(df)} records")
        return df
    
    def get_feature_columns(self, platform: Optional[str] = None) -> List[str]:
        """
        Get list of feature columns for modeling.
        
        Args:
            platform: Optional platform filter
            
        Returns:
            List of feature column names
        """
        base_features = ['day_of_week', 'month', 'quarter', 'year', 'is_weekend']
        
        if platform:
            platform_features = []
            for content_type in self.content_types:
                platform_features.extend([
                    f'engagement_{platform}_{content_type}',
                    f'engagement_rate_{platform}_{content_type}',
                    f'num_likes_{platform}_{content_type}',
                    f'num_comments_{platform}_{content_type}',
                    f'num_reshares_{platform}_{content_type}'
                ])
            
            profile_features = [
                f'num_followers_{platform}',
                f'follower_growth_{platform}',
                f'follower_growth_rate_{platform}'
            ]
            
            return base_features + platform_features + profile_features
        else:
            # Cross-platform features
            cross_platform_features = []
            for content_type in self.content_types:
                cross_platform_features.extend([
                    f'total_engagement_{content_type}',
                    f'avg_engagement_{content_type}',
                    f'max_engagement_{content_type}',
                    f'engagement_std_{content_type}'
                ])
            
            return base_features + cross_platform_features