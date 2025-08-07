"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from .database import SupabaseClient

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for content performance prediction."""
    
    def __init__(self, supabase_client: Optional[SupabaseClient] = None):
        """Initialize data loader."""
        self.supabase_client = supabase_client or SupabaseClient()
        self.platforms = ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        self.content_types = ['no_video', 'video']
        
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
                    df[f'engagement_{platform}_{content_type}'] = (
                        df[likes_col].fillna(0) + 
                        df.get(f'num_comments_{platform}_{content_type}', pd.Series(0)).fillna(0) + 
                        df.get(f'num_reshares_{platform}_{content_type}', pd.Series(0)).fillna(0)
                    )
                
                # Engagement rate (if we have follower data)
                if f'engagement_{platform}_{content_type}' in df.columns:
                    df[f'engagement_rate_{platform}_{content_type}'] = (
                        df[f'engagement_{platform}_{content_type}'] / 
                        df.get(f'num_followers_{platform}', pd.Series(1)).fillna(1)
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
        
        # Select platform-specific columns
        platform_cols = ['date', 'day_of_week', 'month', 'quarter', 'year', 'is_weekend']
        
        for content_type in self.content_types:
            # Engagement metrics
            engagement_col = f'engagement_{platform}_{content_type}'
            if engagement_col in df.columns:
                platform_cols.extend([
                    engagement_col,
                    f'engagement_rate_{platform}_{content_type}',
                    f'num_likes_{platform}_{content_type}',
                    f'num_comments_{platform}_{content_type}',
                    f'num_reshares_{platform}_{content_type}'
                ])
        
        # Profile metrics
        followers_col = f'num_followers_{platform}'
        if followers_col in df.columns:
            platform_cols.extend([
                followers_col,
                f'follower_growth_{platform}',
                f'follower_growth_rate_{platform}'
            ])
        
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
        
        # Create cross-platform features
        for content_type in self.content_types:
            engagement_cols = [f'engagement_{platform}_{content_type}' for platform in self.platforms]
            available_cols = [col for col in engagement_cols if col in df.columns]
            
            if available_cols:
                df[f'total_engagement_{content_type}'] = df[available_cols].sum(axis=1)
                df[f'avg_engagement_{content_type}'] = df[available_cols].mean(axis=1)
                df[f'max_engagement_{content_type}'] = df[available_cols].max(axis=1)
                df[f'engagement_std_{content_type}'] = df[available_cols].std(axis=1)
        
        # Platform comparison features
        for platform in self.platforms:
            for content_type in self.content_types:
                engagement_col = f'engagement_{platform}_{content_type}'
                if engagement_col in df.columns:
                    total_engagement_col = f'total_engagement_{content_type}'
                    if total_engagement_col in df.columns:
                        df[f'{platform}_share_{content_type}'] = (
                            df[engagement_col] / df[total_engagement_col].replace(0, 1)
                        )
        
        logger.info(f"Created cross-platform dataset with {len(df)} records")
        return df
    
    def load_notion_content_data(self, table_name: str = 'notion_posts') -> pd.DataFrame:
        """
        Load and preprocess Notion content data.
        
        Args:
            table_name: Notion table name
            
        Returns:
            Preprocessed Notion data
        """
        df = self.supabase_client.get_notion_data(table_name)
        
        if df.empty:
            logger.warning(f"No data found in {table_name}")
            return df
        
        # Convert timestamps
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])
        if 'last_edited_time' in df.columns:
            df['last_edited_time'] = pd.to_datetime(df['last_edited_time'])
        
        # Extract JSON data if needed
        if 'notion_data_jsonb' in df.columns:
            # This would need to be customized based on the actual JSON structure
            pass
        
        logger.info(f"Loaded {len(df)} records from {table_name}")
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