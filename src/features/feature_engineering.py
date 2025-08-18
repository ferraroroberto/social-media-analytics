"""Feature engineering for content performance prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from textblob import TextBlob

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for content performance prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.vectorizers = {}
        self.feature_columns = []
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        if date_col not in df.columns:
            logger.warning(f"Date column {date_col} not found")
            return df
        
        # Handle missing dates
        if df[date_col].isnull().any():
            logger.warning(f"Found {df[date_col].isnull().sum()} missing dates, filling with forward fill")
            df[date_col] = df[date_col].fillna(method='ffill')
            # If still have missing values at the beginning, fill with backward fill
            if df[date_col].isnull().any():
                df[date_col] = df[date_col].fillna(method='bfill')
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        # Cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time-based features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        # Holiday features (basic implementation)
        df['is_holiday'] = self._is_holiday(df[date_col])
        
        logger.info("Created temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], lags: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features for time series analysis.
        
        Args:
            df: Input DataFrame
            target_cols: Target columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        df = df.sort_values('date')
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    # Create lag features
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                    # Rolling statistics with min_periods=1 to avoid NaN
                    df[f'{col}_rolling_mean_{lag}'] = df[col].rolling(window=lag, min_periods=1).mean()
                    df[f'{col}_rolling_std_{lag}'] = df[col].rolling(window=lag, min_periods=1).std()
                    df[f'{col}_rolling_min_{lag}'] = df[col].rolling(window=lag, min_periods=1).min()
                    df[f'{col}_rolling_max_{lag}'] = df[col].rolling(window=lag, min_periods=1).max()
                    
                    # Fill NaN values in lag features with appropriate defaults
                    df[f'{col}_lag_{lag}'] = df[f'{col}_lag_{lag}'].fillna(df[col].median())
                    
                    # Fill any remaining NaN values in rolling features
                    df[f'{col}_rolling_std_{lag}'] = df[f'{col}_rolling_std_{lag}'].fillna(0)
        
        logger.info(f"Created lag features for {len(target_cols)} columns with lags {lags}")
        return df
    
    def create_engagement_features(self, df: pd.DataFrame, platforms: List[str] = None) -> pd.DataFrame:
        """
        Create engagement-related features.
        
        Args:
            df: Input DataFrame
            platforms: List of platforms to process
            
        Returns:
            DataFrame with engagement features
        """
        df = df.copy()
        
        if platforms is None:
            platforms = ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        
        for platform in platforms:
            # Engagement rate features
            engagement_cols = [col for col in df.columns if col.startswith(f'engagement_{platform}')]
            follower_col = f'num_followers_{platform}'
            
            if engagement_cols and follower_col in df.columns:
                for col in engagement_cols:
                    # Handle division by zero and missing values
                    df[f'{col}_rate'] = df[col] / df[follower_col].replace(0, 1)
                    df[f'{col}_rate'] = df[f'{col}_rate'].fillna(0)
            
            # Engagement ratio features
            for content_type in ['no_video', 'video']:
                likes_col = f'num_likes_{platform}_{content_type}'
                comments_col = f'num_comments_{platform}_{content_type}'
                shares_col = f'num_reshares_{platform}_{content_type}'
                
                if likes_col in df.columns and comments_col in df.columns:
                    # Handle division by zero and missing values
                    df[f'{platform}_{content_type}_comment_ratio'] = (
                        df[comments_col] / df[likes_col].replace(0, 1)
                    )
                    df[f'{platform}_{content_type}_comment_ratio'] = df[f'{platform}_{content_type}_comment_ratio'].fillna(0)
                
                if likes_col in df.columns and shares_col in df.columns:
                    # Handle division by zero and missing values
                    df[f'{platform}_{content_type}_share_ratio'] = (
                        df[shares_col] / df[likes_col].replace(0, 1)
                    )
                    df[f'{platform}_{content_type}_share_ratio'] = df[f'{platform}_{content_type}_share_ratio'].fillna(0)
        
        logger.info("Created engagement features")
        return df
    
    def create_cross_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-platform comparison features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cross-platform features
        """
        df = df.copy()
        
        platforms = ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        content_types = ['no_video', 'video']
        
        for content_type in content_types:
            # Total engagement across platforms
            engagement_cols = [f'engagement_{platform}_{content_type}' for platform in platforms]
            available_cols = [col for col in engagement_cols if col in df.columns]
            
            if available_cols:
                df[f'total_engagement_{content_type}'] = df[available_cols].sum(axis=1)
                df[f'avg_engagement_{content_type}'] = df[available_cols].mean(axis=1)
                df[f'max_engagement_{content_type}'] = df[available_cols].max(axis=1)
                df[f'engagement_std_{content_type}'] = df[available_cols].std(axis=1)
                
                # Platform share
                for platform in platforms:
                    col = f'engagement_{platform}_{content_type}'
                    if col in df.columns:
                        # Handle division by zero and missing values
                        df[f'{platform}_share_{content_type}'] = (
                            df[col] / df[f'total_engagement_{content_type}'].replace(0, 1)
                        )
                        df[f'{platform}_share_{content_type}'] = df[f'{platform}_share_{content_type}'].fillna(0)
        
        # Cross-platform follower features
        follower_cols = [f'num_followers_{platform}' for platform in platforms]
        available_follower_cols = [col for col in follower_cols if col in df.columns]
        
        if available_follower_cols:
            df['total_followers'] = df[available_follower_cols].sum(axis=1)
            df['avg_followers'] = df[available_follower_cols].mean(axis=1)
            df['max_followers'] = df[available_follower_cols].max(axis=1)
            df['follower_std'] = df[available_follower_cols].std(axis=1)
            
            # Fill NaN values in follower features
            df['follower_std'] = df['follower_std'].fillna(0)
        
        logger.info("Created cross-platform features")
        return df
    
    def create_content_features(self, df: pd.DataFrame, text_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create content-related features.
        
        Args:
            df: Input DataFrame
            text_col: Text column for NLP features
            
        Returns:
            DataFrame with content features
        """
        df = df.copy()
        
        if text_col and text_col in df.columns:
            # Text length features
            df[f'{text_col}_length'] = df[text_col].str.len().fillna(0)
            df[f'{text_col}_word_count'] = df[text_col].str.split().str.len().fillna(0)
            df[f'{text_col}_char_count'] = df[text_col].str.replace(' ', '').str.len().fillna(0)
            
            # Sentiment analysis
            df[f'{text_col}_sentiment'] = df[text_col].apply(self._get_sentiment)
            
            # Text complexity
            df[f'{text_col}_avg_word_length'] = (
                df[text_col].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
            )
            df[f'{text_col}_avg_word_length'] = df[f'{text_col}_avg_word_length'].fillna(0)
            
            # Hashtag and mention features
            df[f'{text_col}_hashtag_count'] = df[text_col].str.count(r'#\w+').fillna(0)
            df[f'{text_col}_mention_count'] = df[text_col].str.count(r'@\w+').fillna(0)
            df[f'{text_col}_url_count'] = df[text_col].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+').fillna(0)
        
        logger.info("Created content features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        platforms = ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        
        for platform in platforms:
            for content_type in ['no_video', 'video']:
                likes_col = f'num_likes_{platform}_{content_type}'
                comments_col = f'num_comments_{platform}_{content_type}'
                shares_col = f'num_reshares_{platform}_{content_type}'
                
                if likes_col in df.columns:
                    # Interaction ratios
                    if comments_col in df.columns:
                        # Handle division by zero and missing values
                        df[f'{platform}_{content_type}_comment_like_ratio'] = (
                            df[comments_col] / df[likes_col].replace(0, 1)
                        )
                        df[f'{platform}_{content_type}_comment_like_ratio'] = df[f'{platform}_{content_type}_comment_like_ratio'].fillna(0)
                    
                    if shares_col in df.columns:
                        # Handle division by zero and missing values
                        df[f'{platform}_{content_type}_share_like_ratio'] = (
                            df[shares_col] / df[likes_col].replace(0, 1)
                        )
                        df[f'{platform}_{content_type}_share_like_ratio'] = df[f'{platform}_{content_type}_share_like_ratio'].fillna(0)
                    
                    # Total interactions
                    interaction_cols = [col for col in [likes_col, comments_col, shares_col] if col in df.columns]
                    if len(interaction_cols) > 1:
                        df[f'{platform}_{content_type}_total_interactions'] = df[interaction_cols].sum(axis=1)
        
        logger.info("Created interaction features")
        return df
    
    def create_trend_features(self, df: pd.DataFrame, window_sizes: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create trend-based features.
        
        Args:
            df: Input DataFrame
            window_sizes: List of window sizes for trend calculation
            
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        df = df.sort_values('date')
        
        engagement_cols = [col for col in df.columns if 'engagement' in col]
        
        for col in engagement_cols:
            for window in window_sizes:
                # Trend direction with min_periods=1 to avoid NaN
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_trend_{window}'] = rolling_mean - rolling_mean.shift(1)
                
                # Trend strength with min_periods=1 to avoid NaN
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_trend_strength_{window}'] = (
                    rolling_std / rolling_mean.replace(0, 1)
                )
                
                # Fill NaN values in trend features
                df[f'{col}_trend_{window}'] = df[f'{col}_trend_{window}'].fillna(0)
                df[f'{col}_trend_strength_{window}'] = df[f'{col}_trend_strength_{window}'].fillna(0)
        
        logger.info("Created trend features")
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str], scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to scale
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No feature columns found for scaling")
            return df
        
        # Handle missing values before scaling
        for col in available_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Scale features
        df_scaled = scaler.fit_transform(df[available_cols])
        df_scaled = pd.DataFrame(df_scaled, columns=available_cols, index=df.index)
        
        # Replace original columns with scaled ones
        for col in available_cols:
            df[f'{col}_scaled'] = df_scaled[col]
        
        # Store scaler for later use
        self.scalers[scaler_type] = scaler
        
        logger.info(f"Scaled {len(available_cols)} features using {scaler_type} scaler")
        return df
    
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Basic holiday detection (US holidays)."""
        # This is a simplified implementation
        # In production, you might want to use a proper holiday library
        holidays = []
        for date in dates:
            is_holiday = (
                date.month == 1 and date.day == 1 or  # New Year's Day
                date.month == 7 and date.day == 4 or  # Independence Day
                date.month == 12 and date.day == 25   # Christmas
            )
            holidays.append(is_holiday)
        
        return pd.Series(holidays, index=dates.index)
    
    def _get_sentiment(self, text: str) -> float:
        """Get sentiment score for text."""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    def get_feature_importance_ranking(self, feature_cols: List[str]) -> List[str]:
        """
        Get feature importance ranking based on domain knowledge.
        
        Args:
            feature_cols: List of feature columns
            
        Returns:
            Ranked list of features
        """
        # This is a simplified ranking based on domain knowledge
        # In practice, you would use actual feature importance from models
        
        high_importance = [
            'engagement_linkedin_no_video', 'engagement_instagram_no_video',
            'num_followers_linkedin', 'num_followers_instagram',
            'day_of_week', 'is_weekend'
        ]
        
        medium_importance = [
            'engagement_rate_linkedin_no_video', 'engagement_rate_instagram_no_video',
            'follower_growth_linkedin', 'follower_growth_instagram',
            'month', 'quarter'
        ]
        
        low_importance = [
            'year', 'is_holiday', 'is_month_start', 'is_month_end'
        ]
        
        # Rank features
        ranked_features = []
        for importance_level in [high_importance, medium_importance, low_importance]:
            for feature in importance_level:
                if feature in feature_cols:
                    ranked_features.append(feature)
        
        # Add remaining features
        remaining_features = [f for f in feature_cols if f not in ranked_features]
        ranked_features.extend(remaining_features)
        
        return ranked_features