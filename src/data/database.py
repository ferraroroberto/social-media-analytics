"""Database connection and data loading utilities for Supabase."""

import os
from typing import Dict, List, Optional, Any
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()


class SupabaseClient:
    """Supabase client for data operations."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized")
    
    def get_posts_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load posts data from the consolidated posts table.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with posts data
        """
        query = "SELECT * FROM posts"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("date >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("date <= %s")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        try:
            response = self.client.table("posts").select("*").execute()
            df = pd.DataFrame(response.data)
            
            if start_date or end_date:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            
            logger.info(f"Loaded {len(df)} posts records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading posts data: {e}")
            raise
    
    def get_profile_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load profile data from the consolidated profile table.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with profile data
        """
        query = "SELECT * FROM profile"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("date >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("date <= %s")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        try:
            response = self.client.table("profile").select("*").execute()
            df = pd.DataFrame(response.data)
            
            if start_date or end_date:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            
            logger.info(f"Loaded {len(df)} profile records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading profile data: {e}")
            raise
    
    def get_notion_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from Notion-synced tables.
        
        Args:
            table_name: Name of the Notion table (e.g., 'notion_posts', 'notion_editorial')
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with Notion data
        """
        try:
            query = self.client.table(table_name).select("*")
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            df = pd.DataFrame(response.data)
            
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {table_name}: {e}")
            raise
    
    def get_platform_data(self, platform: str, data_type: str = "posts") -> pd.DataFrame:
        """
        Load platform-specific data.
        
        Args:
            platform: Platform name (linkedin, instagram, twitter, substack, threads)
            data_type: Type of data (posts or profile)
            
        Returns:
            DataFrame with platform data
        """
        table_name = f"{platform}_{data_type}"
        
        try:
            response = self.client.table(table_name).select("*").execute()
            df = pd.DataFrame(response.data)
            
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {table_name}: {e}")
            raise


def get_supabase_client() -> SupabaseClient:
    """Get a Supabase client instance."""
    return SupabaseClient()