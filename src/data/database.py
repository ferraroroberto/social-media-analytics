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
    
    def _get_table_data(self, table: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        try:
            response = self.client.table(table).select("*").execute()
            df = pd.DataFrame(response.data)

            if start_date or end_date:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]

            logger.info(f"Loaded {len(df)} {table} records")
            return df

        except Exception as e:
            logger.error(f"Error loading {table} data: {e}")
            raise

    def get_posts_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load posts data from the consolidated posts table.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with posts data
        """
        return self._get_table_data("posts", start_date, end_date)

    def get_profile_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load profile data from the consolidated profile table.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with profile data
        """
        return self._get_table_data("profile", start_date, end_date)
    

def get_supabase_client() -> SupabaseClient:
    """Get a Supabase client instance."""
    return SupabaseClient()