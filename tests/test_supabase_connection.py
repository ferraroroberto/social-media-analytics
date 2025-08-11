#!/usr/bin/env python3
"""
Simple Supabase Connection Test
==============================

This script tests your Supabase connection and shows:
- Connection status
- Available tables
- Table structure
- First few records

Run this to verify your setup before starting the project.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ Error: Missing environment variables!")
        print("Please create a .env file with:")
        print("SUPABASE_URL=your_supabase_url")
        print("SUPABASE_KEY=your_anon_key")
        print("SUPABASE_SERVICE_ROLE_KEY=your_service_role_key")
        return None, None, None
    
    return url, key, service_key

def test_connection(url: str, key: str) -> Client:
    """Test Supabase connection"""
    try:
        print(f"🔌 Connecting to Supabase...")
        print(f"   URL: {url}")
        print(f"   Key: {key[:20]}...")
        
        client = create_client(url, key)
        
        # Test connection by getting a simple response
        response = client.table("posts").select("count", count="exact").limit(1).execute()
        print("✅ Connection successful!")
        return client
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def show_table_info(client: Client, table_name: str):
    """Show table structure and sample data"""
    try:
        print(f"\n📊 Table: {table_name}")
        print("=" * 50)
        
        # Get table structure (first record to see columns)
        response = client.table(table_name).select("*").limit(1).execute()
        
        if not response.data:
            print(f"   No data found in {table_name}")
            return
        
        # Show columns
        first_record = response.data[0]
        print(f"   Columns ({len(first_record)}):")
        for col, value in first_record.items():
            col_type = type(value).__name__
            print(f"     - {col}: {col_type}")
        
        # Get sample data (first 3 records)
        response = client.table(table_name).select("*").limit(3).execute()
        
        print(f"\n   Sample Data (first {len(response.data)} records):")
        for i, record in enumerate(response.data, 1):
            print(f"     Record {i}:")
            for col, value in list(record.items())[:5]:  # Show first 5 columns
                print(f"       {col}: {value}")
            if len(record) > 5:
                print(f"       ... and {len(record) - 5} more columns")
            print()
        
        # Get total count
        count_response = client.table(table_name).select("count", count="exact").limit(1).execute()
        total_count = count_response.count if hasattr(count_response, 'count') else "Unknown"
        print(f"   Total records: {total_count}")
        
    except Exception as e:
        print(f"   ❌ Error reading table {table_name}: {e}")

def main():
    """Main function"""
    print("🚀 Supabase Connection Test")
    print("=" * 40)
    
    # Load environment
    url, key, service_key = load_environment()
    if not url or not key:
        sys.exit(1)
    
    # Test connection
    client = test_connection(url, key)
    if not client:
        sys.exit(1)
    
    # List of tables to check (based on your project structure)
    tables_to_check = [
        "posts",           # Main consolidated posts table
        "profile",         # Main consolidated profile table
        "linkedin_posts",  # Platform-specific tables
        "instagram_posts",
        "twitter_posts",
        "notion_editorial" # Notion-synced tables
    ]
    
    print(f"\n🔍 Checking {len(tables_to_check)} tables...")
    
    # Check each table
    for table in tables_to_check:
        show_table_info(client, table)
    
    print("\n🎉 Connection test completed!")
    print("\nNext steps:")
    print("1. If you see table structures and data, you're ready to go!")
    print("2. If some tables are missing, that's normal - they'll be created as needed")
    print("3. Run the project with: python -m src.main")

if __name__ == "__main__":
    main()
