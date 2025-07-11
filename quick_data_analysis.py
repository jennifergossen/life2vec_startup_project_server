#!/usr/bin/env python3
"""
Quick analysis of status values and data structure
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data():
    print("🔍 ANALYZING CLEANED DATA STRUCTURE")
    print("=" * 60)
    
    # Load the cleaned data
    company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.csv")
    events_path = Path("data/cleaned/cleaned_startup/combined_events_cleaned.csv")
    
    if not company_path.exists() or not events_path.exists():
        print("❌ Cleaned data files not found!")
        print(f"Looking for: {company_path}")
        print(f"Looking for: {events_path}")
        return
    
    print("📊 Loading data...")
    companies = pd.read_csv(company_path)
    events = pd.read_csv(events_path)
    
    print(f"✅ Companies: {len(companies):,} rows")
    print(f"✅ Events: {len(events):,} rows")
    
    # 1. ANALYZE STATUS VALUES
    print(f"\n📋 STATUS ANALYSIS:")
    print("=" * 30)
    status_counts = companies['status'].value_counts(dropna=False)
    print("Unique status values:")
    for status, count in status_counts.items():
        print(f"  '{status}': {count:,} ({count/len(companies)*100:.1f}%)")
    
    # 2. CHECK ID COLUMNS
    print(f"\n🆔 ID COLUMN ANALYSIS:")
    print("=" * 30)
    
    # Check company ID columns
    company_id_cols = [col for col in companies.columns if 'id' in col.lower() or 'uuid' in col.lower()]
    print(f"Company ID columns: {company_id_cols}")
    
    # Check events ID columns  
    event_id_cols = [col for col in events.columns if 'id' in col.lower() or 'uuid' in col.lower()]
    print(f"Events ID columns: {event_id_cols}")
    
    # Show which column is used to link them
    if 'COMPANY_ID' in events.columns and 'COMPANY_ID' in companies.columns:
        print("✅ Using COMPANY_ID to link companies and events")
        # Check if they match
        company_ids_in_events = set(events['COMPANY_ID'].unique())
        company_ids_in_companies = set(companies['COMPANY_ID'].unique())
        overlap = len(company_ids_in_events & company_ids_in_companies)
        print(f"   Companies with events: {overlap:,}")
        print(f"   Companies without events: {len(company_ids_in_companies - company_ids_in_events):,}")
    
    # 3. CHECK DATE COLUMNS AND PREPROCESSING
    print(f"\n📅 DATE COLUMN ANALYSIS:")
    print("=" * 30)
    
    # Check if dates are already converted
    date_cols = ['founded_on', 'closed_on', 'last_funding_on']
    for col in date_cols:
        if col in companies.columns:
            dtype = companies[col].dtype
            sample_vals = companies[col].dropna().head(3).tolist()
            print(f"  {col}: dtype={dtype}, samples={sample_vals}")
    
    if 'RECORD_DATE' in events.columns:
        dtype = events['RECORD_DATE'].dtype  
        sample_vals = events['RECORD_DATE'].dropna().head(3).tolist()
        print(f"  RECORD_DATE: dtype={dtype}, samples={sample_vals}")
    
    # 4. CHECK IF AGE/BINNING ALREADY EXISTS
    print(f"\n🔢 PREPROCESSED COLUMNS:")
    print("=" * 30)
    
    # Look for age columns
    age_cols = [col for col in companies.columns if 'age' in col.lower()]
    print(f"Age columns: {age_cols}")
    
    # Look for binned columns
    binned_cols = [col for col in companies.columns if 'binned' in col.lower()]
    print(f"Binned columns in companies: {binned_cols}")
    
    binned_event_cols = [col for col in events.columns if 'binned' in col.lower()]
    print(f"Binned columns in events: {binned_event_cols[:10]}...")  # Show first 10
    
    # 5. SAMPLE DATA PREVIEW
    print(f"\n📖 SAMPLE COMPANY DATA:")
    print("=" * 30)
    sample_companies = companies[['COMPANY_ID', 'status', 'founded_on', 'closed_on']].head()
    print(sample_companies.to_string())
    
    print(f"\n📖 SAMPLE EVENT DATA:")
    print("=" * 30)
    sample_events = events[['COMPANY_ID', 'EVENT_TYPE', 'RECORD_DATE']].head()
    print(sample_events.to_string())
    
    # 6. CHECK CLOSED_ON CONSISTENCY
    print(f"\n🔍 STATUS vs CLOSED_ON CONSISTENCY:")
    print("=" * 30)
    
    # Check if closed companies have closed_on dates
    closed_companies = companies[companies['status'] == 'closed']
    closed_with_date = closed_companies['closed_on'].notna().sum()
    print(f"Companies with status='closed': {len(closed_companies):,}")
    print(f"Of those, have closed_on date: {closed_with_date:,} ({closed_with_date/len(closed_companies)*100:.1f}%)")
    
    # Check if non-closed have closed_on dates (should be rare)
    non_closed = companies[companies['status'] != 'closed']
    non_closed_with_date = non_closed['closed_on'].notna().sum()
    print(f"Non-closed companies with closed_on date: {non_closed_with_date:,}")

if __name__ == "__main__":
    analyze_data()
