#!/usr/bin/env python3
"""
DIRECT PARQUET FIXER - Fix negative RECORD_DATE values in existing corpus
This is MUCH faster than regenerating the entire corpus!
"""

import pandas as pd
import os
from pathlib import Path

def fix_record_dates_in_parquet(file_path: str):
    """Fix negative RECORD_DATE values by adjusting reference date"""
    
    print(f"🔧 Fixing {file_path}...")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Check current RECORD_DATE range
    min_date = df['RECORD_DATE'].min()
    max_date = df['RECORD_DATE'].max()
    negative_count = (df['RECORD_DATE'] < 0).sum()
    
    print(f"   Before: min={min_date}, max={max_date}, negatives={negative_count}")
    
    if negative_count > 0:
        # Calculate the offset needed
        # Old reference: 1980-01-01 (day 0)
        # New reference: 1941-01-01 (day 0)
        # Difference: 39 years = 39 * 365.25 ≈ 14,245 days
        
        offset = 39 * 365 + 9  # 39 years + leap days (approximate)
        
        # Add offset to make all values positive
        df['RECORD_DATE'] = df['RECORD_DATE'] + offset
        
        # Verify fix
        new_min = df['RECORD_DATE'].min()
        new_max = df['RECORD_DATE'].max()
        new_negatives = (df['RECORD_DATE'] < 0).sum()
        
        print(f"   After:  min={new_min}, max={new_max}, negatives={new_negatives}")
        
        # Create backup
        backup_path = file_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"   📁 Creating backup: {backup_path}")
            os.rename(file_path, backup_path)
        
        # Save fixed version
        df.to_parquet(file_path)
        print(f"   ✅ Fixed and saved!")
        
    else:
        print(f"   ✅ No negative values found, skipping")

def main():
    """Fix all parquet files in the corpus"""
    
    corpus_dir = Path("data/processed/corpus/startup_corpus/sentences")
    
    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        return
    
    # Find all parquet files
    parquet_files = list(corpus_dir.rglob("*.parquet"))
    
    if not parquet_files:
        print("❌ No parquet files found!")
        return
    
    print(f"🎯 Found {len(parquet_files)} parquet files to fix:")
    for file in parquet_files:
        print(f"   📄 {file}")
    
    print("\n🔧 FIXING PARQUET FILES...")
    print("=" * 60)
    
    for file_path in parquet_files:
        try:
            fix_record_dates_in_parquet(str(file_path))
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print("\n🎉 ALL DONE!")
    print("Now you can restart training with:")
    print("CUDA_VISIBLE_DEVICES=3 python step_6_finetune_survival.py --quick-test --run")

if __name__ == "__main__":
    main()
