#!/usr/bin/env python3
"""
Save Balanced Train Company IDs
Extracts and saves the balanced train company IDs using the same logic as step_4b_create_balanced_datamodule.py
This ensures all future runs use the exact same balanced train split.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for DataSplit import
sys.path.append('src')

def load_company_data():
    """Load company data and create survival labels"""
    company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.csv")
    if not company_path.exists():
        company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.pkl")
        company_df = pd.read_pickle(company_path)
    else:
        company_df = pd.read_csv(company_path, low_memory=False)
    
    # Create survival labels (same logic as data module)
    company_df = company_df[['COMPANY_ID', 'status', 'founded_on', 'closed_on']].copy()
    company_df['founded_on'] = pd.to_datetime(company_df['founded_on'], errors='coerce')
    company_df['closed_on'] = pd.to_datetime(company_df.get('closed_on'), errors='coerce')
    
    founded_valid_mask = company_df['founded_on'].notna()
    survived_mask = (
        company_df['status'].isin(['operating', 'acquired', 'ipo']) & founded_valid_mask
    )
    died_mask = (
        (company_df['status'] == 'closed') & (company_df['closed_on'].notna()) & founded_valid_mask
    )
    
    company_df['survival_label'] = None
    company_df.loc[survived_mask, 'survival_label'] = 1
    company_df.loc[died_mask, 'survival_label'] = 0
    
    valid_mask = company_df['survival_label'].notna()
    company_df = company_df.loc[valid_mask, ['COMPANY_ID', 'survival_label']]
    
    return company_df

def load_data_split():
    """Load the train/val/test split"""
    split_path = Path("data/processed/populations/startups/data_split/result.pkl")
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    return split

def create_balanced_train_ids():
    """Create balanced train IDs using the same logic as the data module"""
    print("Loading company data...")
    company_df = load_company_data()
    
    print("Loading data split...")
    split = load_data_split()
    
    # Get train companies
    train_ids = set(split.train)
    train_companies = company_df[company_df['COMPANY_ID'].isin(train_ids)]
    
    print(f"Train companies before balancing: {len(train_companies)}")
    print(f"Survived in train: {(train_companies['survival_label'] == 1).sum()}")
    print(f"Died in train: {(train_companies['survival_label'] == 0).sum()}")
    
    # Apply balancing (same logic as data module)
    survived_ids = train_companies[train_companies['survival_label'] == 1]['COMPANY_ID'].unique()
    died_ids = train_companies[train_companies['survival_label'] == 0]['COMPANY_ID'].unique()
    
    n_survived = len(survived_ids)
    n_died = len(died_ids)
    
    print(f"Survived companies: {n_survived}")
    print(f"Died companies: {n_died}")
    
    if n_survived > n_died and n_died > 0:
        # Set random seed for reproducible balancing
        np.random.seed(42)
        
        # Downsample survived to match died
        survived_downsampled = np.random.choice(survived_ids, size=n_died, replace=False)
        balanced_train_ids = np.concatenate([survived_downsampled, died_ids])
        
        print(f"Balanced train companies: {len(balanced_train_ids)}")
        print(f"  - Survived (downsampled): {len(survived_downsampled)}")
        print(f"  - Died (all): {len(died_ids)}")
        
        return balanced_train_ids
    else:
        print("No balancing needed or not enough samples")
        return train_companies['COMPANY_ID'].unique()

def main():
    """Main function"""
    print("🎯 SAVING BALANCED TRAIN COMPANY IDs")
    print("=" * 50)
    
    # Create balanced train IDs
    balanced_train_ids = create_balanced_train_ids()
    
    # Save to file
    output_path = Path("balanced_train_ids.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(balanced_train_ids, f)
    
    print(f"\n✅ Balanced train IDs saved to: {output_path}")
    print(f"📊 Total balanced train companies: {len(balanced_train_ids)}")
    
    # Also save as text file for easy inspection
    txt_path = Path("balanced_train_ids.txt")
    with open(txt_path, "w") as f:
        for company_id in balanced_train_ids:
            f.write(f"{company_id}\n")
    
    print(f"📝 Also saved as text file: {txt_path}")
    print("\n🎉 Ready for reproducible training!")

if __name__ == "__main__":
    main() 