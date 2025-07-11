import pandas as pd

def debug_survival_dataset():
    """Debug why no examples are being created"""
    
    # 1. Check corpus sentences structure
    print("=== DEBUGGING CORPUS SENTENCES ===")
    train_sentences = pd.read_parquet("data/processed/corpus/startup_corpus/sentences/train/sentences.parquet")
    print(f"Train sentences shape: {train_sentences.shape}")
    print(f"Train sentences index type: {type(train_sentences.index)}")
    print(f"Train sentences index name: {train_sentences.index.name}")
    print(f"Sample company IDs from corpus:")
    print(train_sentences.index.unique()[:10])
    
    # 2. Check company data structure  
    print("\n=== DEBUGGING COMPANY DATA ===")
    company_df = pd.read_csv("data/cleaned/cleaned_startup/company_base_cleaned.csv")
    print(f"Company data shape: {company_df.shape}")
    print(f"Company data columns: {list(company_df.columns)}")
    
    # Check if there's a UUID column
    if 'uuid' in company_df.columns:
        print(f"Sample UUIDs from company data:")
        print(company_df['uuid'].head(10).tolist())
        
        # Check for overlap
        corpus_ids = set(train_sentences.index.unique())
        company_ids = set(company_df['uuid'].dropna())
        overlap = corpus_ids.intersection(company_ids)
        print(f"\nOverlap between corpus and company data: {len(overlap)} companies")
        print(f"Total corpus companies: {len(corpus_ids)}")
        print(f"Total company data companies: {len(company_ids)}")
        
        if len(overlap) == 0:
            print("\n❌ NO OVERLAP FOUND!")
            print("Sample corpus IDs:", list(corpus_ids)[:5])
            print("Sample company IDs:", list(company_ids)[:5])
        else:
            print(f"\n✅ Found {len(overlap)} overlapping companies")
    
    else:
        print("Available columns in company data:")
        for col in company_df.columns:
            print(f"  - {col}")

if __name__ == "__main__":
    debug_survival_dataset()
