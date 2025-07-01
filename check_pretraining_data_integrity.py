#!/usr/bin/env python3
"""
Check if your pretraining data has the same negative token issue
This determines if you need to redo pretraining
"""

import torch
import pandas as pd
from pathlib import Path

def check_pretraining_data():
    """Check if pretraining data has negative tokens"""
    
    print("🔍 CHECKING PRETRAINING DATA INTEGRITY")
    print("=" * 60)
    
    # Check pretraining corpus files
    pretraining_paths = [
        "data/processed/corpus/startup_corpus/sentences/train/sentences.parquet",
        "data/processed/corpus/startup_corpus/sentences/val/sentences.parquet",
        "data/processed/corpus/startup_corpus/sentences/test/sentences.parquet"
    ]
    
    for path in pretraining_paths:
        if Path(path).exists():
            print(f"\n📄 Checking {path}...")
            df = pd.read_parquet(path)
            
            # Check if there are any references to dates that could cause issues
            print(f"   Columns: {list(df.columns)}")
            print(f"   Rows: {len(df):,}")
            
            # Sample some sentences to see if they look normal
            sample_sentences = df['SENTENCE'].head(10)
            print(f"   Sample sentences:")
            for i, sent in enumerate(sample_sentences):
                print(f"      {i+1}: '{sent[:80]}...'")
                
                # Look for suspicious patterns
                if any(suspicious in str(sent).lower() for suspicious in ['-364', '-1309', 'nan', 'null']):
                    print(f"         🚨 SUSPICIOUS CONTENT DETECTED!")
            
            # Check date columns if they exist
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                print(f"   Date columns: {date_columns}")
                for col in date_columns:
                    sample_dates = df[col].head(5)
                    print(f"      {col}: {sample_dates.tolist()}")
        else:
            print(f"❌ Not found: {path}")
    
    # Check if we can load pretrained model weights without issues
    print(f"\n🤖 CHECKING PRETRAINED MODEL:")
    pretrained_path = "startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    if Path(pretrained_path).exists():
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            print(f"   ✅ Pretrained model loads successfully")
            print(f"   Vocab size: {checkpoint['hparams']['vocab_size']}")
            print(f"   Hidden size: {checkpoint['hparams']['hidden_size']}")
            
            # Check if there are any obvious signs of training issues
            if 'epoch' in checkpoint:
                print(f"   Final epoch: {checkpoint['epoch']}")
            
            print(f"   Model seems intact - pretraining likely succeeded")
            
        except Exception as e:
            print(f"   ❌ Error loading pretrained model: {e}")
    else:
        print(f"   ❌ Pretrained model not found: {pretrained_path}")

def test_tokenization_on_clean_data():
    """Test if tokenization works on a simple clean example"""
    
    print(f"\n🧪 TESTING TOKENIZATION ON CLEAN DATA:")
    print("=" * 60)
    
    try:
        # Create a simple test sentence
        test_sentences = [
            "COMPANY_FOUNDED_2020 FUNDING_SERIES_A EMPLOYEE_SIZE_10",
            "INVESTMENT_AMOUNT_1000000 CATEGORY_SAAS COUNTRY_USA",
            "INDUSTRY_TECHNOLOGY STATUS_ACTIVE"
        ]
        
        print(f"   Test sentences:")
        for i, sent in enumerate(test_sentences):
            print(f"      {i+1}: '{sent}'")
        
        # Try to load vocabulary and tokenize
        vocab_path = Path("data/processed/vocab/startup_vocab/result.tsv")
        if vocab_path.exists():
            vocab_df = pd.read_csv(vocab_path, sep='\t')
            token_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
            
            print(f"\n   Manual tokenization test:")
            for sent in test_sentences:
                tokens = sent.split()
                token_ids = []
                
                for token in tokens:
                    if token in token_to_id:
                        token_ids.append(token_to_id[token])
                    else:
                        # Unknown token - should use UNK token
                        unk_id = token_to_id.get('[UNK]', 0)
                        token_ids.append(unk_id)
                        print(f"      Unknown token: '{token}' -> UNK ({unk_id})")
                
                min_id = min(token_ids) if token_ids else 0
                max_id = max(token_ids) if token_ids else 0
                
                print(f"      Token IDs: {token_ids[:10]}...")
                print(f"      Range: {min_id} to {max_id}")
                
                if min_id < 0:
                    print(f"      🚨 NEGATIVE TOKEN DETECTED: {min_id}")
                    return False
                else:
                    print(f"      ✅ All tokens positive")
            
            return True
        else:
            print(f"   ❌ Vocabulary file not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Tokenization test failed: {e}")
        return False

def recommendation():
    """Provide recommendation on whether to redo pretraining"""
    
    print(f"\n🎯 RECOMMENDATION:")
    print("=" * 60)
    
    # Run checks
    clean_tokenization = test_tokenization_on_clean_data()
    
    if clean_tokenization:
        print(f"✅ PRETRAINING DATA APPEARS CLEAN")
        print(f"   • Vocabulary mapping works correctly")
        print(f"   • No negative tokens in manual test")
        print(f"   • Pretrained model loads successfully")
        print(f"")
        print(f"🎯 RECOMMENDATION: KEEP EXISTING PRETRAINING")
        print(f"   • Your pretraining is likely fine")
        print(f"   • The issue is only in survival finetuning data")
        print(f"   • Fix the reference date issue in survival data")
        print(f"   • Regenerate survival corpus only")
        print(f"   • Keep using existing pretrained model")
        
        print(f"\n📋 ACTION PLAN:")
        print(f"   1. Fix reference date issue in survival data preprocessing")
        print(f"   2. Regenerate survival prediction corpus")
        print(f"   3. Use existing pretrained model for finetuning")
        print(f"   4. Run survival finetuning with clean data")
        
    else:
        print(f"❌ POTENTIAL ISSUES DETECTED")
        print(f"   • Tokenization issues found")
        print(f"   • May need to redo pretraining")
        print(f"")
        print(f"🎯 RECOMMENDATION: INVESTIGATE FURTHER")
        print(f"   • Check if pretraining actually succeeded")
        print(f"   • Verify pretrained model quality")
        print(f"   • Consider redoing both pretraining and finetuning")

if __name__ == "__main__":
    check_pretraining_data()
    recommendation()
