#!/usr/bin/env python3
"""
STEP 1: Check vocabulary size mismatch between pretraining and finetuning
This might be the root cause of your "index out of range" errors
"""

import torch
import pandas as pd
from pathlib import Path

def check_vocabulary_sizes():
    """Check vocabulary sizes in pretrained model vs current vocab"""
    
    print("🔍 STEP 1: CHECKING VOCABULARY SIZES")
    print("=" * 60)
    
    # Check pretrained model vocabulary
    print("1. Checking pretrained model...")
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    if Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        pretrained_vocab_size = checkpoint['hparams']['vocab_size']
        print(f"   ✅ Pretrained model vocab size: {pretrained_vocab_size:,}")
    else:
        print(f"   ❌ Pretrained model not found: {pretrained_path}")
        return False
    
    # Check current vocabulary file
    print("\n2. Checking current vocabulary file...")
    vocab_path = Path("data/processed/vocab/startup_vocab/result.tsv")
    
    if vocab_path.exists():
        vocab_df = pd.read_csv(vocab_path, sep='\t')
        current_vocab_size = len(vocab_df)
        print(f"   ✅ Current vocab file size: {current_vocab_size:,}")
        
        # Show some sample tokens
        print(f"   📊 Sample tokens:")
        print(f"      First 5: {list(vocab_df['TOKEN'].head())}")
        print(f"      Last 5: {list(vocab_df['TOKEN'].tail())}")
        
        # Check for special tokens
        special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        for token in special_tokens:
            if token in vocab_df['TOKEN'].values:
                token_id = vocab_df[vocab_df['TOKEN'] == token]['ID'].iloc[0]
                print(f"      {token}: ID {token_id}")
            else:
                print(f"      {token}: MISSING!")
    else:
        print(f"   ❌ Vocabulary file not found: {vocab_path}")
        return False
    
    # Check for mismatch
    print(f"\n3. Comparison:")
    print(f"   Pretrained vocab size: {pretrained_vocab_size:,}")
    print(f"   Current vocab size: {current_vocab_size:,}")
    
    if pretrained_vocab_size != current_vocab_size:
        print(f"   🚨 VOCABULARY SIZE MISMATCH!")
        print(f"   🚨 Difference: {abs(pretrained_vocab_size - current_vocab_size):,} tokens")
        print(f"   🚨 This WILL cause 'index out of range' errors!")
        
        print(f"\n💡 SOLUTIONS:")
        if current_vocab_size > pretrained_vocab_size:
            print(f"   1. Use pretrained vocab size ({pretrained_vocab_size:,}) in finetuning")
            print(f"   2. Retrain pretrained model with current vocab ({current_vocab_size:,})")
        else:
            print(f"   1. Use current vocab size ({current_vocab_size:,}) - safer option")
            print(f"   2. Extend current vocab to match pretrained ({pretrained_vocab_size:,})")
        
        return False
    else:
        print(f"   ✅ VOCABULARY SIZES MATCH!")
        print(f"   ✅ No vocabulary mismatch issues")
        return True

def check_token_id_ranges():
    """Check if token IDs in data are within vocabulary range"""
    
    print(f"\n4. Checking token ID ranges in data...")
    
    # Load vocabulary
    vocab_path = Path("data/processed/vocab/startup_vocab/result.tsv")
    vocab_df = pd.read_csv(vocab_path, sep='\t')
    max_valid_id = vocab_df['ID'].max()
    min_valid_id = vocab_df['ID'].min()
    
    print(f"   Valid token ID range: {min_valid_id} to {max_valid_id}")
    
    # Check a sample of corpus data
    corpus_paths = [
        "data/processed/corpus/startup_corpus/sentences/train/sentences.parquet",
        "data/processed/corpus/startup_corpus/sentences/val/sentences.parquet"
    ]
    
    for corpus_path in corpus_paths:
        if Path(corpus_path).exists():
            print(f"   Checking {corpus_path}...")
            
            # Load a sample
            df = pd.read_parquet(corpus_path)
            
            if 'SEQUENCE' in df.columns:
                # Check token IDs in sequences
                sample_sequences = df['SEQUENCE'].head(100)
                all_token_ids = []
                
                for seq in sample_sequences:
                    if isinstance(seq, str):
                        # Parse sequence if it's a string
                        try:
                            token_ids = [int(x) for x in seq.split()]
                            all_token_ids.extend(token_ids)
                        except:
                            continue
                    elif isinstance(seq, list):
                        all_token_ids.extend(seq)
                
                if all_token_ids:
                    min_found = min(all_token_ids)
                    max_found = max(all_token_ids)
                    
                    print(f"      Found token ID range: {min_found} to {max_found}")
                    
                    if max_found > max_valid_id:
                        print(f"      🚨 INVALID TOKEN IDS FOUND!")
                        print(f"      🚨 Token {max_found} > max valid {max_valid_id}")
                        return False
                    elif min_found < min_valid_id:
                        print(f"      🚨 INVALID TOKEN IDS FOUND!")
                        print(f"      🚨 Token {min_found} < min valid {min_valid_id}")
                        return False
                    else:
                        print(f"      ✅ All token IDs are valid")
            else:
                print(f"      ⚠️ No SEQUENCE column found")
        else:
            print(f"   ⚠️ Corpus file not found: {corpus_path}")
    
    return True

if __name__ == "__main__":
    print("🔍 VOCABULARY SIZE AND TOKEN ID CHECK")
    print("This will identify vocabulary-related issues causing training failures")
    print()
    
    vocab_ok = check_vocabulary_sizes()
    token_ok = check_token_id_ranges()
    
    print(f"\n🎯 RESULTS:")
    if vocab_ok and token_ok:
        print(f"   ✅ Vocabulary is consistent")
        print(f"   ✅ Ready to proceed with finetuning")
    else:
        print(f"   ❌ Vocabulary issues found")
        print(f"   ❌ Must fix before finetuning")
    
    print(f"\n🔧 NEXT STEPS:")
    if not vocab_ok:
        print(f"   1. Fix vocabulary size mismatch")
        print(f"   2. Update model definition to use correct vocab size")
    if not token_ok:
        print(f"   1. Check data preprocessing")
        print(f"   2. Ensure token IDs are in valid range")
    
    if vocab_ok and token_ok:
        print(f"   1. Proceed to Step 2: Fix training configuration")
        print(f"   2. Use the correct vocabulary size in model")
