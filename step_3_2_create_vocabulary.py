#!/usr/bin/env python3
# step_3_2_create_vocabulary.py - Fixed for life2vec corpus structure

import sys
import os
import argparse
import time
import shutil
from pathlib import Path
from collections import Counter
import pandas as pd
import dask.dataframe as dd
import json
import logging

# Add src to path
project_root = Path("/data/kebl8110/life2vec_startup_project")
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def create_startup_vocabulary(corpus_name="startup_corpus", min_token_count=2):
    """
    Create vocabulary from startup corpus sentences (life2vec format)
    
    Args:
        corpus_name: Name of the corpus to create vocabulary from
        min_token_count: Minimum occurrences for a token to be included
    """
    
    log.info("🚀 CREATING STARTUP VOCABULARY")
    log.info("=" * 60)
    
    # Setup paths - FIXED: Use corpus sentences, not raw tokenized data
    corpus_path = project_root / "data/processed/corpus" / corpus_name / "sentences" / "train"
    vocab_dir = project_root / "data/processed/vocab" / "startup_vocab"
    
    # Clean and create vocab directory
    if vocab_dir.exists():
        log.info(f"🗑️  Removing existing vocabulary: {vocab_dir}")
        shutil.rmtree(vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    # Check corpus exists
    sentences_file = corpus_path / "sentences.parquet"
    if not sentences_file.exists():
        log.error(f"❌ Corpus sentences not found: {sentences_file}")
        log.error(f"💡 Available corpus:")
        corpus_root = project_root / "data/processed/corpus"
        if corpus_root.exists():
            for corpus_dir in corpus_root.iterdir():
                if corpus_dir.is_dir():
                    log.error(f"   📁 {corpus_dir.name}")
        return None
    
    log.info(f"📁 Loading corpus sentences from: {sentences_file}")
    
    try:
        # Load corpus sentences
        start_time = time.time()
        df = pd.read_parquet(sentences_file)
        log.info(f"✅ Loaded corpus: {len(df):,} sentences")
        
        # Count tokens from sentences
        log.info("🔄 Counting tokens from sentences...")
        token_counts = Counter()
        
        total_sentences = len(df)
        for i, sentence in enumerate(df['SENTENCE']):
            if pd.notna(sentence) and str(sentence).strip():
                tokens = str(sentence).split()
                token_counts.update(tokens)
            
            # Progress update
            if (i + 1) % 50000 == 0:
                progress = (i + 1) / total_sentences * 100
                log.info(f"  📊 Progress: {i+1:,}/{total_sentences:,} ({progress:.1f}%)")
        
        count_time = time.time() - start_time
        log.info(f"✅ Token counting completed in {count_time:.1f}s")
        log.info(f"📈 Found {len(token_counts):,} unique tokens")
        
        # Show top tokens
        log.info("🔝 Top 10 most frequent tokens:")
        for token, count in token_counts.most_common(10):
            log.info(f"   • '{token}': {count:,} occurrences")
        
        # Filter tokens by minimum count
        log.info(f"🔍 Filtering tokens with min_count >= {min_token_count}")
        filtered_tokens = {token: count for token, count in token_counts.items() 
                          if count >= min_token_count}
        removed_count = len(token_counts) - len(filtered_tokens)
        log.info(f"📉 Filtered to {len(filtered_tokens):,} tokens (removed {removed_count:,})")
        
        # Build vocabulary structure (life2vec format)
        log.info("🔧 Building vocabulary structure...")
        vocab_parts = []
        
        # 1. General tokens (life2vec standard)
        general_tokens = [
            '[PAD]', '[CLS]', '[SEP]', '[MASK]', 
            '[PLCH0]', '[PLCH1]', '[PLCH2]', '[PLCH3]', '[PLCH4]', 
            '[UNK]'
        ]
        vocab_parts.append(pd.DataFrame({
            "TOKEN": general_tokens,
            "CATEGORY": "GENERAL"
        }))
        log.info(f"  ✅ Added {len(general_tokens)} general tokens")
        
        # 2. Background tokens (for person demographics - minimal for startups)
        background_tokens = ["FEMALE", "MALE"]  # Basic life2vec requirement
        vocab_parts.append(pd.DataFrame({
            "TOKEN": background_tokens,
            "CATEGORY": "BACKGROUND"
        }))
        log.info(f"  ✅ Added {len(background_tokens)} background tokens")
        
        # 3. Month tokens
        month_tokens = [f"MONTH_{i}" for i in range(1, 13)]
        vocab_parts.append(pd.DataFrame({
            "TOKEN": month_tokens,
            "CATEGORY": "MONTH"
        }))
        log.info(f"  ✅ Added {len(month_tokens)} month tokens")
        
        # 4. Year tokens (startup era: 1980-2025)
        year_tokens = [f"YEAR_{i}" for i in range(1980, 2026)]
        vocab_parts.append(pd.DataFrame({
            "TOKEN": year_tokens,
            "CATEGORY": "YEAR"
        }))
        log.info(f"  ✅ Added {len(year_tokens)} year tokens (1980-2025)")
        
        # 5. Field-specific tokens (categorized by prefix)
        log.info("📂 Categorizing startup tokens by prefix...")
        
        # Group tokens by prefix
        prefix_groups = {}
        for token in filtered_tokens.keys():
            # Find prefix (everything before first underscore)
            if '_' in token:
                prefix = token.split('_')[0]
            else:
                prefix = 'OTHER'
            
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(token)
        
        # Add each prefix group as a category
        total_field_tokens = 0
        for prefix, tokens in prefix_groups.items():
            if tokens:
                # Sort tokens for consistency
                tokens.sort()
                vocab_parts.append(pd.DataFrame({
                    "TOKEN": tokens,
                    "CATEGORY": prefix
                }))
                total_field_tokens += len(tokens)
                log.info(f"  📂 {prefix}: {len(tokens):,} tokens")
        
        log.info(f"  ✅ Added {total_field_tokens:,} field-specific tokens")
        
        # 6. Combine all parts
        log.info("🔗 Combining vocabulary parts...")
        vocab_df = pd.concat(vocab_parts, ignore_index=True)
        vocab_df = vocab_df.reset_index().rename(columns={'index': 'ID'})
        
        # Final statistics
        log.info(f"📊 FINAL VOCABULARY STATISTICS:")
        log.info(f"  📋 Total tokens: {len(vocab_df):,}")
        log.info(f"  📂 Categories: {vocab_df['CATEGORY'].nunique()}")
        
        # Show category breakdown
        category_counts = vocab_df['CATEGORY'].value_counts()
        log.info(f"📈 Category breakdown:")
        for cat, count in category_counts.items():
            log.info(f"  {cat}: {count:,} tokens")
        
        # 7. Save vocabulary (life2vec TSV format)
        log.info("💾 Saving vocabulary...")
        
        # Save main vocabulary file (life2vec format: ID as index)
        vocab_file = vocab_dir / "result.tsv"
        vocab_df.set_index('ID').to_csv(vocab_file, sep="\t", index=True)
        log.info(f"  ✅ Saved vocabulary: {vocab_file}")
        
        # Save arguments file (life2vec format)
        arguments = {
            "corpus_name": corpus_name,
            "name": "startup_vocab",
            "general_tokens": general_tokens,
            "background_tokens": background_tokens,
            "year_range": [1980, 2025],
            "min_token_count": min_token_count,
            "min_token_count_field": {}
        }
        
        arguments_file = vocab_dir / "_arguments.json"
        with open(arguments_file, 'w') as f:
            json.dump(arguments, f, indent=2)
        log.info(f"  ✅ Saved arguments: {arguments_file}")
        
        # Validation
        log.info("🔍 Validating vocabulary...")
        
        # Check for required tokens
        required_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
        missing_tokens = [token for token in required_tokens if token not in vocab_df['TOKEN'].values]
        if missing_tokens:
            log.warning(f"⚠️  Missing required tokens: {missing_tokens}")
        else:
            log.info("✅ All required tokens present")
        
        # Check for startup-specific tokens
        startup_prefixes = ['EVT', 'INV', 'ACQ', 'IPO', 'PPL', 'EDU', 'TYPE']
        found_prefixes = []
        for prefix in startup_prefixes:
            prefix_tokens = vocab_df[vocab_df['TOKEN'].str.startswith(prefix)]
            if len(prefix_tokens) > 0:
                found_prefixes.append(f"{prefix} ({len(prefix_tokens)})")
        
        if found_prefixes:
            log.info(f"🎯 Startup prefixes found: {', '.join(found_prefixes)}")
        else:
            log.warning("⚠️  No startup-specific prefixes found")
        
        total_time = time.time() - start_time
        log.info(f"🎉 VOCABULARY CREATION COMPLETE!")
        log.info(f"📊 Created {len(vocab_df):,} tokens in {total_time:.1f}s")
        log.info(f"📁 Saved to: {vocab_dir}")
        
        return vocab_df
        
    except Exception as e:
        log.error(f"❌ Error creating vocabulary: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_vocabulary_loading(vocab_name="startup_vocab"):
    """Test that the created vocabulary can be loaded correctly"""
    
    log.info("\n🧪 TESTING VOCABULARY LOADING")
    log.info("=" * 40)
    
    try:
        # Test direct file loading
        vocab_dir = project_root / "data/processed/vocab" / vocab_name
        vocab_file = vocab_dir / "result.tsv"
        
        if not vocab_file.exists():
            log.error(f"❌ Vocabulary file not found: {vocab_file}")
            return False
        
        # Load vocabulary
        vocab_df = pd.read_csv(vocab_file, sep="\t", index_col=0)
        log.info(f"✅ Vocabulary loaded: {len(vocab_df)} tokens")
        
        # Test token mappings
        token2index = dict(zip(vocab_df['TOKEN'], vocab_df.index))
        index2token = dict(zip(vocab_df.index, vocab_df['TOKEN']))
        
        log.info(f"📊 Token mappings created: {len(token2index)} entries")
        
        # Test some basic lookups
        test_tokens = ['[PAD]', '[CLS]', '[UNK]', '[MASK]']
        log.info("🔍 Testing token lookups:")
        for token in test_tokens:
            token_id = token2index.get(token, 'NOT_FOUND')
            log.info(f"  {token}: ID {token_id}")
        
        # Test some startup tokens
        startup_tokens = [token for token in vocab_df['TOKEN'] if any(token.startswith(p) for p in ['EVT_', 'INV_', 'PPL_'])]
        if startup_tokens:
            log.info("🎯 Sample startup tokens:")
            for token in startup_tokens[:5]:
                token_id = token2index[token]
                log.info(f"  {token}: ID {token_id}")
        
        log.info("✅ Vocabulary test PASSED")
        return True
        
    except Exception as e:
        log.error(f"❌ Vocabulary loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='Create startup vocabulary from corpus sentences')
    parser.add_argument('--corpus', type=str, default='startup_corpus', 
                       help='Name of the corpus to create vocabulary from')
    parser.add_argument('--min-count', type=int, default=2, 
                       help='Minimum token count threshold')
    parser.add_argument('--test', action='store_true', 
                       help='Test vocabulary loading after creation')
    parser.add_argument('--run', action='store_true', 
                       help='Actually run the vocabulary creation')
    
    args = parser.parse_args()
    
    if not args.run:
        log.info("🔧 STARTUP VOCABULARY CREATOR")
        log.info("=" * 40)
        log.info("Creates vocabulary from corpus sentences (life2vec format)")
        log.info("")
        log.info("Options:")
        log.info(f"  --run                    # Actually create the vocabulary")
        log.info(f"  --corpus {args.corpus}         # Use corpus '{args.corpus}'")
        log.info(f"  --min-count {args.min_count}             # Minimum {args.min_count} occurrences per token") 
        log.info(f"  --test                   # Test vocabulary loading")
        log.info("")
        log.info("Available corpus:")
        corpus_root = project_root / "data/processed/corpus"
        if corpus_root.exists():
            for corpus_dir in corpus_root.iterdir():
                if corpus_dir.is_dir():
                    log.info(f"  📁 {corpus_dir.name}")
        log.info("")
        log.info("🚀 Example: python step_3_2_create_vocabulary.py --run --corpus startup_corpus --test")
        return
    
    start_time = time.time()
    
    # Create vocabulary
    vocab_df = create_startup_vocabulary(corpus_name=args.corpus, min_token_count=args.min_count)
    
    if vocab_df is not None:
        total_time = time.time() - start_time
        log.info(f"⏱️  Total processing time: {total_time:.1f} seconds")
        
        # Test loading if requested
        if args.test:
            test_vocabulary_loading()
        
        log.info("\n🎉 SUCCESS!")
        log.info("✅ Vocabulary ready for task creation!")
        log.info("🚀 Next step: Create tasks and datamodule")
    else:
        log.error("❌ Vocabulary creation failed")

if __name__ == "__main__":
    main()
