#!/usr/bin/env python3

"""
Step 3.1: Create Corpus - Hybrid Approach
Combines efficient chunked processing with life2vec framework compatibility
"""

import sys
import os
from pathlib import Path
import time
import logging
import pandas as pd
import dask.dataframe as dd
import numpy as np
import traceback
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import argparse

# Add src to path
project_root = Path("/data/kebl8110/life2vec_startup_project")
sys.path.insert(0, str(project_root / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corpus_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_parquet_file(parquet_file):
    """Process a single parquet file - categorical fields only"""
    
    try:
        # Read entire parquet file
        df = pd.read_parquet(parquet_file)
        if len(df) == 0:
            return []

        # Convert ALL events to sentences (no filtering!)
        sentences = []
        
        # CATEGORICAL FIELDS ONLY - no text tokens
        priority_fields = [
            # Core event info
            'EVENT_TYPE',
            # Event-specific fields
            'INV_investment_type',
            'ACQ_acquisition_type',
            'EVENT_appearance_type',
            'PEOPLE_job_type',
            # Location fields
            'EVENT_city',
            'INV_investor_city',
            'ACQ_target_city',
            'PEOPLE_city',
            'EVENT_country_code',
            'INV_investor_country_code',
            'ACQ_target_country_code',
            # Financial info (binned)
            'INV_raised_amount_usd_binned',
            'ACQ_price_usd_binned',
            'INV_post_money_valuation_usd_binned',
            # Education info
            'EDU_degree_type',
            'EDU_institution',
            # Time info (properly binned!)
            'DAYS_SINCE_FOUNDING_BINNED'
        ]

        # Filter to available fields
        available_fields = [f for f in priority_fields if f in df.columns]

        # Process ALL events in this file
        for startup_id, startup_events in df.groupby(level=0):
            for _, event in startup_events.iterrows():
                tokens = []
                for field in available_fields:
                    if pd.notna(event.get(field)):
                        value = str(event[field])
                        if value and value not in ['nan', 'NaN', '', 'UNK']:
                            # All fields are categorical - ensure proper prefix
                            clean_token = ensure_proper_prefix(field, value)
                            if clean_token and len(clean_token) > 3:
                                tokens.append(clean_token)

                # Create sentence from tokens
                if len(tokens) >= 1:
                    # Use reasonable sentence length for categorical data
                    sentence = ' '.join(tokens[:15])
                    sentences.append({
                        'STARTUP_ID': startup_id,
                        'RECORD_DATE': event.get('RECORD_DATE', pd.Timestamp.now()),
                        'SENTENCE': sentence
                    })

        return sentences

    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")
        return []

def ensure_proper_prefix(field_name, value):
    """Ensure proper prefixes for categorical fields only"""
    
    # Skip completely invalid values
    if not value or value in ['nan', 'NaN', '', 'UNK']:
        return None
    
    # If value already has correct prefix, return as-is
    if field_name == 'EVENT_TYPE' and value.startswith('TYPE_'):
        return value
    elif field_name.startswith('INV_') and value.startswith('INV_'):
        return value
    elif field_name.startswith('ACQ_') and value.startswith('ACQ_'):
        return value
    elif field_name.startswith('EVENT_') and value.startswith('EVT_'):
        return value
    elif field_name.startswith('PEOPLE_') and value.startswith('PPL_'):
        return value
    elif field_name.startswith('EDU_') and value.startswith('EDU_'):
        return value
    elif field_name.startswith('IPO_') and value.startswith('IPO_'):
        return value
    elif field_name == 'DAYS_SINCE_FOUNDING_BINNED' and value.startswith('DAYS_'):
        return value
    
    # Add appropriate prefix based on field name
    if field_name == 'EVENT_TYPE':
        return f'TYPE_{value}'
    elif field_name.startswith('INV_'):
        return f'INV_{value}'
    elif field_name.startswith('ACQ_'):
        return f'ACQ_{value}'
    elif field_name.startswith('EVENT_'):
        return f'EVT_{value}'
    elif field_name.startswith('PEOPLE_'):
        return f'PPL_{value}'
    elif field_name.startswith('EDU_'):
        return f'EDU_{value}'
    elif field_name.startswith('IPO_'):
        return f'IPO_{value}'
    elif field_name == 'DAYS_SINCE_FOUNDING_BINNED':
        return f'DAYS_{value}'
    else:
        # For unknown fields, use first 3 characters
        return f'{field_name[:3].upper()}_{value}'

def analyze_token_frequencies(all_sentences, min_frequency=5):
    """Analyze token frequencies and filter rare tokens"""
    
    logger.info(f"🔍 Analyzing token frequencies (min_frequency={min_frequency})...")
    
    # Count all tokens
    token_counter = Counter()
    total_tokens = 0
    
    for sentence_data in all_sentences:
        tokens = sentence_data['SENTENCE'].split()
        token_counter.update(tokens)
        total_tokens += len(tokens)
    
    # Get frequency stats
    unique_tokens = len(token_counter)
    frequent_tokens = {token: count for token, count in token_counter.items() if count >= min_frequency}
    rare_tokens = {token: count for token, count in token_counter.items() if count < min_frequency}
    
    logger.info(f"📊 Token Frequency Analysis:")
    logger.info(f"   • Total tokens: {total_tokens:,}")
    logger.info(f"   • Unique tokens: {unique_tokens:,}")
    logger.info(f"   • Frequent tokens (≥{min_frequency}): {len(frequent_tokens):,}")
    logger.info(f"   • Rare tokens (<{min_frequency}): {len(rare_tokens):,}")
    logger.info(f"   • Vocabulary reduction: {len(rare_tokens)/unique_tokens*100:.1f}%")
    
    # Show top frequent tokens (should be mostly prefixed categorical values)
    logger.info("🔝 Top 20 most frequent tokens:")
    top_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    for token, count in top_tokens:
        logger.info(f"   • '{token}': {count:,} occurrences")
    
    return frequent_tokens, rare_tokens

def filter_sentences_by_frequency(all_sentences, frequent_tokens):
    """Filter sentences to only include frequent tokens"""
    
    logger.info("🔧 Filtering sentences to remove rare tokens...")
    
    filtered_sentences = []
    filtered_token_count = 0
    total_token_count = 0
    
    for sentence_data in all_sentences:
        original_tokens = sentence_data['SENTENCE'].split()
        # Keep only frequent tokens
        filtered_tokens = [token for token in original_tokens if token in frequent_tokens]
        total_token_count += len(original_tokens)
        filtered_token_count += len(filtered_tokens)
        
        # Only keep sentences with at least 1 frequent token
        if len(filtered_tokens) >= 1:
            sentence_data['SENTENCE'] = ' '.join(filtered_tokens)
            filtered_sentences.append(sentence_data)
    
    logger.info(f"📊 Sentence Filtering Results:")
    logger.info(f"   • Original sentences: {len(all_sentences):,}")
    logger.info(f"   • Filtered sentences: {len(filtered_sentences):,}")
    logger.info(f"   • Sentences kept: {len(filtered_sentences)/len(all_sentences)*100:.1f}%")
    logger.info(f"   • Tokens before: {total_token_count:,}")
    logger.info(f"   • Tokens after: {filtered_token_count:,}")
    logger.info(f"   • Token reduction: {(total_token_count-filtered_token_count)/total_token_count*100:.1f}%")
    
    return filtered_sentences

def create_life2vec_splits(sentences_df, corpus_name):
    """Create train/val/test splits compatible with life2vec framework"""
    
    logger.info("🔀 Creating train/val/test splits...")
    
    try:
        # Load population splits
        from dataloaders.populations.startups import StartupPopulation
        population = StartupPopulation()
        splits = population.data_split()
        
        logger.info(f"📊 Split sizes:")
        logger.info(f"   • Train: {len(splits.train):,}")
        logger.info(f"   • Val: {len(splits.val):,}")
        logger.info(f"   • Test: {len(splits.test):,}")
        
        # Create output directories
        output_dir = Path("data") / "processed" / "corpus" / corpus_name / "sentences"
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split and save data
        splits_info = {}
        
        for split_name, startup_ids in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
            # Filter sentences for this split
            split_sentences = sentences_df[sentences_df.index.isin(startup_ids)]
            
            if len(split_sentences) > 0:
                # Save to parquet
                output_file = output_dir / split_name / "sentences.parquet"
                split_sentences.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy'
                )
                
                splits_info[split_name] = len(split_sentences)
                logger.info(f"   ✅ {split_name}: {len(split_sentences):,} sentences saved")
            else:
                splits_info[split_name] = 0
                logger.warning(f"   ⚠️  {split_name}: No sentences found")
        
        return splits_info
        
    except Exception as e:
        logger.error(f"❌ Error creating splits: {e}")
        # Fallback: save everything as train
        train_dir = Path("data") / "processed" / "corpus" / corpus_name / "sentences" / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        sentences_df.to_parquet(
            train_dir / "sentences.parquet",
            engine='pyarrow',
            compression='snappy'
        )
        
        return {"train": len(sentences_df), "val": 0, "test": 0}

def create_startup_corpus(corpus_name="startup_corpus", sample_size=None, max_workers=None, min_token_frequency=5):
    """Create startup corpus with life2vec compatibility"""
    
    if sample_size:
        logger.info(f"🌎 CREATING STARTUP CORPUS - SAMPLE ({sample_size:,} startups)")
        corpus_name = f"{corpus_name}_sample"
    else:
        logger.info("🌎 CREATING STARTUP CORPUS - ALL STARTUPS")
    
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # Check tokenized data
        tokenized_path = Path("data") / "processed" / "sources" / "startup_events" / "tokenized"
        if not tokenized_path.exists():
            logger.error(f"❌ Tokenized data not found: {tokenized_path}")
            return {'success': False, 'error': 'Tokenized data not found'}
        
        # Get all parquet files
        parquet_files = list(tokenized_path.glob("*.parquet"))
        logger.info(f"📁 Found {len(parquet_files)} parquet files")
        
        if sample_size:
            # For testing: only process first few files
            num_files_for_sample = max(1, min(50, len(parquet_files) // 20))
            parquet_files = parquet_files[:num_files_for_sample]
            logger.info(f"🧪 Sample mode: processing {len(parquet_files)} files")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 16)
        logger.info(f"🔄 Using {max_workers} parallel workers")
        
        # Process ALL files in parallel
        logger.info("🚀 Starting parallel processing...")
        process_start = time.time()
        all_sentences = []
        processed_files = 0
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(process_single_parquet_file, pf) for pf in parquet_files]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    sentences = future.result()
                    all_sentences.extend(sentences)
                    processed_files += 1
                    
                    # Progress update every 50 files
                    if processed_files % 50 == 0:
                        elapsed = time.time() - process_start
                        rate = processed_files / elapsed
                        eta = (len(parquet_files) - processed_files) / rate
                        logger.info(f"📦 Progress: {processed_files}/{len(parquet_files)} files "
                                  f"({processed_files/len(parquet_files)*100:.1f}%) "
                                  f"| Sentences: {len(all_sentences):,} "
                                  f"| Rate: {rate:.1f} files/sec "
                                  f"| ETA: {eta/60:.1f}m")
                except Exception as e:
                    logger.warning(f"Failed to process file: {e}")
        
        process_time = time.time() - process_start
        logger.info(f"✅ Parallel processing completed in {process_time:.1f}s ({process_time/60:.1f}m)")
        logger.info(f"📊 Total sentences (before filtering): {len(all_sentences):,}")
        
        # Apply frequency filtering
        if len(all_sentences) > 0:
            logger.info("🔍 APPLYING FREQUENCY FILTERING...")
            freq_start = time.time()
            
            # Analyze token frequencies
            frequent_tokens, rare_tokens = analyze_token_frequencies(all_sentences, min_token_frequency)
            
            # Filter sentences
            filtered_sentences = filter_sentences_by_frequency(all_sentences, frequent_tokens)
            freq_time = time.time() - freq_start
            logger.info(f"✅ Frequency filtering completed in {freq_time:.1f}s")
            
            # Use filtered sentences for corpus
            all_sentences = filtered_sentences
        
        # Create output structure efficiently
        if len(all_sentences) > 0:
            logger.info("💾 Creating life2vec-compatible output structure...")
            save_start = time.time()
            
            # Convert to DataFrame efficiently
            logger.info("📊 Converting to DataFrame...")
            sentences_df = pd.DataFrame(all_sentences)
            sentences_df = sentences_df.set_index('STARTUP_ID')
            
            # Add life2vec required columns
            logger.info("🔧 Adding life2vec metadata...")
            sentences_df['AGE'] = 1  # Default age
            
            # Add threshold flag
            threshold_date = pd.Timestamp('2025-01-01')
            sentences_df['AFTER_THRESHOLD'] = pd.to_datetime(sentences_df['RECORD_DATE']) >= threshold_date
            
            # Convert dates to days since reference (life2vec format)
            reference_date = pd.Timestamp('1980-01-01')  # Earlier for startups
            sentences_df['RECORD_DATE'] = (pd.to_datetime(sentences_df['RECORD_DATE']) - reference_date).dt.days
            
            # Create train/val/test splits
            splits_info = create_life2vec_splits(sentences_df, corpus_name)
            
            save_time = time.time() - save_start
            logger.info(f"✅ Saved corpus in life2vec format in {save_time:.1f}s")
            
            # Show final stats
            logger.info("📊 Final Dataset Statistics:")
            logger.info(f"   • Total sentences: {len(sentences_df):,}")
            logger.info(f"   • Unique startups: {sentences_df.index.nunique():,}")
            logger.info(f"   • Avg sentences per startup: {len(sentences_df)/sentences_df.index.nunique():.1f}")
            logger.info(f"   • Vocabulary size: {len(frequent_tokens):,} tokens")
            logger.info(f"   • Train sentences: {splits_info.get('train', 0):,}")
            logger.info(f"   • Val sentences: {splits_info.get('val', 0):,}")
            logger.info(f"   • Test sentences: {splits_info.get('test', 0):,}")
            
            logger.info("�� Sample sentences:")
            for i, sent in enumerate(sentences_df['SENTENCE'].head(5)):
                logger.info(f"  {i+1}. {sent}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 STARTUP CORPUS CREATION COMPLETE!")
        logger.info(f"⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"📋 Final sentences: {len(all_sentences):,}")
        logger.info(f"📖 Final vocabulary: {len(frequent_tokens):,} tokens")
        
        return {
            'success': True,
            'corpus_name': corpus_name,
            'sentences_count': len(all_sentences),
            'vocabulary_size': len(frequent_tokens),
            'splits_info': splits_info,
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating corpus: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Create startup corpus with life2vec compatibility')
    
    # Core options
    parser.add_argument('--corpus-name', type=str, default='startup_corpus',
                       help='Name for the corpus (default: startup_corpus)')
    parser.add_argument('--sample', action='store_true', 
                       help='Process sample for testing')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers (default: 16)')
    parser.add_argument('--min-freq', type=int, default=5, 
                       help='Minimum token frequency (default: 5)')
    parser.add_argument('--run', action='store_true', required=True,
                       help='Actually run the processing')
    
    args = parser.parse_args()
    
    sample_size = 10000 if args.sample else None
    result = create_startup_corpus(
        corpus_name=args.corpus_name,
        sample_size=sample_size, 
        max_workers=args.workers, 
        min_token_frequency=args.min_freq
    )
    
    if result['success']:
        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"📊 Corpus: {result['corpus_name']}")
        logger.info(f"📋 Sentences: {result['sentences_count']:,}")
        logger.info(f"📖 Vocabulary: {result['vocabulary_size']:,} tokens")
        logger.info(f"⏱️ Time: {result['elapsed_time']:.1f} seconds")
        logger.info("\n🚀 Ready for vocabulary and task creation!")
        return 0
    else:
        logger.error(f"\n❌ FAILED: {result.get('error')}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
