#!/usr/bin/env python3

"""
Step 3.1: Create Corpus - FIXED: Combines startup events + company static information
Following Life2Vec methodology by adding company attributes as background tokens
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

def load_company_static_data():
    """FIXED: Load and cache company static information using COMPANY_ID"""
    
    logger.info("📊 Loading company static data...")
    
    # Try to load from tokenized company data
    company_tokenized_path = Path("data") / "processed" / "sources" / "startup" / "tokenized"
    
    if not company_tokenized_path.exists():
        logger.error(f"❌ Company tokenized data not found: {company_tokenized_path}")
        logger.error("💡 Run step_2_define_tokensource.py --run first to tokenize company data")
        return None
    
    try:
        # Get all company parquet files
        company_files = list(company_tokenized_path.glob("*.parquet"))
        if not company_files:
            logger.error(f"❌ No company parquet files found in {company_tokenized_path}")
            return None
        
        logger.info(f"📁 Found {len(company_files)} company parquet files")
        
        # Read all company data
        company_dfs = []
        for file in company_files:
            try:
                df = pd.read_parquet(file)
                company_dfs.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read {file}: {e}")
        
        if not company_dfs:
            logger.error("❌ No company data could be loaded")
            return None
        
        # Combine all company data (keep index!)
        company_data = pd.concat(company_dfs, ignore_index=False)
        
        # FIXED: Use COMPANY_ID instead of STARTUP_ID
        if company_data.index.name != 'COMPANY_ID':
            if 'COMPANY_ID' in company_data.columns:
                company_data = company_data.set_index('COMPANY_ID')
            else:
                company_data.index.name = 'COMPANY_ID'
        
        logger.info(f"✅ Loaded company data for {len(company_data)} companies")
        
        # Show sample of available company attributes
        sample_company = company_data.iloc[0] if len(company_data) > 0 else None
        if sample_company is not None:
            company_attrs = [col for col in company_data.columns if col not in ['RECORD_DATE']]
            logger.info(f"📊 Available company attributes: {company_attrs}")
        
        return company_data
        
    except Exception as e:
        logger.error(f"❌ Error loading company data: {e}")
        return None

def get_company_background_tokens(company_id, company_data):
    """FIXED: Get background tokens for a specific company using COMPANY_ID"""
    
    if company_data is None or company_id not in company_data.index:
        # Return default background tokens if no company data
        return ["COUNTRY_Unknown", "CATEGORY_Unknown", "EMPLOYEE_Unknown", "INDUSTRY_OTHER", "MODEL_OTHER", "TECH_OTHER"]
    
    try:
        company_info = company_data.loc[company_id]
        background_tokens = []
        
        # Extract simple categorical fields
        simple_fields = ['COUNTRY', 'CATEGORY', 'EMPLOYEE']
        for field in simple_fields:
            if field in company_info and pd.notna(company_info[field]):
                value = str(company_info[field])
                # Value should already be prefixed (e.g., "COUNTRY_USA")
                if value and value not in ['nan', 'NaN', '', 'UNK']:
                    background_tokens.append(value)
        
        # CRITICAL FIX: Handle the DESCRIPTION field properly
        # The description is now categorized tokens, not raw text
        if 'DESCRIPTION' in company_info and pd.notna(company_info['DESCRIPTION']):
            description_value = str(company_info['DESCRIPTION'])
            if description_value and description_value not in ['nan', 'NaN', '', 'UNK']:
                # Description is now multiple categorical tokens separated by spaces
                # e.g., "INDUSTRY_FINTECH MODEL_B2B TECH_CLOUD"
                description_tokens = description_value.split()
                background_tokens.extend(description_tokens)
        
        # Add defaults for missing fields
        if not any(token.startswith('COUNTRY_') for token in background_tokens):
            background_tokens.append("COUNTRY_Unknown")
        if not any(token.startswith('CATEGORY_') for token in background_tokens):
            background_tokens.append("CATEGORY_Unknown")
        if not any(token.startswith('EMPLOYEE_') for token in background_tokens):
            background_tokens.append("EMPLOYEE_Unknown")
        if not any(token.startswith('INDUSTRY_') for token in background_tokens):
            background_tokens.append("INDUSTRY_OTHER")
        if not any(token.startswith('MODEL_') for token in background_tokens):
            background_tokens.append("MODEL_OTHER")
        if not any(token.startswith('TECH_') for token in background_tokens):
            background_tokens.append("TECH_OTHER")
        
        return background_tokens
        
    except Exception as e:
        logger.warning(f"⚠️ Error getting company background for {company_id}: {e}")
        return ["COUNTRY_Unknown", "CATEGORY_Unknown", "EMPLOYEE_Unknown", "INDUSTRY_OTHER", "MODEL_OTHER", "TECH_OTHER"]

def process_single_parquet_file_with_background(args):
    """Process a single parquet file - FIXED: Use COMPANY_ID"""
    
    parquet_file, company_data = args
    
    try:
        # Read entire parquet file
        df = pd.read_parquet(parquet_file)
        if len(df) == 0:
            return []

        # Convert ALL events to sentences with company background
        sentences = []
        
        # CATEGORICAL FIELDS ONLY - no text tokens (same as before)
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

        # FIXED: Process ALL events in this file using COMPANY_ID
        for company_id, company_events in df.groupby(level=0):
            
            # Get company background tokens for this company
            background_tokens = get_company_background_tokens(company_id, company_data)
            
            for _, event in company_events.iterrows():
                tokens = []
                
                # LIFE2VEC METHODOLOGY: Add company background tokens FIRST
                tokens.extend(background_tokens)
                
                # Then add event tokens (same logic as before)
                for field in available_fields:
                    if pd.notna(event.get(field)):
                        value = str(event[field])
                        if value and value not in ['nan', 'NaN', '', 'UNK']:
                            # All fields are categorical - ensure proper prefix
                            clean_token = ensure_proper_prefix(field, value)
                            if clean_token and len(clean_token) > 3:
                                tokens.append(clean_token)

                # Create sentence from tokens (background + event)
                if len(tokens) >= 5:  # At least 4 background + 1 event token
                    # Use reasonable sentence length for background + events
                    sentence = ' '.join(tokens[:25])  # Increased to accommodate all background tokens
                    sentences.append({
                        'COMPANY_ID': company_id,  # FIXED: Use COMPANY_ID
                        'RECORD_DATE': event.get('RECORD_DATE', pd.Timestamp.now()),
                        'SENTENCE': sentence
                    })

        return sentences

    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")
        return []

def ensure_proper_prefix(field_name, value):
    """Ensure proper prefixes for categorical fields only (same as before)"""
    
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
    """Analyze token frequencies and filter rare tokens (same as before)"""
    
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
    
    # Show top frequent tokens (should include company background tokens now)
    logger.info("🔝 Top 20 most frequent tokens:")
    top_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    for token, count in top_tokens:
        logger.info(f"   • '{token}': {count:,} occurrences")
    
    return frequent_tokens, rare_tokens

def filter_sentences_by_frequency(all_sentences, frequent_tokens):
    """Filter sentences to only include frequent tokens (same as before)"""
    
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
        
        # Only keep sentences with at least 4 frequent tokens (4 background + event tokens)
        if len(filtered_tokens) >= 4:
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
    """COMPLETELY FIXED: Create correct directory structure that vocabulary script expects"""
    
    logger.info("🔀 Creating train/val/test splits...")
    
    try:
        # COMPLETELY FIXED: Use the EXACT path structure that vocabulary script expects
        # data/processed/corpus/startup_corpus/sentences/train/sentences.parquet
        output_dir = Path("data") / "processed" / "corpus" / corpus_name / "sentences"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple train/val/test split by companies
        unique_companies = list(sentences_df['COMPANY_ID'].unique())
        total_companies = len(unique_companies)
        
        train_size = int(total_companies * 0.7)
        val_size = int(total_companies * 0.15)
        
        train_companies = unique_companies[:train_size]
        val_companies = unique_companies[train_size:train_size + val_size]
        test_companies = unique_companies[train_size + val_size:]
        
        logger.info(f"📊 Split sizes:")
        logger.info(f"   • Train: {len(train_companies):,}")
        logger.info(f"   • Val: {len(val_companies):,}")
        logger.info(f"   • Test: {len(test_companies):,}")
        
        # Split and save data
        splits_info = {}
        
        for split_name, company_ids in [("train", train_companies), ("val", val_companies), ("test", test_companies)]:
            # Filter sentences for this split
            split_sentences = sentences_df[sentences_df['COMPANY_ID'].isin(company_ids)]
            
            if len(split_sentences) > 0:
                # COMPLETELY FIXED: Create the EXACT directory structure vocabulary expects
                # Create split-specific directory (train/, val/, test/)
                split_dir = output_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Save to parquet in the split directory with EXACT filename vocabulary expects
                output_file = split_dir / "sentences.parquet"
                split_sentences.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy'
                )
                
                # Also save as .txt file for life2vec compatibility
                text_file = split_dir / "sentences.txt"
                with open(text_file, 'w') as f:
                    for sentence in split_sentences['SENTENCE']:
                        f.write(sentence + '\n')
                
                splits_info[split_name] = len(split_sentences)
                logger.info(f"   ✅ {split_name}: {len(split_sentences):,} sentences saved")
                logger.info(f"       📁 Parquet: {output_file}")
                logger.info(f"       📁 Text: {text_file}")
            else:
                splits_info[split_name] = 0
                logger.warning(f"   ⚠️  {split_name}: No sentences found")
        
        return splits_info
        
    except Exception as e:
        logger.error(f"❌ Error creating splits: {e}")
        return {"train": 0, "val": 0, "test": 0}

def create_startup_corpus_with_background(corpus_name="startup_corpus", sample_size=None, max_workers=None, min_token_frequency=5):
    """Create startup corpus with company background tokens - LIFE2VEC METHODOLOGY"""
    
    if sample_size:
        logger.info(f"🌎 CREATING COMBINED STARTUP CORPUS - SAMPLE ({sample_size:,} startups)")
        corpus_name = f"{corpus_name}_sample"
    else:
        logger.info("🌎 CREATING COMBINED STARTUP CORPUS - FULL (Events + Company Background)")
    
    logger.info("=" * 60)
    logger.info("🔄 LIFE2VEC METHODOLOGY: Adding company static info as background tokens")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # STEP 1: Load company static data
        logger.info("🏢 STEP 1: Loading company static data...")
        company_data = load_company_static_data()
        if company_data is None:
            logger.error("❌ Cannot proceed without company data")
            return {'success': False, 'error': 'Company data not found'}
        
        # STEP 2: Check events tokenized data
        logger.info("📦 STEP 2: Loading events data...")
        tokenized_path = Path("data") / "processed" / "sources" / "startup_events" / "tokenized"
        if not tokenized_path.exists():
            logger.error(f"❌ Events tokenized data not found: {tokenized_path}")
            return {'success': False, 'error': 'Events tokenized data not found'}
        
        # Get all parquet files
        parquet_files = list(tokenized_path.glob("*.parquet"))
        logger.info(f"📁 Found {len(parquet_files)} event parquet files")
        
        if sample_size:
            # For testing: only process first few files
            num_files_for_sample = max(1, min(50, len(parquet_files) // 20))
            parquet_files = parquet_files[:num_files_for_sample]
            logger.info(f"🧪 Sample mode: processing {len(parquet_files)} files")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 16)
        logger.info(f"🔄 Using {max_workers} parallel workers")
        
        # STEP 3: Process ALL files in parallel with company background
        logger.info("🚀 STEP 3: Starting parallel processing with company background...")
        process_start = time.time()
        all_sentences = []
        processed_files = 0
        
        # Prepare arguments for parallel processing (include company_data)
        processing_args = [(pf, company_data) for pf in parquet_files]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(process_single_parquet_file_with_background, args) for args in processing_args]
            
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
        
        # Show sample sentence with background tokens
        if len(all_sentences) > 0:
            sample_sentence = all_sentences[0]['SENTENCE']
            logger.info(f"📝 Sample sentence with background: {sample_sentence}")
        
        # Apply frequency filtering
        if len(all_sentences) > 0:
            logger.info("🔍 STEP 4: APPLYING FREQUENCY FILTERING...")
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
            logger.info("💾 STEP 5: Creating life2vec-compatible output structure...")
            save_start = time.time()
            
            # Convert to DataFrame efficiently
            logger.info("📊 Converting to DataFrame...")
            sentences_df = pd.DataFrame(all_sentences)
            
            # Add life2vec required columns
            logger.info("🔧 Adding life2vec metadata...")
            sentences_df['AGE'] = 1  # Default age
            
            # Add threshold flag
            threshold_date = pd.Timestamp('2025-01-01')
            sentences_df['AFTER_THRESHOLD'] = pd.to_datetime(sentences_df['RECORD_DATE']) >= threshold_date
            
            # Convert dates to days since reference (life2vec format)
            reference_date = pd.Timestamp('1941-01-01')  # Earlier for startups
            sentences_df['RECORD_DATE'] = (pd.to_datetime(sentences_df['RECORD_DATE']) - reference_date).dt.days
            
            # Create train/val/test splits
            splits_info = create_life2vec_splits(sentences_df, corpus_name)
            
            save_time = time.time() - save_start
            logger.info(f"✅ Saved corpus in life2vec format in {save_time:.1f}s")
            
            # Show final stats
            logger.info("📊 Final Dataset Statistics:")
            logger.info(f"   • Total sentences: {len(sentences_df):,}")
            logger.info(f"   • Unique startups: {sentences_df['COMPANY_ID'].nunique():,}")
            logger.info(f"   • Avg sentences per startup: {len(sentences_df)/sentences_df['COMPANY_ID'].nunique():.1f}")
            logger.info(f"   • Vocabulary size: {len(frequent_tokens):,} tokens")
            logger.info(f"   • Train sentences: {splits_info.get('train', 0):,}")
            logger.info(f"   • Val sentences: {splits_info.get('val', 0):,}")
            logger.info(f"   • Test sentences: {splits_info.get('test', 0):,}")
            
            logger.info("🔍 Sample sentences with background tokens:")
            for i, sent in enumerate(sentences_df['SENTENCE'].head(5)):
                logger.info(f"  {i+1}. {sent}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 COMBINED STARTUP CORPUS CREATION COMPLETE!")
        logger.info(f"⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"📋 Final sentences: {len(all_sentences):,}")
        logger.info(f"📖 Final vocabulary: {len(frequent_tokens):,} tokens")
        logger.info(f"🔄 LIFE2VEC METHODOLOGY: Company background tokens included in every sequence")
        
        return {
            'success': True,
            'corpus_name': corpus_name,
            'sentences_count': len(all_sentences),
            'vocabulary_size': len(frequent_tokens),
            'splits_info': splits_info,
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating combined corpus: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Create startup corpus with company background tokens')
    
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
    result = create_startup_corpus_with_background(
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
        logger.info("🔄 LIFE2VEC METHODOLOGY: Company attributes now included as background tokens")
        return 0
    else:
        logger.error(f"\n❌ FAILED: {result.get('error')}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)