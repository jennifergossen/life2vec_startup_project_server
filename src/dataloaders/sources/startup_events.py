from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import gc
import psutil
import warnings
import pickle
import re
import pandas as pd
import dask.dataframe as dd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned

# OPTIMIZED DASK CONFIGURATION
import dask

dask.config.set({
    'array.chunk-size': '256MB',
    'dataframe.query-planning': False,
    'dataframe.shuffle.method': 'tasks',
    'distributed.worker.memory.target': 0.75,
    'distributed.worker.memory.spill': 0.85,
    'distributed.worker.memory.pause': 0.90,
    'dataframe.backend': 'pandas',
    'array.slicing.split_large_chunks': True,
    # ADD THESE FOR MORE PARALLELISM:
    'num_workers': 8,
    'threads_per_worker': 4,
    'memory_limit': '16GB',
})

def print_memory_usage(step=""):
    """Helper function to monitor memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[{step}] Memory usage: {memory_mb:.1f} MB")

def bin_days_since_founding(days_series, strategy='quantile', n_bins=20):
    """
    Bin DAYS_SINCE_FOUNDING into meaningful categories
    
    Args:
        days_series: pandas Series with days since founding
        strategy: 'quantile' (recommended for consistency)
        n_bins: number of bins for quantile strategy
    
    Returns:
        pandas Series with binned values
    """
    if strategy == 'quantile':
        # Quantile-based binning - consistent with other continuous variables
        binned = pd.qcut(
            days_series, 
            q=n_bins, 
            labels=[f"DAYS_Q{i+1}" for i in range(n_bins)], 
            duplicates='drop'
        )
    else:
        raise ValueError("Only 'quantile' strategy is supported")
    
    # Handle missing values
    binned = binned.cat.add_categories(['DAYS_UNK']).fillna('DAYS_UNK')
    
    return binned.astype(str)

class TextTokenizer:
    """Helper class for processing ONLY the 3 main text fields"""
    
    def __init__(self):
        self.stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'who', 'what', 'where', 'when', 'why', 'how',
            'can', 'said', 'get', 'go', 'know', 'take', 'see', 'come', 'think', 'look',
            'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try'
        }

        # Event categorization rules
        self.event_categories = {
            'CONFERENCE': ['conference', 'summit', 'convention', 'expo', 'forum', 'symposium'],
            'TRAINING': ['training', 'course', 'workshop', 'bootcamp', 'certification', 'masterclass', 'class'],
            'NETWORKING': ['meetup', 'networking', 'mixer', 'social', 'gathering', 'community'],
            'PITCH': ['pitch', 'demo', 'presentation', 'showcase', 'demoday', 'demo day'],
            'COMPETITION': ['competition', 'contest', 'challenge', 'award', 'prize'],
            'HACKATHON': ['hackathon', 'hack', 'datathon', 'codeathon'],
            'FUNDING': ['funding', 'investment', 'round', 'series', 'seed', 'vc', 'venture'],
            'STARTUP': ['startup', 'entrepreneur', 'founder', 'innovation', 'incubator', 'accelerator'],
            'TECH': ['tech', 'technology', 'digital', 'ai', 'ml', 'blockchain', 'fintech', 'saas'],
            'BUSINESS': ['business', 'corporate', 'enterprise', 'company', 'industry'],
            'MEDIA': ['media', 'press', 'news', 'launch', 'announcement']
        }

        # Company categorization rules  
        self.company_categories = {
            'PLATFORM': ['platform', 'marketplace', 'network', 'hub', 'portal'],
            'SOFTWARE': ['software', 'app', 'application', 'tool', 'solution', 'system'],
            'SERVICE': ['service', 'services', 'consulting', 'advisory', 'support'],
            'ECOMMERCE': ['ecommerce', 'e-commerce', 'online', 'retail', 'store', 'shop'],
            'FINANCE': ['financial', 'finance', 'banking', 'payment', 'fintech', 'lending'],
            'HEALTH': ['health', 'healthcare', 'medical', 'wellness', 'fitness'],
            'EDUCATION': ['education', 'learning', 'training', 'educational', 'school'],
            'MEDIA': ['media', 'content', 'publishing', 'entertainment', 'video', 'streaming'],
            'SOCIAL': ['social', 'community', 'networking', 'communication', 'messaging'],
            'ANALYTICS': ['analytics', 'data', 'intelligence', 'insights', 'reporting'],
            'SECURITY': ['security', 'privacy', 'encryption', 'protection', 'safety'],
            'MOBILE': ['mobile', 'ios', 'android', 'smartphone', 'tablet']
        }

    def extract_key_terms(self, text: str, max_terms: int = 2) -> List[str]:
        """Extract meaningful terms from text - FIXED TO HANDLE SPACES PROPERLY"""
        if pd.isna(text) or text == "" or str(text).lower() in ['nan', 'none']:
            return []
        
        # Clean text - handle multiple languages by keeping alphanumeric AND spaces
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        
        # Split into words but then also look for multi-word phrases
        words = text.split()
        
        # Filter stopwords and short words
        meaningful_words = [
            w for w in words 
            if w not in self.stopwords and len(w) > 2 and w.replace('_', '').isalnum()
        ]
        
        # FIXED: Join consecutive words with underscores for multi-word terms
        # This handles cases like "New York" -> "new_york"
        if len(meaningful_words) >= 2:
            # Create combined terms for consecutive words
            combined_terms = []
            for i in range(len(meaningful_words) - 1):
                combined_term = f"{meaningful_words[i]}_{meaningful_words[i+1]}"
                combined_terms.append(combined_term)
            
            # Also include individual words
            result_terms = meaningful_words[:max_terms] + combined_terms[:max_terms-len(meaningful_words[:max_terms])]
            return result_terms[:max_terms]
        else:
            return meaningful_words[:max_terms]

    def categorize_event(self, description: str) -> str:
        """Categorize event based on description"""
        if pd.isna(description) or str(description).lower() in ['nan', 'none']:
            return "OTHER"
        
        desc_lower = str(description).lower()
        
        # Check categories
        for category, keywords in self.event_categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category
        
        return "OTHER"
    
    def categorize_company(self, description: str) -> str:
        """Categorize company based on description"""
        if pd.isna(description) or str(description).lower() in ['nan', 'none']:
            return "OTHER"
        
        desc_lower = str(description).lower()
        
        # Check categories
        for category, keywords in self.company_categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category
        
        return "OTHER"
    
    def process_event_description(self, description: str) -> List[str]:
        """Convert event description to meaningful tokens"""
        tokens = []

        # Add category token
        category = self.categorize_event(description)
        tokens.append(f"EVT_CAT_{category}")

        # Add key terms from description
        desc_terms = self.extract_key_terms(description, max_terms=2)
        for term in desc_terms:
            tokens.append(f"EVT_TERM_{term}")

        return tokens if tokens else ["EVT_UNKNOWN"]

    def process_person_description(self, description: str) -> List[str]:
        """Convert person bio to meaningful tokens"""
        if pd.isna(description) or str(description).lower() in ['nan', 'none']:
            return ["PPL_UNKNOWN"]

        # Extract job-related terms
        desc_lower = str(description).lower()
        job_keywords = ['founder', 'ceo', 'cto', 'cfo', 'vp', 'director', 'manager', 
                       'engineer', 'developer', 'analyst', 'consultant', 'executive',
                       'president', 'partner', 'lead', 'head', 'chief']

        tokens = []

        # Add job-related tokens
        for job in job_keywords:
            if job in desc_lower:
                tokens.append(f"PPL_JOB_{job.upper()}")

        # Add key terms
        key_terms = self.extract_key_terms(description, max_terms=2)
        for term in key_terms:
            tokens.append(f"PPL_TERM_{term}")

        return tokens if tokens else ["PPL_UNKNOWN"]
    
    def process_company_description(self, description: str) -> List[str]:
        """Convert company description to meaningful tokens"""
        if pd.isna(description) or str(description).lower() in ['nan', 'none']:
            return ["COMP_UNKNOWN"]

        tokens = []

        # Add category token
        category = self.categorize_company(description)
        tokens.append(f"COMP_CAT_{category}")

        # Add key terms
        key_terms = self.extract_key_terms(description, max_terms=2)
        for term in key_terms:
            tokens.append(f"COMP_TERM_{term}")

        return tokens if tokens else ["COMP_UNKNOWN"]

@dataclass
class StartupEventsSource(TokenSource):
    """SIMPLIFIED VERSION: Only process 3 main text fields + proper binning for DAYS_SINCE_FOUNDING"""
    
    name: str = "startup_events"
    test_mode: bool = False
    test_rows: int = 1000
    
    # SIMPLIFIED field list - only 3 new TOKEN fields + binned DAYS_SINCE_FOUNDING
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            # Core event info
            "EVENT_TYPE",
            "EVENT_name",              # Keep event names as unique identifiers
            
            # ONLY 3 processed text fields
            "EVENT_TOKENS",            # Processed EVENT_description
            "PEOPLE_TOKENS",           # Processed PEOPLE_description  
            "COMPANY_TOKENS",          # Processed company short_description
            
            # Keep original list fields as-is (they look fine)
            "EDU_degree_type",
            "EDU_institution",
            "EDU_subject",
            "EVENT_roles",
            "INV_investor_roles", 
            "PEOPLE_job_type",
            
            # Geographic and categorical fields (keep as-is)
            "ACQ_acquirer_city",
            "ACQ_acquirer_country_code",
            "ACQ_acquirer_region",
            "ACQ_acquirer_state_code", 
            "ACQ_acquisition_type",
            "ACQ_target_city",
            "ACQ_target_region",
            "ACQ_target_country_code",
            "ACQ_target_state_code",
            "EVENT_appearance_type",
            "EVENT_city", 
            "EVENT_country_code",
            "EVENT_state_code",
            "EVENT_region", 
            "INV_investment_type",
            "INV_investor_city",
            "INV_investor_country_code",
            "INV_investor_state_code", 
            "INV_investor_name",
            "INV_investor_region",
            "INV_investor_type",
            "INV_investor_types",
            "INV_is_lead",
            "INV_partner_name",
            "INV_round_city",
            "INV_round_country_code",
            "INV_round_region",
            "INV_round_state_code",
            "IPO_city",
            "IPO_country_code",
            "IPO_exchange",
            "IPO_region",
            "IPO_state_code",
            "IPO_symbol",
            "PEOPLE_city",
            "PEOPLE_country_code",
            "PEOPLE_job_title",
            "PEOPLE_region",
            "PEOPLE_state_code",

            # Binned numeric fields (these are fine)
            "ACQ_price_usd_binned",
            "INV_fund_size_usd_binned",
            "INV_investor_count_binned", 
            "INV_investor_investment_count_binned",
            "INV_post_money_valuation_usd_binned",
            "INV_raised_amount_usd_binned",
            "IPO_money_raised_usd_binned",
            "IPO_share_price_usd_binned",
            "IPO_valuation_usd_binned",

            # FIXED: Use binned version instead of raw DAYS_SINCE_FOUNDING
            "DAYS_SINCE_FOUNDING_BINNED",
        ]
    )
    
    # File paths
    input_csv: Path = DATA_ROOT / "cleaned" / "cleaned_startup" / "combined_events_cleaned.csv"
    input_pickle: Path = DATA_ROOT / "cleaned" / "cleaned_startup" / "combined_events_cleaned.pkl"

    def get_field_prefix(self, field_name: str) -> str:
        """Get the correct prefix for each field"""
        if field_name.startswith('ACQ_'):
            return 'ACQ_'
        elif field_name.startswith('EDU_'):
            return 'EDU_'
        elif field_name.startswith('EVENT_'):
            return 'EVT_'
        elif field_name.startswith('INV_'):
            return 'INV_'
        elif field_name.startswith('IPO_'):
            return 'IPO_'
        elif field_name.startswith('PEOPLE_'):
            return 'PPL_'
        elif field_name.startswith('COMPANY_'):
            return 'COMP_'
        elif field_name == 'EVENT_TYPE':
            return 'TYPE_'
        elif field_name == 'DAYS_SINCE_FOUNDING_BINNED':
            return 'DAYS_'
        else:
            return f"{field_name[:3].upper()}_"

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="recompute",
        verify_index=False,
        parquet_kwargs={
            "engine": "pyarrow",
            "compression": "snappy",
            "row_group_size": 50000,
            "write_index": True,
        }
    )
    def tokenized(self) -> dd.DataFrame:
        """SIMPLIFIED: Only process 3 text fields + bin DAYS_SINCE_FOUNDING"""
        mode_text = "TEST MODE" if self.test_mode else "FULL MODE"
        print(f"🚀 Starting SIMPLIFIED text tokenization ({mode_text})...")
        print_memory_usage("tokenization start")

        try:
            # Get processed data with text tokenization
            processed_data = self.indexed()
            print_memory_usage("after indexing")

            # Sort by date
            print("🔄 Sorting by RECORD_DATE...")
            result = processed_data.pipe(sort_partitions, columns=["RECORD_DATE"])
            print_memory_usage("after sorting")

            # Select required columns including uuid
            columns_to_keep = ["RECORD_DATE", "uuid"] + [f for f in self.field_labels() if f in result.columns]
            result = result[columns_to_keep]
            print_memory_usage("after column selection")

            # Apply prefixes to simple fields (the processed text fields already have tokens)
            print("🔧 Adding prefixes to categorical fields...")
            assign_dict = {}

            for field in self.field_labels():
                if field in result.columns:
                    # Skip token fields - they're already processed
                    if field.endswith('_TOKENS'):
                        continue

                    prefix = self.get_field_prefix(field)

                    # FIXED: Convert categorical columns to string first
                    if result[field].dtype.name == 'category':
                        result[field] = result[field].astype('string')

                    # FIXED: Handle spaces in values by replacing with underscores
                    if '_binned' in field:
                        # For binned fields, clean the values
                        cleaned_values = result[field].fillna(f"UNK_{field}").astype('string').str.replace(' ', '_', regex=False)
                        assign_dict[field] = prefix + cleaned_values
                    else:
                        # For regular fields, clean the values
                        cleaned_values = result[field].fillna("UNK").astype('string').str.replace(' ', '_', regex=False)
                        assign_dict[field] = prefix + cleaned_values

            # Apply transformations
            if assign_dict:
                result = result.assign(**assign_dict)

            print("✅ Simplified tokenization complete!")
            print_memory_usage("after tokenization")

            # FIXED: Handle index properly - use uuid as you specified
            print("🏁 Finalizing...")

            # Check what columns we have after reset_index
            result = result.reset_index()
            print(f"🔍 Available columns after reset: {list(result.columns)}")

            # Use uuid as the index (this is what your process expects)
            if 'uuid' in result.columns:
                result = result.set_index("uuid")
                print("✅ Set uuid as index")
            else:
                print(f"⚠️ Warning: uuid not found. Available: {list(result.columns)}")
                # If uuid doesn't exist, we have a problem - your process needs it
                raise ValueError("uuid column is required but not found!")

            result = result.repartition(partition_size="500MB")  # Larger partitions = less overhead

            print_memory_usage("tokenization complete")
            gc.collect()

            assert isinstance(result, dd.DataFrame)
            return result

        except Exception as e:
            print(f"❌ Error in tokenized method: {e}")
            print(f"🔍 Available columns: {list(result.columns) if 'result' in locals() else 'N/A'}")
            print_memory_usage("error state")
            gc.collect()
            raise

    def indexed(self) -> dd.DataFrame:
        """Process data with ONLY 3 text fields + bin DAYS_SINCE_FOUNDING"""
        print("🚀 Starting SIMPLIFIED text processing...")
        print_memory_usage("text processing start")

        try:
            parsed_data = self.parsed()
            print_memory_usage("after parsing")

            # Apply text tokenization AND days binning
            tokenizer = TextTokenizer()
            print("🔥 Processing ONLY 3 text fields + binning DAYS_SINCE_FOUNDING...")

            def process_partition_with_binning(df):
                """Process text fields AND bin DAYS_SINCE_FOUNDING for entire partition"""
                # Process event description
                event_tokens = df['EVENT_description'].apply(
                    lambda x: ' '.join(tokenizer.process_event_description(x))
                )
                df['EVENT_TOKENS'] = event_tokens

                # Process people description  
                people_tokens = df['PEOPLE_description'].apply(
                    lambda x: ' '.join(tokenizer.process_person_description(x))
                )
                df['PEOPLE_TOKENS'] = people_tokens

                # Process company description
                company_tokens = df['short_description'].apply(
                    lambda x: ' '.join(tokenizer.process_company_description(x))
                )
                df['COMPANY_TOKENS'] = company_tokens

                # IMPORTANT: Bin DAYS_SINCE_FOUNDING 
                if 'DAYS_SINCE_FOUNDING' in df.columns:
                    df['DAYS_SINCE_FOUNDING_BINNED'] = bin_days_since_founding(
                        df['DAYS_SINCE_FOUNDING'], 
                        strategy='quantile', 
                        n_bins=20
                    )
                else:
                    df['DAYS_SINCE_FOUNDING_BINNED'] = 'DAYS_UNK'

                return df

            # Apply text processing + binning with updated metadata
            print("⚡ Applying text processing + days binning to all partitions...")

            # Create updated metadata that includes ALL existing columns PLUS the new columns
            updated_meta = parsed_data._meta.copy()
            new_columns = ['EVENT_TOKENS', 'PEOPLE_TOKENS', 'COMPANY_TOKENS', 'DAYS_SINCE_FOUNDING_BINNED']
            for col in new_columns:
                updated_meta[col] = 'string'

            result = parsed_data.map_partitions(
                process_partition_with_binning,
                meta=updated_meta
            )

            print("✅ Simplified text processing + days binning complete!")
            print_memory_usage("text processing complete")

            gc.collect()
            assert isinstance(result, dd.DataFrame)
            return result

        except Exception as e:
            print(f"❌ Error in indexed method: {e}")
            print_memory_usage("text processing error")
            gc.collect()
            raise

    def parsed(self) -> dd.DataFrame:
        """Load and basic clean data"""
        mode_text = "TEST MODE" if self.test_mode else "FULL MODE"
        print(f"�� Loading data ({mode_text})...")
        print_memory_usage("parsing start")

        try:
            # Load data
            if self.input_pickle.exists():
                print(f"📁 Loading from pickle: {self.input_pickle}")
                with open(self.input_pickle, 'rb') as f:
                    df = pickle.load(f)
                if isinstance(df, list):
                    df = pd.DataFrame(df)
            else:
                print(f"📁 Loading from CSV: {self.input_csv}")
                df = pd.read_csv(self.input_csv)

            print(f"✅ Loaded {len(df)} rows")
            print_memory_usage("after loading")

            # Test mode truncation
            if self.test_mode:
                df = df.head(self.test_rows)
                print(f"🧪 TEST MODE: Using {len(df)} rows")

            # Load company data for short_description
            print("📊 Loading company descriptions...")
            company_data = self._load_company_data()
            if company_data is not None:
                print(f"✅ Loaded {len(company_data)} company descriptions")
                # Merge company descriptions
                df = df.merge(
                    company_data[['COMPANY_ID', 'short_description']], 
                    on='COMPANY_ID', 
                    how='left'
                )
                print(f"✅ Merged company descriptions")
            else:
                print("⚠️ No company data found, using empty descriptions")
                df['short_description'] = ""

            print_memory_usage("after company merge")

            # Basic processing
            df = self._process_dataframe_fast(df)
            print_memory_usage("after processing")

            # Create Dask DataFrame
            optimal_partitions = max(4, min(30, len(df) // 100000))
            print(f"🔧 Creating dask DataFrame with {optimal_partitions} partitions")
            ddf = dd.from_pandas(df, npartitions=optimal_partitions)

            del df
            gc.collect()

            print_memory_usage("parsing complete")
            assert isinstance(ddf, dd.DataFrame)
            return ddf

        except Exception as e:
            print(f"❌ Error in parsed method: {e}")
            print_memory_usage("parsing error")
            gc.collect()
            raise

    def _load_company_data(self) -> pd.DataFrame:
        """Load company data with descriptions"""
        try:
            # Try pickle first
            company_pickle = DATA_ROOT / "cleaned" / "cleaned_startup" / "company_base_cleaned.pkl"
            company_csv = DATA_ROOT / "cleaned" / "cleaned_startup" / "company_base_cleaned.csv"

            if company_pickle.exists():
                company_df = pd.read_pickle(company_pickle)
            elif company_csv.exists():
                company_df = pd.read_csv(company_csv)
            else:
                print("⚠️ No company data file found")
                return None

            # Keep only what we need
            if 'short_description' in company_df.columns:
                return company_df[['COMPANY_ID', 'short_description']].copy()
            else:
                print("⚠️ No short_description column in company data")
                return None

        except Exception as e:
            print(f"⚠️ Error loading company data: {e}")
            return None

    def _process_dataframe_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast dataframe processing"""
        print("🔥 Processing dataframe...")

        # Keep ALL columns (we need text ones for processing)
        df = df.copy()

        # Convert date
        df['RECORD_DATE'] = pd.to_datetime(df['RECORD_DATE'], errors='coerce')

        # IMPORTANT: Rename COMPANY_ID to uuid for consistency with startup pipeline
        if 'COMPANY_ID' in df.columns:
            df['uuid'] = df['COMPANY_ID']
            # Keep COMPANY_ID as well in case it's needed elsewhere
            print(f"✅ Created uuid column from COMPANY_ID")
        else:
            print(f"⚠️ Warning: COMPANY_ID not found in columns: {list(df.columns)}")

        # DON'T create founded_on here - that should come from population data
        # Events data only has event dates, not company founding dates

        # Drop invalid rows
        initial_len = len(df)
        required_cols = ['RECORD_DATE', 'uuid']
        df = df.dropna(subset=required_cols)
        if len(df) < initial_len:
            print(f"🧹 Dropped {initial_len - len(df)} rows with invalid data")

        # Optimize dtypes
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['RECORD_DATE', 'founded_on']:
                df[col] = df[col].astype('string')

        print(f"✅ Processed {len(df)} rows")
        return df

    def prepare(self) -> None:
        """Prepare with timing"""
        mode_text = "TEST MODE" if self.test_mode else "FULL MODE"
        print(f"🚀 Preparing {self.name} with SIMPLIFIED TEXT TOKENIZATION + QUANTILE BINNING ({mode_text})...")
        print_memory_usage("preparation start")

        # Print system info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        print(f"💻 System: {memory_gb:.1f} GB RAM, {cpu_count} CPU cores")

        try:
            import time
            start_time = time.time()
            self.tokenized()
            end_time = time.time()
            duration = end_time - start_time

            print(f"🎉 {self.name} preparation complete in {duration:.1f} seconds!")
            print_memory_usage("preparation complete")

        except Exception as e:
            print(f"❌ Error during preparation: {e}")
            print_memory_usage("preparation error")
            gc.collect()
            raise
