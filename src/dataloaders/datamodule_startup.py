# src/dataloaders/datamodule_startup.py
"""
Startup-specific data module using uuid and founded_on throughout
"""
import logging
import pandas as pd
import dask.dataframe as dd
from typing import List, Dict, Any, Optional
import torch
from pathlib import Path
import pickle
import numpy as np
from .datamodule import Corpus as BaseCorpus, L2VDataModule 
from .populations.base import Population
from .sources.base import TokenSource, Field
from .serialize import DATA_ROOT, ValidationError, _jsonify
from .ops import concat_columns_dask, concat_sorted
from .dataset import DocumentDataset, ShardedDocumentDataset
from .decorators import save_parquet
from torch.utils.data import Dataset
import dask

log = logging.getLogger(__name__)

class StartupCorpus(BaseCorpus):
    """
    Startup corpus that uses uuid and founded_on consistently
    """
    
    def __init__(
        self,
        name: str,
        sources: List[TokenSource],
        population: Population,
        reference_date: str = "2020-01-01",
        threshold: str = "2026-01-01",
    ):
        super().__init__(name, sources, population, reference_date, threshold)
        print(f"🏢 StartupCorpus initialized: {name} (using uuid and founded_on)")
    
    @property
    def _population_columns(self) -> List[str]:
        """All startup characteristics to include"""
        return [
            'roles', 'country_code', 'state_code', 'region', 'city', 
            'founded_on', 'short_description', 'category_list', 'category_groups_list',
            'num_funding_rounds_binned', 'total_funding_usd_binned', 'last_funding_on', 
            'closed_on', 'employee_count', 'num_exits_binned', 'status', 'company_age_years'
        ]
    
    def sentences(self, source: TokenSource) -> dd.DataFrame:
        """Create startup event sentences"""
        print(f"🔄 Creating sentences from source: {source.name}")
        tokenized = self.tokenized_and_transformed(source)
        field_labels = source.field_labels()
        print(f"✓ Processing {len(field_labels)} event fields")
        
        # Create sentences by concatenating event fields
        def create_startup_sentences(df):
            df = df.copy()
            sentences = []
            for _, row in df.iterrows():
                event_tokens = []
                for field in field_labels:
                    if field in row and pd.notna(row[field]) and str(row[field]).strip():
                        event_tokens.append(str(row[field]))
                sentence = ' '.join(event_tokens) if event_tokens else 'EMPTY_EVENT'
                sentences.append(sentence)
            df['SENTENCE'] = sentences
            return df[['RECORD_DATE', 'SENTENCE']]
        
        sentences_df = tokenized.map_partitions(
            create_startup_sentences, 
            meta=pd.DataFrame({
                'RECORD_DATE': pd.Series(dtype='datetime64[ns]'), 
                'SENTENCE': pd.Series(dtype='object')
            })
        )
        return sentences_df
    
    @save_parquet(
        DATA_ROOT / "processed/corpus/{self.name}/sentences/{split}",
        on_validation_error="recompute",
        verify_index=False,
    )
    def combined_sentences(self, split: str) -> dd.DataFrame:
        """
        Create combined startup sequences - keeps uuid as index, uses founded_on not BIRTHDAY
        """
        print(f"🏢 StartupCorpus: Creating combined sentences for {split} split (uuid + founded_on)")
        try:
            # Get startup population
            population: pd.DataFrame = self.population.population()
            print(f"✓ Population shape: {population.shape}")
            print(f"📋 Population index: {population.index.name}")
            
            # Get split startup IDs
            data_split = getattr(self.population.data_split(), split)
            print(f"✓ Split '{split}' size: {len(data_split):,} startups")
            
            # Get event sentences
            sentences_parts = []
            for i, source in enumerate(self.sources):
                print(f"  Processing source {i+1}/{len(self.sources)}: {source.name}")
                source_sentences = self.sentences(source)
                sentences_parts.append(source_sentences)
            
            # Combine sources
            if len(sentences_parts) > 1:
                combined_sentences = concat_sorted(sentences_parts, columns=["RECORD_DATE"])
            else:
                combined_sentences = sentences_parts[0]
            
            print(f"📋 Combined sentences index: {combined_sentences.index.name}")
            
            # Filter to split
            split_ids_set = set(data_split)
            combined_sentences = combined_sentences.loc[
                combined_sentences.index.isin(split_ids_set)
            ]
            
            # Join with population
            available_cols = [col for col in self._population_columns if col in population.columns]
            print(f"✓ Using {len(available_cols)}/{len(self._population_columns)} population columns")
            
            population_subset = population[available_cols]
            combined_sentences = combined_sentences.join(population_subset, how='left')
            
            # FIXED: Use founded_on instead of BIRTHDAY
            if 'founded_on' in combined_sentences.columns:
                print("✓ Processing founded_on dates")
                combined_sentences['founded_on'] = combined_sentences['founded_on'].apply(
                    lambda x: pd.to_datetime(x, errors='coerce'),
                    meta=('founded_on', 'datetime64[ns]')
                )
                
                # Calculate AGE based on founded_on (company age, not person age)
                combined_sentences["AGE"] = combined_sentences.apply(
                    lambda x: max(0, (x.RECORD_DATE.year - x.founded_on.year)) 
                    if pd.notna(x.founded_on) else 0, 
                    axis=1, 
                    meta=('AGE', 'int64')
                )
                print("✓ Calculated company AGE from founded_on")
            else:
                print("⚠️ founded_on not found, using default AGE=0")
                combined_sentences["AGE"] = 0
            
            # Add threshold flag
            combined_sentences["AFTER_THRESHOLD"] = (
                combined_sentences.RECORD_DATE >= self._threshold
            )
            
            # Convert dates to days
            combined_sentences["RECORD_DATE"] = (
                combined_sentences.RECORD_DATE - self._reference_date
            ).dt.days.astype(int)
            
            # Final preparation - KEEP uuid as index
            combined_sentences = combined_sentences.reset_index()
            
            # The index column should be uuid
            index_col = combined_sentences.columns[0]
            print(f"📋 Keeping original index column: '{index_col}' (should be uuid)")
            
            # Set the uuid column as index
            combined_sentences = combined_sentences.set_index(index_col, sorted=True)
            
            # Optimize partitioning
            combined_sentences = combined_sentences.repartition(partition_size="100MB")
            
            print(f"✅ {split} split completed!")
            print(f"📊 Final index: {combined_sentences.index.name}")
            print(f"📊 Final columns: {list(combined_sentences.columns)}")
            
            return combined_sentences
            
        except Exception as e:
            print(f"❌ Error in startup combined_sentences for {split}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare(self) -> None:
        """Prepare all splits"""
        print("🚀 Starting startup corpus preparation (uuid + founded_on)...")
        for i, split in enumerate(["train", "val", "test"]):
            try:
                print(f"\n{'='*50}")
                print(f"PREPARING {split.upper()} SPLIT ({i+1}/3)")
                print(f"{'='*50}")
                result = self.combined_sentences(split)
                print(f"✅ {split} split prepared successfully")
            except Exception as e:
                print(f"❌ Error preparing {split} split: {e}")
                raise
        print("\n🎉 All startup corpus preparation complete!")

class StartupDataModule(L2VDataModule):
    """
    Startup DataModule that works with uuid-based data
    """
    
    def prepare_data_split(self, split: str) -> dd.Series:
        """Prepare data split - works with uuid index"""
        data = self.corpus.combined_sentences(split)
        N = data.npartitions
        
        def process_startup_partition(
            partition: pd.DataFrame, 
            partition_info: Optional[Dict[str, int]] = None
        ) -> bool:
            assert partition_info is not None
            
            from math import log10
            file_name = str(partition_info["number"]).zfill(
                int(log10(N) + 1) if N > 1 else 1) + ".hdf5"
            path = self.dataset_root / split / file_name
            
            # Group by index (uuid) - no need to look for USER_ID
            records = []
            for startup_uuid, group in partition.groupby(level=0):
                try:
                    # Use startup-specific task method
                    doc = self.task.get_document(group)
                    records.append(doc)
                except Exception as e:
                    log.warning(f"Error creating document for startup {startup_uuid}: {e}")
                    continue
            
            DocumentDataset(file=path).save_data(records)
            
            if N > 10 and partition_info["number"] % max(1, N//10) == 0:
                print(f"  ✓ Processed {partition_info['number']}/{N} partitions")
            
            return True
        
        result = data.map_partitions(process_startup_partition, meta=(None, bool))
        return result
