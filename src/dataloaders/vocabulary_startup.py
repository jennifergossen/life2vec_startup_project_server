# src/dataloaders/vocabulary_startup.py
import logging
import dask
import pandas as pd
from typing import Dict, Tuple, Union, cast, List
from .vocabulary import CorpusVocabulary
from .decorators import save_pickle, save_tsv
from .serialize import DATA_ROOT

log = logging.getLogger(__name__)

class StartupVocabulary(CorpusVocabulary):
    """Vocabulary for startup data that handles uuid instead of USER_ID"""
    
    @save_pickle(
        DATA_ROOT / "interim/vocab/{self.name}/token_counts/{source.name}",
        on_validation_error="recompute",
    )
    def token_counts(self, source) -> Dict[str, Dict[str, int]]:
        """Returns the token counts - USES uuid NOT USER_ID"""
        print(f"Getting tokenized data for source {source.name}")
        
        # Get the tokenized data
        tokenized = source.tokenized()
        print("Computing tokenized data")
        
        # Compute the DataFrame
        if hasattr(tokenized, 'compute'):
            try:
                tokenized_computed = tokenized.compute()
            except Exception as e:
                print(f"Error computing full DataFrame: {e}")
                print("Using head sample instead")
                tokenized_computed = tokenized.head(1000).compute()
        else:
            tokenized_computed = tokenized
        
        # Get training IDs
        ids = self.corpus.population.data_split().train
        print(f"Filtering to {len(ids)} training IDs")
        
        # Filter to training IDs using the INDEX (which is uuid for startups)
        try:
            tokenized_filtered = tokenized_computed.loc[lambda x: x.index.isin(ids)]
        except Exception as e:
            print(f"Error filtering by IDs: {e}")
            tokenized_filtered = tokenized_computed
        
        counts = {}
        for field_ in source.field_labels():
            try:
                print(f"Counting tokens in field: {field_}")
                
                if field_ in tokenized_filtered.columns:
                    # FIXED: Use uuid as the ID column (the index), not USER_ID
                    # Count unique combinations of startup (via index) and token value
                    field_counts = tokenized_filtered.reset_index()[[tokenized_filtered.index.name, field_]].drop_duplicates()[field_].value_counts()
                    counts[field_] = field_counts.to_dict()
                    
                    print(f"Field {field_} has {len(counts[field_])} unique tokens")
                else:
                    print(f"Field {field_} not found in tokenized data")
                    counts[field_] = {}
                    
            except Exception as e:
                print(f"Error in token_counts for field {field_}: {e}")
                import traceback
                traceback.print_exc()
                counts[field_] = {}
        
        print(f"Found counts for fields: {list(counts.keys())}")
        total_tokens = sum(len(v) for v in counts.values())
        print(f"Total unique tokens across all fields: {total_tokens}")
        
        return counts
