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
    """Vocabulary for startup data that handles the different ID column name"""
    
    @save_pickle(
        DATA_ROOT / "interim/vocab/{self.name}/token_counts/{source.name}",
        on_validation_error="recompute",
    )
    def token_counts(self, source) -> Dict[str, Dict[str, int]]:
        """Returns the token counts and preserves the prefixes"""
        print(f"Getting tokenized data for source {source.name}")
        
        # Get the tokenized data with prefixes
        tokenized = source.tokenized()
        
        print("Computing tokenized data")
        # For dask DataFrame, compute the entire DataFrame once to avoid repeated computations
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
        
        # Filter to training IDs
        try:
            tokenized_filtered = tokenized_computed.loc[lambda x: x.index.isin(ids)]
        except Exception as e:
            print(f"Error filtering by IDs: {e}")
            # Just use what we have
            tokenized_filtered = tokenized_computed
        
        counts = {}
        for field_ in source.field_labels():
            try:
                print(f"Counting tokens in field: {field_}")
                
                # Check what's actually in the column
                sample_values = tokenized_filtered[field_].head(5).tolist()
                print(f"First 5 values in {field_}: {sample_values}")
                
                # Check if prefixes are already present
                has_prefix = False
                if len(sample_values) > 0 and isinstance(sample_values[0], str):
                    has_prefix = field_ in sample_values[0]
                
                if has_prefix:
                    print(f"Field {field_} already has prefixes")
                    # Use values directly
                    counts[field_] = tokenized_filtered[field_].value_counts().to_dict()
                else:
                    print(f"Field {field_} needs prefixes added")
                    # Add the prefix
                    prefixed_values = tokenized_filtered[field_].apply(
                        lambda x: f"{field_}_{x}" if pd.notna(x) else None
                    )
                    counts[field_] = prefixed_values.dropna().value_counts().to_dict()
                
            except Exception as e:
                print(f"Error in token_counts for field {field_}: {e}")
                import traceback
                traceback.print_exc()
                counts[field_] = {}
        
        print(f"Found counts for fields: {list(counts.keys())}")
        return counts
    
    @save_tsv(DATA_ROOT / "processed/vocab/{self.name}/", on_validation_error="error")
    def vocab(self) -> pd.DataFrame:
        """Modified vocab method that handles mixed token types"""
        # Keep the same base code from CorpusVocabulary
        general = pd.DataFrame(
            {"TOKEN": self.general_tokens, "CATEGORY": "GENERAL"})
        background = pd.DataFrame(
            {"TOKEN": self.background_tokens, "CATEGORY": "BACKGROUND"}
        )
        month = pd.DataFrame(
            {"TOKEN": [f"MONTH_{i}" for i in range(
                1, 13)], "CATEGORY": "MONTH"}
        )
        year = pd.DataFrame(
            {
                "TOKEN": [
                    f"YEAR_{i}"
                    for i in range(self.year_range[0], self.year_range[1] + 1)
                ],
                "CATEGORY": "YEAR",
            }
        )

        vocab_parts = [general, background, month, year]

        # Define a safer sort_key function that ensures all tokens are strings
        def safe_sort_key(x):
            # Convert to string first
            x_str = str(x)
        
            # Split by underscore
            parts = x_str.split("_")
            result = []
        
            # Process each part
            for part in parts:
                # Try to convert to int if it's numeric
                try:
                    if part.isdigit():
                        result.append(int(part))
                    else:
                        result.append(part)
                except (ValueError, AttributeError):
                    result.append(part)
                
            return tuple(result)

        for source in self.corpus.sources:
            print(f"Building vocabulary for source: {source.name}")
            token_counts = self.token_counts(source)
        
            for label in source.field_labels():
                counts = token_counts[label]
                min_count = self.min_token_count_field.get(
                    label, self.min_token_count)
            
                # Get tokens that meet minimum count threshold
                tokens = [k for k, v in counts.items() if v >= min_count]
                print(f"Field {label}: Found {len(tokens)} tokens with min_count>={min_count}")
            
                if len(tokens) > 0:
                    print(f"Sample tokens from {label}: {tokens[:3]}")
                
                    # Make sure all tokens are strings
                    string_tokens = [str(t) if t is not None else "None" for t in tokens]
                
                    # Use try/except for sorting
                    try:
                        string_tokens.sort(key=safe_sort_key)
                    except Exception as e:
                        print(f"Warning: Couldn't sort tokens for {label}: {e}")
                        # Just sort as strings without the custom key
                        string_tokens.sort()
                    
                    # Create the dataframe with the sorted string tokens    
                    tokens_df = pd.DataFrame({"TOKEN": string_tokens, "CATEGORY": label})
                    vocab_parts.append(tokens_df)

        # Important: Use rename_axis to properly name the index
        result = pd.concat(vocab_parts, ignore_index=True).rename_axis(index="ID")
        print(f"Final vocabulary has {len(result)} entries")
        return result

    # @save_tsv(DATA_ROOT / "processed/vocab/{self.name}/", on_validation_error="error")
    # def vocab(self) -> pd.DataFrame:
    #     """Modified vocab method that handles non-string tokens"""
    #     # Keep the same base code from CorpusVocabulary
    #     general = pd.DataFrame(
    #         {"TOKEN": self.general_tokens, "CATEGORY": "GENERAL"})
    #     background = pd.DataFrame(
    #         {"TOKEN": self.background_tokens, "CATEGORY": "BACKGROUND"})
    #     month = pd.DataFrame(
    #         {"TOKEN": [f"MONTH_{i}" for i in range(
    #             1, 13)], "CATEGORY": "MONTH"})
    #     year = pd.DataFrame(
    #         {"TOKEN": [f"YEAR_{i}" for i in range(self.year_range[0], self.year_range[1] + 1)],
    #          "CATEGORY": "YEAR"})

    #     vocab_parts = [general, background, month, year]

    #     # Define a safer sort_key function
    #     def safe_sort_key(x):
    #         if not isinstance(x, str):
    #             # Convert non-string tokens to strings
    #             x = str(x)
            
    #         parts = x.split("_")
    #         result = []
    #         for part in parts:
    #             # Try to convert to int if it's numeric
    #             try:
    #                 if part.isdigit():
    #                     result.append(int(part))
    #                 else:
    #                     result.append(part)
    #             except (ValueError, AttributeError):
    #                 result.append(part)
    #         return tuple(result)

    #     for source in self.corpus.sources:
    #         print(f"Building vocabulary for source: {source.name}")
    #         token_counts = self.token_counts(source)
            
    #         for label in source.field_labels():
    #             counts = token_counts[label]
    #             min_count = self.min_token_count_field.get(
    #                 label, self.min_token_count)
                
    #             # Get tokens that meet minimum count threshold
    #             tokens = [k for k, v in counts.items() if v >= min_count]
    #             print(f"Field {label}: Found {len(tokens)} tokens with min_count>={min_count}")
                
    #             if len(tokens) > 0:
    #                 print(f"Sample tokens from {label}: {tokens[:3]}")
                
    #             # Use try/except for sorting
    #             try:
    #                 tokens.sort(key=safe_sort_key)
    #             except Exception as e:
    #                 print(f"Warning: Couldn't sort tokens for {label}: {e}")
    #                 # Just convert to strings and sort normally
    #                 tokens = [str(t) for t in tokens]
    #                 tokens.sort()
                    
    #             # Create the CATEGORY field entries for this label    
    #             tokens_df = pd.DataFrame({"TOKEN": tokens, "CATEGORY": label})
    #             vocab_parts.append(tokens_df)

    #     # Important: Use rename_axis to properly name the index
    #     result = pd.concat(vocab_parts, ignore_index=True).rename_axis(index="ID")
    #     print(f"Final vocabulary has {len(result)} entries")
    #     return result
    
    # # Override these methods to handle the index issue
    # def tokens(self) -> List[str]:
    #     """Return the tokens in order as list of strings"""
    #     df = self.vocab()
    #     return cast(List[str], df["TOKEN"].astype("string").to_list())

    # def token_ids(self) -> List[int]:
    #     """Return the token ids in order as a list of integers"""
    #     df = self.vocab()
    #     return cast(List[int], list(range(len(df))))  # Use range instead of ID columny
