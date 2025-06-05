from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from .base import DataSplit, Population


@dataclass
class StartupPopulation(Population):
    """
    A cohort of startups. This object stores static information about each startup, as well as creates datasplits.

    :param name: Name of the population
    :param input_csv: path to the startup base (or pickle file)
    :param seed: Seed for splitting training, validation and test dataset
    :param train_val_test: Fraction of the data to be included in the three data splits.
        Must sum to 1.
    """

    name: str = "startups"
    input_csv: Path = DATA_ROOT / "cleaned" / "cleaned_startup" / "company_base_cleaned.csv"
    input_pickle: Path = DATA_ROOT / "cleaned" / "cleaned_startup" / "company_base_cleaned.pkl"

    seed: int = 42
    train_val_test: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        """
        Perform operations right after the object was initialized
        """
        assert sum(self.train_val_test) == 1.0

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """
        Creates (or loads) the population base. 
        Load from pickle if available, otherwise from CSV.
        """
        print("Loading startup population data...")
        
        # Try to load from pickle first, then CSV
        if self.input_pickle.exists():
            print(f"Loading from pickle file: {self.input_pickle}")
            companies_df = pd.read_pickle(self.input_pickle)
        else:
            print(f"Loading from CSV file: {self.input_csv}")
            companies_df = pd.read_csv(self.input_csv)
        
        print(f"Loaded {len(companies_df)} companies from file")
        print(f"Available columns: {list(companies_df.columns)[:10]}...")
        
        # Select relevant columns for static information
        columns_to_keep = [
            'COMPANY_ID', 'roles', 'country_code', 'state_code', 'region', 
            'city', 'founded_on', 'short_description', 'category_list', 
            'category_groups_list', 'num_funding_rounds_binned', 
            'total_funding_usd_binned', 'last_funding_on', 'closed_on', 
            'employee_count', 'num_exits_binned', 'status', 'company_age_years'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in columns_to_keep if col in companies_df.columns]
        missing_columns = [col for col in columns_to_keep if col not in companies_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}")
        
        result = companies_df[available_columns].copy()
        
        # Convert founded_on to datetime if it's not already
        if 'founded_on' in result.columns:
            result['founded_on'] = pd.to_datetime(result['founded_on'], errors='coerce')
            
            # Exclude companies with missing founded_on (should have been cleaned already)
            initial_count = len(result)
            result = result.dropna(subset=['founded_on'])
            if len(result) < initial_count:
                print(f"Warning: Excluded {initial_count - len(result)} companies with missing founded_on")
        
        # Rename COMPANY_ID to match life2vec expectations
        if 'COMPANY_ID' in result.columns:
            result = result.rename(columns={'COMPANY_ID': 'USER_ID'})
        
        # Set USER_ID as index (following life2vec pattern)
        if 'USER_ID' in result.columns:
            result = result.set_index('USER_ID')
        
        print(f"Final population: {len(result)} companies")
        print(f"Index name: {result.index.name}")
        print(f"Sample data shape: {result.shape}")
        
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """
        Split data based on :attr:`seed` using :attr:`train_val_test` as ratios
        """
        print("Creating data splits...")
        population_df = self.population()
        ids = population_df.index.to_numpy()
        
        print(f"Total companies for splitting: {len(ids)}")
        
        np.random.default_rng(self.seed).shuffle(ids)
        split_idxs = np.round(np.cumsum(self.train_val_test) * len(ids))[:2].astype(int)
        train_ids, val_ids, test_ids = np.split(ids, split_idxs)
        
        print(f"Split sizes: Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
        
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )

    def prepare(self) -> None:
        """Prepares the population by calling population and data_split."""
        print(f"Preparing {self.name} population...")
        self.population()
        self.data_split()
        print(f"{self.name} population preparation complete!")
