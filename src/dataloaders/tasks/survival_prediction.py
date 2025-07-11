# src/dataloaders/tasks/survival_prediction.py
"""
Startup Survival Prediction Task
Following life2vec methodology for binary classification
"""

import logging
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional
from functools import cached_property
from pathlib import Path

from ..types import StartupBackground, StartupDocument, EncodedDocument
from .startup_base import StartupTask

log = logging.getLogger(__name__)

@dataclass
class SurvivalPredictionEncodedDocument(EncodedDocument["SurvivalPrediction"]):
    """Encoded document for survival prediction task"""
    sequence_id: np.ndarray
    input_ids: np.ndarray           # [4, max_length] - life2vec format
    padding_mask: np.ndarray        # [max_length]
    survival_label: np.ndarray      # Binary: 0=died, 1=survived
    prediction_window: np.ndarray   # Which prediction window (1-4 years)
    company_founded_year: np.ndarray
    company_age_at_prediction: np.ndarray
    original_sequence: np.ndarray

@dataclass  
class SurvivalPrediction(StartupTask):
    """
    Startup survival prediction task following life2vec methodology
    
    Predicts 5-year survival using 1-4 year temporal windows
    """
    
    # Task configuration
    name: str = "survival_prediction"
    max_length: int = 512
    no_sep: bool = False
    
    # Prediction configuration  
    prediction_windows: list = None  # [1, 2, 3, 4] years
    survival_horizon: int = 5        # Predict survival to 5 years
    
    # Data augmentation (same as life2vec)
    p_sequence_timecut: float = 0.0   # Disable during finetuning
    p_sequence_resample: float = 0.0  # Disable during finetuning  
    p_sequence_abspos_noise: float = 0.1  # Keep some noise
    p_sequence_hide_background: float = 0.01
    p_sentence_drop_tokens: float = 0.01
    shuffle_within_sentences: bool = True
    
    def __post_init__(self):
        if self.prediction_windows is None:
            self.prediction_windows = [1, 2, 3, 4]
    
    def get_survival_labels(self) -> pd.DataFrame:
        """
        Create survival labels from company base data
        Following methodology: survived=1 (operating/acquired/ipo), died=0 (closed with date)
        
        STRICT FILTERING:
        - Companies without founded_on: EXCLUDE
        - Companies closed without closed_on date: EXCLUDE  
        - Only include companies with clear, complete information
        """
        # Load company base data
        company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.csv")
        if not company_path.exists():
            company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.pkl")
            company_df = pd.read_pickle(company_path)
        else:
            company_df = pd.read_csv(company_path)
        
        log.info(f"Loading company data from {company_path}")
        log.info(f"Company data shape: {company_df.shape}")
        
        # Create survival labels with strict filtering
        survival_labels = pd.DataFrame(index=company_df.index)
        survival_labels['survival_label'] = None
        survival_labels['founded_on'] = pd.to_datetime(company_df['founded_on'], errors='coerce')
        survival_labels['closed_on'] = pd.to_datetime(company_df.get('closed_on'), errors='coerce')
        survival_labels['status'] = company_df['status']
        
        # STRICT FILTERING: Only companies with valid founded_on dates
        founded_valid_mask = survival_labels['founded_on'].notna()
        log.info(f"Companies with valid founded_on: {founded_valid_mask.sum():,}/{len(company_df):,}")
        
        # Survived companies: operating, acquired, ipo (WITH valid founded_on)
        survived_mask = (
            company_df['status'].isin(['operating', 'acquired', 'ipo']) & 
            founded_valid_mask
        )
        survival_labels.loc[survived_mask, 'survival_label'] = 1
        
        # Died companies: closed AND has explicit closure date AND valid founded_on
        died_mask = (
            (company_df['status'] == 'closed') & 
            (company_df['closed_on'].notna()) &
            founded_valid_mask
        )
        survival_labels.loc[died_mask, 'survival_label'] = 0
        
        # EXCLUDE companies that are closed but have no closed_on date
        closed_no_date = (company_df['status'] == 'closed') & (company_df['closed_on'].isna())
        log.info(f"Excluded closed companies without closure date: {closed_no_date.sum():,}")
        
        # Filter to only companies with valid labels
        valid_mask = survival_labels['survival_label'].notna()
        survival_labels = survival_labels.loc[valid_mask]
        
        # Log statistics
        total_valid = len(survival_labels)
        survived_count = (survival_labels['survival_label'] == 1).sum()
        died_count = (survival_labels['survival_label'] == 0).sum()
        
        log.info(f"STRICT FILTERING - Survival label statistics:")
        log.info(f"  Total valid companies: {total_valid:,}")
        log.info(f"  Survived (1): {survived_count:,} ({survived_count/total_valid*100:.1f}%)")
        log.info(f"  Died (0): {died_count:,} ({died_count/total_valid*100:.1f}%)")
        log.info(f"  Companies excluded: {len(company_df) - total_valid:,}")
        
        return survival_labels
    
    def create_temporal_cutoff(self, document: StartupDocument, 
                             window_years: int, survival_labels: pd.DataFrame) -> StartupDocument:
        """
        Apply temporal cutoff for prediction window
        Only include events up to window_years after founding
        """
        startup_id = document.startup_id
        
        # Get company info
        if startup_id not in survival_labels.index:
            # Return empty document for companies without survival labels
            return StartupDocument(
                startup_id=startup_id,
                sentences=[],
                abspos=[],
                age=[],
                timecut_pos=0,
                background=document.background,
                segment=[]
            )
        
        company_info = survival_labels.loc[startup_id]
        founded_on = company_info['founded_on']
        
        if pd.isna(founded_on):
            # Return empty document if no founding date
            return StartupDocument(
                startup_id=startup_id,
                sentences=[],
                abspos=[],
                age=[],
                timecut_pos=0,
                background=document.background,
                segment=[]
            )
        
        # Calculate cutoff date (window_years after founding)
        cutoff_days = window_years * 365.25  # Account for leap years
        reference_date = pd.to_datetime("2020-01-01")  # Your corpus reference date
        founded_days = (founded_on - reference_date).days
        cutoff_abspos = founded_days + cutoff_days
        
        # Filter events to those before cutoff
        filtered_sentences = []
        filtered_abspos = []
        filtered_age = []
        filtered_segment = []
        
        for i, abspos in enumerate(document.abspos):
            if abspos <= cutoff_abspos:
                filtered_sentences.append(document.sentences[i])
                filtered_abspos.append(document.abspos[i])
                filtered_age.append(document.age[i])
                if document.segment:
                    filtered_segment.append(document.segment[i])
        
        # If no events after filtering, create minimal document
        if not filtered_sentences:
            filtered_sentences = [["NO_EVENTS"]]  # Special token for companies with no events
            filtered_abspos = [founded_days]
            filtered_age = [0.0]
            filtered_segment = [1]
        
        return StartupDocument(
            startup_id=startup_id,
            sentences=filtered_sentences,
            abspos=filtered_abspos,
            age=filtered_age,
            timecut_pos=len(filtered_sentences),
            background=document.background,
            segment=filtered_segment
        )
    
    def encode_document(self, document: StartupDocument) -> SurvivalPredictionEncodedDocument:
        """
        Encode startup document for survival prediction
        
        IMPORTANT: We should NOT re-tokenize! The corpus already has tokenized sentences.
        We just need to:
        1. Get the pre-tokenized sentence from corpus  
        2. Apply temporal cutoff
        3. Create 4D format for life2vec
        """
        # Get survival labels
        survival_labels = self.get_survival_labels()
        
        startup_id = document.startup_id
        
        # Default values for missing companies
        survival_label = -1  # Will be filtered out
        company_founded_year = -1  # Invalid marker
        prediction_window = 1
        
        if startup_id in survival_labels.index:
            company_info = survival_labels.loc[startup_id]
            survival_label = company_info['survival_label']
            if pd.notna(company_info['founded_on']):
                company_founded_year = company_info['founded_on'].year
        
        # If no valid data, return invalid example
        if survival_label == -1 or company_founded_year == -1:
            return SurvivalPredictionEncodedDocument(
                sequence_id=np.array([-1]),
                input_ids=np.zeros((4, self.max_length), dtype=np.int64),
                padding_mask=np.zeros(self.max_length, dtype=bool),
                survival_label=np.array([-1]),  # Invalid - will be filtered
                prediction_window=np.array([prediction_window]),
                company_founded_year=np.array([company_founded_year]),
                company_age_at_prediction=np.array([prediction_window]),
                original_sequence=np.zeros(self.max_length, dtype=np.int64)
            )
        
        # REUSE PRE-TOKENIZED DATA: document.sentences are already tokenized!
        # Each sentence is already a list of tokens, not raw text
        
        # Create prefix sentence (company background)
        prefix_sentence = (
            ["[CLS]"] +
            StartupBackground.get_sentence(document.background) + ["[SEP]"]
        )
        
        # Build full sentence sequence - sentences are already tokenized
        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]
        
        # Expand metadata to token level
        def expand(x):
            return list(
                sum(([val] * length for val, length in zip([0] + x, sentence_lengths)), [])
            )
        
        abspos_expanded = expand(document.abspos)
        age_expanded = expand(document.age)
        
        # Segment expansion
        if document.segment:
            segment_expanded = expand(document.segment)
        else:
            segment_expanded = [1] * len(abspos_expanded)  # Default segment
        
        # Flatten sentences (these are already tokens, not text!)
        flat_sentences = []
        for sentence in sentences:
            flat_sentences.extend(sentence)
        
        # Convert tokens to indices using existing vocabulary
        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index.get("[UNK]", len(token2index) - 1)
        
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])
        
        length = min(len(token_ids), self.max_length)
        
        # Create 4D input format [4, max_length] - life2vec format
        input_ids = np.zeros((4, self.max_length), dtype=np.int64)
        input_ids[0, :length] = token_ids[:length]      # tokens
        input_ids[1, :length] = abspos_expanded[:length]  # absolute positions
        input_ids[2, :length] = age_expanded[:length]     # ages
        input_ids[3, :length] = segment_expanded[:length] # segments
        
        # Padding mask
        padding_mask = np.zeros(self.max_length, dtype=bool)
        padding_mask[:length] = True
        
        # Original sequence for reference
        original_sequence = np.zeros(self.max_length, dtype=np.int64)
        original_sequence[:length] = token_ids[:length]
        
        return SurvivalPredictionEncodedDocument(
            sequence_id=np.array([hash(startup_id) % 1000000]),  # Numeric ID
            input_ids=input_ids,
            padding_mask=padding_mask,
            survival_label=np.array([survival_label]),
            prediction_window=np.array([prediction_window]),
            company_founded_year=np.array([company_founded_year]),
            company_age_at_prediction=np.array([prediction_window]),  # Age at prediction time
            original_sequence=original_sequence
        )
    
    def get_document(self, startup_sentences: pd.DataFrame) -> StartupDocument:
        """
        Override to create multiple temporal windows per company
        This will be called for each startup in your corpus
        """
        # Use the startup_base.py implementation but extend for temporal windows
        base_document = super().get_document(startup_sentences)
        
        # For now, return base document - temporal windowing will be handled in encode_document
        return base_document
