# src/dataloaders/exit_datamodule.py
"""
Exit Prediction DataModule - FIXED VERSION
Fixed the index issue that was causing 0 training examples
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Import only what we need for exit prediction
from .tasks.exit_prediction import ExitPrediction

log = logging.getLogger(__name__)

class ExitDataset(Dataset):
    """
    Dataset for exit prediction that creates multiple temporal windows per company
    """
    
    def __init__(
        self,
        corpus_sentences: pd.DataFrame,
        task: ExitPrediction,
        split: str = "train"
    ):
        self.corpus_sentences = corpus_sentences
        self.task = task
        self.split = split
        
        # Get exit labels
        self.exit_labels = task.get_exit_labels()
        
        # Create dataset examples (multiple windows per company)
        self.examples = self._create_examples()
        
        log.info(f"Created {len(self.examples)} examples for {split} split")
    
    def _create_examples(self):
        """
        Create multiple training examples per company (one per prediction window)
        """
        examples = []
        
        # Group sentences by company UUID (FIXED: now works with proper index)
        grouped = self.corpus_sentences.groupby(level=0)  # Group by index (UUID)
        
        for company_uuid, company_sentences in grouped:
            # Skip companies without exit labels
            if company_uuid not in self.exit_labels.index:
                continue
            
            company_info = self.exit_labels.loc[company_uuid]
            exit_label = company_info['exit_label']
            
            # Skip invalid labels
            if pd.isna(exit_label) or exit_label < 0:
                continue
            
            # Create one example per prediction window
            for window_years in self.task.prediction_windows:
                examples.append({
                    'company_uuid': company_uuid,
                    'window_years': window_years,
                    'exit_label': int(exit_label),
                    'company_sentences': company_sentences
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single training example with temporal cutoff applied
        """
        example = self.examples[idx]
        
        # Get base document from sentences
        document = self.task.get_document(example['company_sentences'])
        
        # Apply temporal cutoff for this window
        document = self.task.create_temporal_cutoff(
            document, 
            example['window_years'], 
            self.exit_labels
        )
        
        # Encode document
        encoded = self.task.encode_document(document)
        
        # Override exit label and window info
        encoded.exit_label = np.array([example['exit_label']])
        encoded.prediction_window = np.array([example['window_years']])
        
        # Convert to tensors (maintaining your 4D format)
        return {
            'sequence_id': torch.tensor(encoded.sequence_id, dtype=torch.long),
            'input_ids': torch.tensor(encoded.input_ids, dtype=torch.long),
            'padding_mask': torch.tensor(encoded.padding_mask, dtype=torch.bool),
            'exit_label': torch.tensor(encoded.exit_label, dtype=torch.long),
            'prediction_window': torch.tensor(encoded.prediction_window, dtype=torch.long),
            'company_founded_year': torch.tensor(encoded.company_founded_year, dtype=torch.long),
            'company_age_at_prediction': torch.tensor(encoded.company_age_at_prediction, dtype=torch.long)
        }

class ExitDataModule(pl.LightningDataModule):
    """
    DataModule for startup exit prediction - FIXED VERSION
    Now properly handles the corpus data indexing
    """
    
    def __init__(
        self,
        corpus_name: str = "startup_corpus",
        vocab_name: str = "startup_vocab", 
        batch_size: int = 32,
        num_workers: int = 4,
        prediction_windows: list = None,
        **kwargs
    ):
        super().__init__()
        self.corpus_name = corpus_name
        self.vocab_name = vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if prediction_windows is None:
            prediction_windows = [1, 2, 3, 4]
        self.prediction_windows = prediction_windows
        
        # Load your existing vocabulary (created during pretraining)
        self.vocabulary = self._load_vocabulary()
        
        # Initialize exit prediction task
        self.exit_task = ExitPrediction(
            name="exit_prediction",
            max_length=512,
            prediction_windows=prediction_windows,
            **kwargs
        )
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _load_vocabulary(self):
        """Load the existing vocabulary from pretraining"""
        vocab_path = Path(f"data/processed/vocab/{self.vocab_name}/result.tsv")
        
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        # Load the vocabulary file directly (created during pretraining)
        vocab_df = pd.read_csv(vocab_path, sep='\t')
        
        # Create a simple vocabulary wrapper that provides the interface we need
        class VocabularyWrapper:
            def __init__(self, vocab_df):
                self.vocab_df = vocab_df
                self.token2index = dict(zip(vocab_df.TOKEN, vocab_df.ID))
                self.index2token = dict(zip(vocab_df.ID, vocab_df.TOKEN))
            
            def vocab(self):
                return self.vocab_df
            
            def get_pad_token_id(self):
                return self.token2index.get('[PAD]', 0)
            
            def get_cls_token_id(self):
                return self.token2index.get('[CLS]', 1)
            
            def get_sep_token_id(self):
                return self.token2index.get('[SEP]', 2)
            
            def size(self):
                return len(self.vocab_df)
        
        vocabulary = VocabularyWrapper(vocab_df)
        log.info(f"Loaded existing vocabulary with {vocabulary.size()} tokens from pretraining")
        return vocabulary
    
    def _load_corpus_sentences(self, split: str):
        """Load your existing processed corpus sentences - FIXED VERSION"""
        corpus_path = Path(f"data/processed/corpus/{self.corpus_name}/sentences/{split}/sentences.parquet")
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        sentences_df = pd.read_parquet(corpus_path)
        
        # CRITICAL FIX: Set COMPANY_ID as index for groupby to work
        sentences_df = sentences_df.set_index('COMPANY_ID')
        
        log.info(f"Loaded {len(sentences_df)} sentences for {split} split")
        
        return sentences_df
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for exit prediction"""
        # Register exit task with this datamodule
        self.exit_task.datamodule = self
        
        if stage == "fit" or stage is None:
            # Load train and validation data
            train_sentences = self._load_corpus_sentences("train")
            val_sentences = self._load_corpus_sentences("val")
            
            self.train_dataset = ExitDataset(
                train_sentences, self.exit_task, "train"
            )
            self.val_dataset = ExitDataset(
                val_sentences, self.exit_task, "val"
            )
        
        if stage == "test" or stage is None:
            # Load test data
            test_sentences = self._load_corpus_sentences("test")
            
            self.test_dataset = ExitDataset(
                test_sentences, self.exit_task, "test"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
    
    def get_class_distribution(self):
        """Get class distribution for exit prediction"""
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            self.setup('fit')
        
        stats = {}
        for split, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            if dataset is not None:
                labels = [example['exit_label'] for example in dataset.examples]
                no_exit_count = sum(1 for label in labels if label == 0)
                exit_count = sum(1 for label in labels if label == 1)
                total = len(labels)
                
                stats[split] = {
                    'total': total,
                    'no_exit': no_exit_count,
                    'exit': exit_count,
                    'no_exit_pct': (no_exit_count / total) * 100 if total > 0 else 0,
                    'exit_pct': (exit_count / total) * 100 if total > 0 else 0
                }
        
        return stats
    
    def get_vocab_size(self):
        """Get vocabulary size from your existing vocabulary"""
        return self.vocabulary.vocab().shape[0]
    
    def get_pad_token_id(self):
        """Get pad token ID from your existing vocabulary"""
        return self.vocabulary.get_pad_token_id()
    
    def get_cls_token_id(self):
        """Get CLS token ID from your existing vocabulary"""
        return self.vocabulary.get_cls_token_id()
    
    def get_sep_token_id(self):
        """Get SEP token ID from your existing vocabulary"""
        return self.vocabulary.get_sep_token_id()
