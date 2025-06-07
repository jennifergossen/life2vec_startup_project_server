# src/dataloaders/survival_datamodule.py
"""
Survival Prediction DataModule
Extends existing life2vec infrastructure for survival prediction
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .datamodule import L2VDataModule
from .tasks.survival_prediction import SurvivalPrediction
from .vocabulary import CorpusVocabulary

log = logging.getLogger(__name__)

class SurvivalDataset(Dataset):
    """
    Dataset for survival prediction that creates multiple temporal windows per company
    """
    
    def __init__(
        self,
        corpus_sentences: pd.DataFrame,
        task: SurvivalPrediction,
        split: str = "train"
    ):
        self.corpus_sentences = corpus_sentences
        self.task = task
        self.split = split
        
        # Get survival labels
        self.survival_labels = task.get_survival_labels()
        
        # Create dataset examples (multiple windows per company)
        self.examples = self._create_examples()
        
        log.info(f"Created {len(self.examples)} examples for {split} split")
    
    def _create_examples(self):
        """
        Create multiple training examples per company (one per prediction window)
        """
        examples = []
        
        # Group sentences by company
        grouped = self.corpus_sentences.groupby(level=0)  # Group by index (company_id/uuid)
        
        for company_id, company_sentences in grouped:
            # Skip companies without survival labels
            if company_id not in self.survival_labels.index:
                continue
            
            company_info = self.survival_labels.loc[company_id]
            survival_label = company_info['survival_label']
            
            # Skip invalid labels
            if pd.isna(survival_label) or survival_label < 0:
                continue
            
            # Create one example per prediction window
            for window_years in self.task.prediction_windows:
                examples.append({
                    'company_id': company_id,
                    'window_years': window_years,
                    'survival_label': int(survival_label),
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
            self.survival_labels
        )
        
        # Encode document
        encoded = self.task.encode_document(document)
        
        # Override survival label and window info
        encoded.survival_label = np.array([example['survival_label']])
        encoded.prediction_window = np.array([example['window_years']])
        
        # Convert to tensors
        return {
            'sequence_id': torch.tensor(encoded.sequence_id, dtype=torch.long),
            'input_ids': torch.tensor(encoded.input_ids, dtype=torch.long),
            'padding_mask': torch.tensor(encoded.padding_mask, dtype=torch.bool),
            'survival_label': torch.tensor(encoded.survival_label, dtype=torch.long),
            'prediction_window': torch.tensor(encoded.prediction_window, dtype=torch.long),
            'company_founded_year': torch.tensor(encoded.company_founded_year, dtype=torch.long),
            'company_age_at_prediction': torch.tensor(encoded.company_age_at_prediction, dtype=torch.long)
        }

class SurvivalDataModule(pl.LightningDataModule):
    """
    DataModule for startup survival prediction
    Extends your existing corpus/vocabulary infrastructure
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
        
        # Initialize task
        self.task = SurvivalPrediction(
            name="survival_prediction",
            max_length=512,
            prediction_windows=prediction_windows,
            **kwargs
        )
        
        # Load vocabulary (reuse from pretraining)
        self.vocabulary = self._load_vocabulary()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _load_vocabulary(self):
        """Load vocabulary from pretraining"""
        vocab_path = Path(f"data/processed/vocab/{self.vocab_name}/result.tsv")
        
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
        
        # Create a simple vocabulary object
        class SimpleVocabulary:
            def __init__(self, vocab_df):
                self.vocab_df = vocab_df
                self.token2index = dict(zip(vocab_df.TOKEN, vocab_df.ID))
                self.index2token = dict(zip(vocab_df.ID, vocab_df.TOKEN))
            
            def size(self):
                return len(self.token2index)
        
        vocab_df = pd.read_csv(vocab_path, sep='\t')
        vocabulary = SimpleVocabulary(vocab_df)
        
        log.info(f"Loaded vocabulary with {vocabulary.size()} tokens")
        return vocabulary
    
    def _load_corpus_sentences(self, split: str):
        """Load corpus sentences for a split"""
        corpus_path = Path(f"data/processed/corpus/{self.corpus_name}/sentences/{split}/sentences.parquet")
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        sentences_df = pd.read_parquet(corpus_path)
        log.info(f"Loaded {len(sentences_df)} sentences for {split} split")
        
        return sentences_df
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        # Register task with this datamodule
        self.task.register(self)
        
        if stage == "fit" or stage is None:
            # Load train and validation data
            train_sentences = self._load_corpus_sentences("train")
            val_sentences = self._load_corpus_sentences("val")
            
            self.train_dataset = SurvivalDataset(
                train_sentences, self.task, "train"
            )
            self.val_dataset = SurvivalDataset(
                val_sentences, self.task, "val"
            )
        
        if stage == "test" or stage is None:
            # Load test data
            test_sentences = self._load_corpus_sentences("test")
            
            self.test_dataset = SurvivalDataset(
                test_sentences, self.task, "test"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_class_distribution(self):
        """Get class distribution for all splits"""
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            self.setup('fit')
        
        stats = {}
        for split, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            if dataset is not None:
                labels = [example['survival_label'] for example in dataset.examples]
                died_count = sum(1 for label in labels if label == 0)
                survived_count = sum(1 for label in labels if label == 1)
                total = len(labels)
                
                stats[split] = {
                    'total': total,
                    'died': died_count,
                    'survived': survived_count,
                    'died_pct': (died_count / total) * 100 if total > 0 else 0,
                    'survived_pct': (survived_count / total) * 100 if total > 0 else 0
                }
        
        return stats
