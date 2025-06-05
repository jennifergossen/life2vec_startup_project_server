#!/usr/bin/env python3

"""
Step 4: Custom DataModule for Large Startup Dataset (FIXED VERSION)

This creates a life2vec-compatible datamodule that:
- Uses existing corpus files (from step 3.1) 
- Uses existing vocabulary (from step 3.2)
- Handles large datasets without PyArrow overflow
- Produces correct output for step 5 training
- FIXES: sequence_id tensor creation and error handling
"""

import argparse
import logging
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Any
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('custom_datamodule.log')
        ]
    )

class StartupVocabulary:
    """Load vocabulary from step 3.2"""
    def __init__(self, vocab_name="startup_vocab"):
        vocab_path = Path(f"data/processed/vocab/{vocab_name}/result.tsv")
        if not vocab_path.exists():
            raise FileNotFoundError(f"❌ Vocabulary not found: {vocab_path}")
        
        self.vocab_df = pd.read_csv(vocab_path, sep='\t')
        self._token2index = dict(zip(self.vocab_df.TOKEN, self.vocab_df.ID))
        self._index2token = dict(zip(self.vocab_df.ID, self.vocab_df.TOKEN))
        
        # Life2vec expects these special tokens
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]" 
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        
    @property
    def token2index(self):
        return self._token2index
        
    @property 
    def index2token(self):
        return self._index2token
        
    def size(self):
        return len(self._token2index)

class StartupDataset(Dataset):
    """Custom dataset that processes corpus data for life2vec training"""
    
    def __init__(self, corpus_file: Path, vocabulary: StartupVocabulary, 
                 max_length=512, mask_ratio=0.15, smart_masking=False):
        self.vocabulary = vocabulary
        self.max_length = max_length  
        self.mask_ratio = mask_ratio
        self.smart_masking = smart_masking
        
        # Load corpus data with error handling
        logging.info(f"Loading dataset from {corpus_file}")
        try:
            self.data = pd.read_parquet(corpus_file)
            # Reset index to ensure clean integer indices
            self.data = self.data.reset_index(drop=True)
            logging.info(f"Loaded {len(self.data)} sentences")
        except Exception as e:
            logging.error(f"Failed to load {corpus_file}: {e}")
            raise
        
        # Cache token IDs
        self.pad_id = vocabulary.token2index.get(vocabulary.pad_token, 0)
        self.cls_id = vocabulary.token2index.get(vocabulary.cls_token, 1) 
        self.sep_id = vocabulary.token2index.get(vocabulary.sep_token, 2)
        self.mask_id = vocabulary.token2index.get(vocabulary.mask_token, 3)
        self.unk_id = vocabulary.token2index.get(vocabulary.unk_token, 9)
        
        logging.info(f"Token IDs - PAD: {self.pad_id}, CLS: {self.cls_id}, SEP: {self.sep_id}, MASK: {self.mask_id}, UNK: {self.unk_id}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Convert a single sentence to life2vec format"""
        try:
            row = self.data.iloc[idx]
            
            # Get sentence tokens - handle empty sentences
            sentence_text = row.get('SENTENCE', '')
            if not sentence_text or pd.isna(sentence_text):
                sentence_tokens = []
            else:
                sentence_tokens = str(sentence_text).split()
            
            # Convert to token IDs
            token_ids = [self.vocabulary.token2index.get(token, self.unk_id) 
                         for token in sentence_tokens]
            
            # Create sequence: [CLS] + tokens + [SEP]
            sequence = [self.cls_id] + token_ids + [self.sep_id]
            
            # Truncate if too long (keep [CLS] and [SEP])
            if len(sequence) > self.max_length:
                sequence = [self.cls_id] + token_ids[:self.max_length-2] + [self.sep_id]
            
            # Create padding mask (True for real tokens, False for padding)
            seq_len = len(sequence)
            padding_mask = [True] * seq_len + [False] * (self.max_length - seq_len)
            
            # Pad sequence
            sequence = sequence + [self.pad_id] * (self.max_length - seq_len)
            
            # Apply MLM masking
            masked_sequence, target_tokens, target_positions = self.apply_mlm_masking(
                sequence, padding_mask
            )
            
            # Create life2vec-style output - USE IDX AS SEQUENCE_ID
            return {
                'sequence_id': torch.tensor(idx, dtype=torch.long),
                'input_ids': torch.tensor(masked_sequence, dtype=torch.long),
                'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
                'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
                'target_pos': torch.tensor(target_positions, dtype=torch.long),
                'target_sop': torch.tensor(0, dtype=torch.long),
                'original_sequence': torch.tensor(sequence, dtype=torch.long)
            }
            
        except Exception as e:
            logging.error(f"Error processing index {idx}: {e}")
            # Return a minimal valid sample
            max_targets = max(1, int(self.max_length * self.mask_ratio))
            return {
                'sequence_id': torch.tensor(idx, dtype=torch.long),
                'input_ids': torch.tensor([self.cls_id, self.sep_id] + [self.pad_id] * (self.max_length-2), dtype=torch.long),
                'padding_mask': torch.tensor([True, True] + [False] * (self.max_length-2), dtype=torch.bool),
                'target_tokens': torch.tensor([0] * max_targets, dtype=torch.long),
                'target_pos': torch.tensor([self.max_length-1] * max_targets, dtype=torch.long),
                'target_sop': torch.tensor(0, dtype=torch.long),
                'original_sequence': torch.tensor([self.cls_id, self.sep_id] + [self.pad_id] * (self.max_length-2), dtype=torch.long)
            }
    
    def apply_mlm_masking(self, sequence, padding_mask):
        """Apply masked language model masking"""
        sequence = sequence.copy()
        seq_len = sum(padding_mask)
        
        # Don't mask special tokens ([CLS], [SEP], [PAD])
        maskable_positions = []
        for i in range(1, seq_len-1):  # Skip [CLS] and [SEP]
            if sequence[i] not in [self.cls_id, self.sep_id, self.pad_id]:
                maskable_positions.append(i)
        
        # Determine how many tokens to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_ratio))
        
        # Randomly select positions to mask
        if len(maskable_positions) > 0:
            mask_positions = random.sample(maskable_positions, 
                                         min(num_to_mask, len(maskable_positions)))
        else:
            mask_positions = []
        
        # Store original tokens and apply masking
        target_tokens = []
        target_positions = []
        
        for pos in mask_positions:
            target_tokens.append(sequence[pos])
            target_positions.append(pos)
            
            # MLM strategy: 80% [MASK], 10% random, 10% unchanged
            rand = random.random()
            if rand < 0.8:
                sequence[pos] = self.mask_id
            elif rand < 0.9:
                # Replace with random token (excluding special tokens)
                vocab_tokens = [t for t in self.vocabulary.token2index.values() 
                               if t not in [self.cls_id, self.sep_id, self.pad_id, self.mask_id, self.unk_id]]
                if vocab_tokens:
                    sequence[pos] = random.choice(vocab_tokens)
            # 10% keep unchanged
        
        # Pad target arrays to fixed size for batching
        max_targets = max(1, int(self.max_length * self.mask_ratio))
        target_tokens = target_tokens[:max_targets]
        target_positions = target_positions[:max_targets]
        
        # Pad with zeros
        while len(target_tokens) < max_targets:
            target_tokens.append(0)
            target_positions.append(self.max_length - 1)  # Point to last position
            
        return sequence, target_tokens, target_positions

class StartupDataModule(pl.LightningDataModule):
    """Custom life2vec-compatible datamodule"""
    
    def __init__(self, corpus_name="startup_corpus", vocab_name="startup_vocab",
                 max_length=512, mask_ratio=0.15, smart_masking=False,
                 batch_size=32, num_workers=4):
        super().__init__()
        self.corpus_name = corpus_name
        self.vocab_name = vocab_name
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.smart_masking = smart_masking
        self.batch_size = batch_size
        
        # Reduce workers for debugging if too many
        if num_workers > 8:
            logging.warning(f"Reducing num_workers from {num_workers} to 8 for stability")
            num_workers = 8
        self.num_workers = num_workers
        
        # Load vocabulary
        self.vocabulary = StartupVocabulary(vocab_name)
        
        # Corpus paths
        self.corpus_dir = Path(f"data/processed/corpus/{corpus_name}/sentences")
        
        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage=None):
        """Setup train/val/test datasets"""
        if stage == "fit" or stage is None:
            train_file = self.corpus_dir / "train" / "sentences.parquet"
            val_file = self.corpus_dir / "val" / "sentences.parquet"
            
            if not train_file.exists():
                raise FileNotFoundError(f"Training file not found: {train_file}")
            if not val_file.exists():
                raise FileNotFoundError(f"Validation file not found: {val_file}")
            
            self.train_dataset = StartupDataset(
                train_file, self.vocabulary, self.max_length, 
                self.mask_ratio, self.smart_masking
            )
            self.val_dataset = StartupDataset(
                val_file, self.vocabulary, self.max_length,
                self.mask_ratio, self.smart_masking  
            )
            
        if stage == "test" or stage is None:
            test_file = self.corpus_dir / "test" / "sentences.parquet"
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
                
            self.test_dataset = StartupDataset(
                test_file, self.vocabulary, self.max_length,
                self.mask_ratio, self.smart_masking
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True  # Avoid irregular batch sizes
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )

def test_datamodule(datamodule, test_batches=3):
    """Test the custom datamodule"""
    log = logging.getLogger(__name__)
    
    log.info("🧪 Testing custom datamodule...")
    
    try:
        # Setup datamodule
        datamodule.setup()
        
        # Test train loader
        log.info("📊 Testing train dataloader...")
        train_loader = datamodule.train_dataloader()
        
        for i, batch in enumerate(train_loader):
            log.info(f"   📦 Batch {i+1}:")
            log.info(f"      🔑 Keys: {list(batch.keys())}")
            log.info(f"      📏 Input shape: {batch['input_ids'].shape}")
            log.info(f"      🎭 Target tokens shape: {batch['target_tokens'].shape}")
            log.info(f"      📍 Target positions shape: {batch['target_pos'].shape}")
            log.info(f"      ✅ Padding mask shape: {batch['padding_mask'].shape}")
            log.info(f"      🆔 Sequence IDs shape: {batch['sequence_id'].shape}")
            
            # Show some actual values
            log.info(f"      📊 Sample input_ids: {batch['input_ids'][0][:20].tolist()}")
            log.info(f"      🎯 Sample targets: {batch['target_tokens'][0][:5].tolist()}")
            log.info(f"      📍 Sample positions: {batch['target_pos'][0][:5].tolist()}")
            log.info(f"      🆔 Sample sequence_ids: {batch['sequence_id'][:5].tolist()}")
            
            # Validate data types
            assert batch['input_ids'].dtype == torch.long, "input_ids should be long"
            assert batch['padding_mask'].dtype == torch.bool, "padding_mask should be bool"
            assert batch['target_tokens'].dtype == torch.long, "target_tokens should be long"
            assert batch['target_pos'].dtype == torch.long, "target_pos should be long"
            assert batch['sequence_id'].dtype == torch.long, "sequence_id should be long"
            
            if i >= test_batches - 1:
                break
        
        # Test other loaders exist
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        log.info("✅ All dataloaders created successfully")
        
        # Test one batch from each
        log.info("📊 Testing validation dataloader...")
        val_batch = next(iter(val_loader))
        log.info(f"   Val batch shape: {val_batch['input_ids'].shape}")
        
        log.info("📊 Testing test dataloader...")
        test_batch = next(iter(test_loader))
        log.info(f"   Test batch shape: {test_batch['input_ids'].shape}")
        
        # Basic stats
        log.info(f"📊 Dataset sizes:")
        log.info(f"   Train: {len(datamodule.train_dataset):,}")
        log.info(f"   Val: {len(datamodule.val_dataset):,}")
        log.info(f"   Test: {len(datamodule.test_dataset):,}")
        log.info(f"   Vocabulary: {datamodule.vocabulary.size():,}")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error testing datamodule: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 4: Custom DataModule for Large Startup Dataset")
    
    # Component names
    parser.add_argument("--corpus-name", type=str, default="startup_corpus",
                       help="Name of corpus from step 3.1")
    parser.add_argument("--vocab-name", type=str, default="startup_vocab", 
                       help="Name of vocabulary from step 3.2")
    
    # Model parameters
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                       help="Masking ratio for MLM")
    parser.add_argument("--smart-masking", action="store_true",
                       help="Use smart masking")
    
    # DataLoader parameters  
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Testing
    parser.add_argument("--test-batches", type=int, default=3,
                       help="Number of batches to test")
    
    # Control
    parser.add_argument("--run", action="store_true", required=True,
                       help="Actually run the datamodule creation")
    
    args = parser.parse_args()
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        log.info("🚀 CREATING CUSTOM LIFE2VEC DATAMODULE (FIXED VERSION)")
        log.info("=" * 60)
        log.info("📋 Configuration:")
        log.info(f"   📚 Corpus: {args.corpus_name}")
        log.info(f"   📖 Vocabulary: {args.vocab_name}")
        log.info(f"   📏 Max length: {args.max_length}")
        log.info(f"   🎭 Mask ratio: {args.mask_ratio}")
        log.info(f"   🧠 Smart masking: {args.smart_masking}")
        log.info(f"   📦 Batch size: {args.batch_size}")
        log.info(f"   👷 Workers: {args.num_workers}")
        
        # Create datamodule
        log.info("🔧 Creating custom datamodule...")
        datamodule = StartupDataModule(
            corpus_name=args.corpus_name,
            vocab_name=args.vocab_name,
            max_length=args.max_length,
            mask_ratio=args.mask_ratio,
            smart_masking=args.smart_masking,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        log.info("✅ Custom datamodule created")
        
        # Test the datamodule
        log.info("🧪 Testing datamodule...")
        test_success = test_datamodule(datamodule, args.test_batches)
        
        if not test_success:
            log.error("❌ Datamodule testing failed")
            return 1
        
        # Save datamodule for training
        output_dir = Path(f"data/processed/datamodules")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datamodule_path = output_dir / f"{args.corpus_name}_datamodule.pkl"
        
        # Save key info (not the whole object due to pickling issues)
        datamodule_config = {
            'corpus_name': args.corpus_name,
            'vocab_name': args.vocab_name,
            'max_length': args.max_length,
            'mask_ratio': args.mask_ratio,
            'smart_masking': args.smart_masking,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'vocab_size': datamodule.vocabulary.size(),
            'train_size': len(datamodule.train_dataset),
            'val_size': len(datamodule.val_dataset),
            'test_size': len(datamodule.test_dataset)
        }
        
        import pickle
        with open(datamodule_path, 'wb') as f:
            pickle.dump(datamodule_config, f)
        
        total_time = time.time() - start_time
        
        log.info("🎉 SUCCESS!")
        log.info(f"✅ Custom datamodule ready")
        log.info(f"📊 Vocabulary size: {datamodule.vocabulary.size():,}")
        log.info(f"📦 Batch size: {args.batch_size}")
        log.info(f"📏 Max length: {args.max_length}")
        log.info(f"🔢 Dataset sizes:")
        log.info(f"   📚 Train: {len(datamodule.train_dataset):,}")
        log.info(f"   📝 Val: {len(datamodule.val_dataset):,}")
        log.info(f"   🧪 Test: {len(datamodule.test_dataset):,}")
        log.info(f"💾 Config saved to: {datamodule_path}")
        log.info(f"⏱️  Total time: {total_time:.1f} seconds")
        log.info("")
        log.info("🚀 Ready for model training!")
        log.info("💡 Use this datamodule in your training script")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Error: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
