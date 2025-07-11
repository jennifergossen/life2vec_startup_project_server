#!/usr/bin/env python3

"""
Step 4: Custom DataModule for Large Startup Dataset (FIXED VERSION - 4D INPUT)

This creates a life2vec-compatible datamodule that:
- Uses existing corpus files (from step 3.1) 
- Uses existing vocabulary (from step 3.2)
- Handles large datasets without PyArrow overflow
- Produces CORRECT 4D INPUT for step 5 training
- FIXES: Creates proper [4, sequence_length] input format
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
    """Custom dataset that processes corpus data for life2vec training with 4D input"""
    
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
        """Convert a single sentence to life2vec 4D format"""
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
            
            # Get actual sequence length
            seq_len = len(sequence)
            
            # Pad sequence to max_length
            sequence = sequence + [self.pad_id] * (self.max_length - seq_len)
            
            # Create padding mask (True for real tokens, False for padding)
            padding_mask = [True] * seq_len + [False] * (self.max_length - seq_len)
            
            # ========== LIFE2VEC 4D INPUT FORMAT ==========
            # Create 4D input: [4, max_length] where dimensions are:
            # 0: token IDs
            # 1: absolute position (days since reference)
            # 2: age values 
            # 3: segment IDs
            
            input_4d = np.zeros((4, self.max_length), dtype=np.int64)
            
            # Dimension 0: Token IDs
            input_4d[0, :] = sequence
            
            # Dimension 1: Absolute position (sequential for now)
            # In real life2vec, this would be days since reference date
            abs_positions = list(range(seq_len)) + [0] * (self.max_length - seq_len)
            input_4d[1, :] = abs_positions
            
            # Dimension 2: Age values (dummy - in real life2vec, this is person's age)
            # For startup data, we can use company age or dummy values
            age_values = [30] * seq_len + [0] * (self.max_length - seq_len)  # dummy age 30
            input_4d[2, :] = age_values
            
            # Dimension 3: Segment IDs (alternating pattern)
            # Life2vec uses this to distinguish different types of events
            segment_values = [(i % 4) for i in range(seq_len)] + [0] * (self.max_length - seq_len)
            input_4d[3, :] = segment_values
            
            # Apply MLM masking to the token dimension only
            masked_tokens, target_tokens, target_positions = self.apply_mlm_masking(
                input_4d[0].copy(), padding_mask
            )
            
            # Update the token dimension with masked tokens
            input_4d[0, :] = masked_tokens
            
            # Create life2vec-style output
            return {
                'sequence_id': torch.tensor(idx, dtype=torch.long),
                'input_ids': torch.tensor(input_4d, dtype=torch.long),  # Shape: [4, max_length]
                'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
                'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
                'target_pos': torch.tensor(target_positions, dtype=torch.long),
                'target_sop': torch.tensor(0, dtype=torch.long),
                'original_sequence': torch.tensor(sequence, dtype=torch.long)
            }
            
        except Exception as e:
            logging.error(f"Error processing index {idx}: {e}")
            # Return a minimal valid sample with 4D format
            max_targets = max(1, int(self.max_length * self.mask_ratio))
            
            # Create 4D input for error case
            input_4d = np.zeros((4, self.max_length), dtype=np.int64)
            input_4d[0, 0] = self.cls_id
            input_4d[0, 1] = self.sep_id
            
            return {
                'sequence_id': torch.tensor(idx, dtype=torch.long),
                'input_ids': torch.tensor(input_4d, dtype=torch.long),
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
    """Custom life2vec-compatible datamodule with 4D input"""
    
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
    """Test the custom datamodule with 4D format"""
    log = logging.getLogger(__name__)
    
    log.info("�� Testing custom datamodule with 4D input format...")
    
    try:
        # Setup datamodule
        datamodule.setup()
        
        # Test train loader
        log.info("📊 Testing train dataloader...")
        train_loader = datamodule.train_dataloader()
        
        for i, batch in enumerate(train_loader):
            log.info(f"   📦 Batch {i+1}:")
            log.info(f"      🔑 Keys: {list(batch.keys())}")
            log.info(f"      📏 Input shape: {batch['input_ids'].shape} (should be [batch, 4, sequence])")
            
            # Validate 4D format
            input_shape = batch['input_ids'].shape
            if len(input_shape) != 3:
                log.error(f"❌ Wrong input dimensions! Expected 3D [batch, 4, sequence], got {len(input_shape)}D")
                return False
            
            if input_shape[1] != 4:
                log.error(f"❌ Wrong feature dimensions! Expected 4 features, got {input_shape[1]}")
                return False
            
            log.info(f"      ✅ Correct 4D format: [batch={input_shape[0]}, features=4, sequence={input_shape[2]}]")
            log.info(f"      🎭 Target tokens shape: {batch['target_tokens'].shape}")
            log.info(f"      📍 Target positions shape: {batch['target_pos'].shape}")
            log.info(f"      ✅ Padding mask shape: {batch['padding_mask'].shape}")
            log.info(f"      🆔 Sequence IDs shape: {batch['sequence_id'].shape}")
            
            # Show actual values for each dimension
            input_4d = batch['input_ids'][0]  # First sample
            log.info(f"      📊 Sample values for first sequence:")
            log.info(f"         Tokens (dim 0): {input_4d[0][:10].tolist()}")
            log.info(f"         Abspos (dim 1): {input_4d[1][:10].tolist()}")
            log.info(f"         Age (dim 2): {input_4d[2][:10].tolist()}")
            log.info(f"         Segment (dim 3): {input_4d[3][:10].tolist()}")
            
            # Validate ranges
            tokens = input_4d[0]
            segments = input_4d[3]
            vocab_size = datamodule.vocabulary.size()
            
            log.info(f"      🔍 Value ranges:")
            log.info(f"         Tokens: {tokens.min().item()} - {tokens.max().item()} (vocab: {vocab_size})")
            log.info(f"         Segments: {segments.min().item()} - {segments.max().item()} (should be 0-3)")
            
            # Check bounds
            if tokens.max().item() >= vocab_size:
                log.error(f"❌ Token ID out of bounds: {tokens.max().item()} >= {vocab_size}")
                return False
            
            if segments.max().item() >= 4:
                log.error(f"❌ Segment ID out of bounds: {segments.max().item()} >= 4")
                return False
            
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
        
        log.info("✅ 4D FORMAT VALIDATION PASSED!")
        return True
        
    except Exception as e:
        log.error(f"❌ Error testing datamodule: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 4: Custom DataModule with 4D Input Format")
    
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
        log.info("🚀 CREATING LIFE2VEC 4D DATAMODULE")
        log.info("=" * 60)
        log.info("📋 Configuration:")
        log.info(f"   📚 Corpus: {args.corpus_name}")
        log.info(f"   📖 Vocabulary: {args.vocab_name}")
        log.info(f"   📏 Max length: {args.max_length}")
        log.info(f"   🎭 Mask ratio: {args.mask_ratio}")
        log.info(f"   🧠 Smart masking: {args.smart_masking}")
        log.info(f"   📦 Batch size: {args.batch_size}")
        log.info(f"   👷 Workers: {args.num_workers}")
        log.info(f"   🎯 Input format: [batch, 4, sequence] - 4D LIFE2VEC FORMAT")
        
        # Create datamodule
        log.info("🔧 Creating 4D datamodule...")
        datamodule = StartupDataModule(
            corpus_name=args.corpus_name,
            vocab_name=args.vocab_name,
            max_length=args.max_length,
            mask_ratio=args.mask_ratio,
            smart_masking=args.smart_masking,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        log.info("✅ 4D datamodule created")
        
        # Test the datamodule
        log.info("🧪 Testing 4D datamodule...")
        test_success = test_datamodule(datamodule, args.test_batches)
        
        if not test_success:
            log.error("❌ 4D datamodule testing failed")
            return 1
        
        total_time = time.time() - start_time
        
        log.info("�� SUCCESS!")
        log.info(f"✅ 4D Life2vec datamodule ready")
        log.info(f"📊 Input format: [batch, 4, {args.max_length}]")
        log.info(f"📖 Vocabulary size: {datamodule.vocabulary.size():,}")
        log.info(f"📦 Batch size: {args.batch_size}")
        log.info(f"🔢 Dataset sizes:")
        log.info(f"   📚 Train: {len(datamodule.train_dataset):,}")
        log.info(f"   📝 Val: {len(datamodule.val_dataset):,}")
        log.info(f"   🧪 Test: {len(datamodule.test_dataset):,}")
        log.info(f"⏱️  Total time: {total_time:.1f} seconds")
        log.info("")
        log.info("�� Ready for life2vec training!")
        log.info("💡 The 4D input format matches life2vec requirements")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Error: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
