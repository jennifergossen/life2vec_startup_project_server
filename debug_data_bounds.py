#!/usr/bin/env python3

"""
Debug script to find out-of-bounds token IDs in the startup data
"""

import torch
import sys
import importlib.util
from pathlib import Path

def import_datamodule():
    """Import the datamodule from the .py file"""
    try:
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        if spec is None:
            raise ImportError("Could not find step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        return datamodule_module.StartupDataModule, datamodule_module.StartupVocabulary
    except Exception as e:
        print(f"❌ Error importing datamodule: {e}")
        sys.exit(1)

def debug_data_bounds():
    """Find out-of-bounds token IDs"""
    print("🔍 DEBUGGING DATA BOUNDS ISSUE")
    print("=" * 50)
    
    # Import classes
    StartupDataModule, StartupVocabulary = import_datamodule()
    
    # Create datamodule
    print("📊 Creating datamodule...")
    datamodule = StartupDataModule(
        batch_size=4,  # Small batch for debugging
        num_workers=1,
        max_length=512,
        mask_ratio=0.15
    )
    datamodule.setup()
    
    vocab_size = datamodule.vocabulary.size()
    print(f"📖 Vocabulary size: {vocab_size}")
    
    # Check train dataloader
    print("\n🔍 Checking train dataloader...")
    train_loader = datamodule.train_dataloader()
    
    issues_found = 0
    total_batches_checked = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:  # Check first 10 batches
            break
            
        total_batches_checked += 1
        
        # Get input tensors
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        
        # Check input_ids bounds
        min_id = input_ids.min().item()
        max_id = input_ids.max().item()
        
        print(f"Batch {batch_idx}:")
        print(f"  Shape: {input_ids.shape}")
        print(f"  Min ID: {min_id}")
        print(f"  Max ID: {max_id}")
        print(f"  Vocab size: {vocab_size}")
        
        # Check for out-of-bounds
        if min_id < 0:
            print(f"  ❌ NEGATIVE TOKEN IDs FOUND: {min_id}")
            negative_positions = (input_ids < 0).nonzero()
            print(f"     Positions: {negative_positions[:5].tolist()}")
            issues_found += 1
            
        if max_id >= vocab_size:
            print(f"  ❌ OUT-OF-BOUNDS TOKEN IDs FOUND: {max_id} >= {vocab_size}")
            oob_positions = (input_ids >= vocab_size).nonzero()
            print(f"     Positions: {oob_positions[:5].tolist()}")
            print(f"     Values: {input_ids[input_ids >= vocab_size][:5].tolist()}")
            issues_found += 1
            
        # Check padding mask
        mask_min = padding_mask.min().item()
        mask_max = padding_mask.max().item()
        if mask_min < 0 or mask_max > 1:
            print(f"  ❌ INVALID PADDING MASK: min={mask_min}, max={mask_max}")
            issues_found += 1
            
        # Check for NaN/inf
        if torch.isnan(input_ids).any():
            print(f"  ❌ NaN VALUES FOUND in input_ids")
            issues_found += 1
            
        if torch.isinf(input_ids).any():
            print(f"  ❌ INF VALUES FOUND in input_ids")
            issues_found += 1
            
        if issues_found == 0:
            print(f"  ✅ Batch {batch_idx} OK")
        else:
            print(f"  🔴 Batch {batch_idx} has {issues_found} issues")
            
            # Show some actual problematic values
            print(f"  Sample tokens: {input_ids[0, :10].tolist()}")
            
    print(f"\n📊 SUMMARY:")
    print(f"   Batches checked: {total_batches_checked}")
    print(f"   Issues found: {issues_found}")
    print(f"   Vocabulary size: {vocab_size}")
    
    if issues_found > 0:
        print(f"\n🔧 RECOMMENDED FIXES:")
        print(f"   1. Check vocabulary creation in step_4_create_datamodule.py")
        print(f"   2. Ensure all tokens are mapped correctly")
        print(f"   3. Add bounds checking in tokenization")
        return False
    else:
        print(f"\n✅ NO ISSUES FOUND - this shouldn't happen if CUDA error occurred")
        return True

if __name__ == "__main__":
    success = debug_data_bounds()
    sys.exit(0 if success else 1)
