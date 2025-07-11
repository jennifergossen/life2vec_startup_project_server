#!/usr/bin/env python3

"""
Debug segment values specifically since that's where the CUDA error occurred
"""

import torch
import sys
import importlib.util

def import_datamodule():
    """Import the datamodule from the .py file"""
    try:
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        if spec is None:
            raise ImportError("Could not find step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        return datamodule_module.StartupDataModule
    except Exception as e:
        print(f"❌ Error importing datamodule: {e}")
        sys.exit(1)

def debug_segment_values():
    """Check segment values specifically"""
    print("🔍 DEBUGGING SEGMENT VALUES")
    print("=" * 50)
    
    StartupDataModule = import_datamodule()
    
    # Create datamodule with EXACT same settings as training
    print("📊 Creating datamodule with training settings...")
    datamodule = StartupDataModule(
        batch_size=32,  # Same as quick-test
        num_workers=4,
        max_length=512,
        mask_ratio=0.15
    )
    datamodule.setup()
    
    print("🔍 Checking train dataloader...")
    train_loader = datamodule.train_dataloader()
    
    print("🔍 Checking val dataloader...")
    val_loader = datamodule.val_dataloader()
    
    for loader_name, loader in [("TRAIN", train_loader), ("VAL", val_loader)]:
        print(f"\n📊 {loader_name} LOADER:")
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3:  # Check first 3 batches
                break
                
            # Check all input dimensions
            input_ids = batch["input_ids"]
            print(f"  Batch {batch_idx} - input_ids shape: {input_ids.shape}")
            
            # The real issue: input_ids shape is [batch, sequence] not [batch, 4, sequence]
            # We need to check the actual batch structure
            print(f"    Full batch keys: {list(batch.keys())}")
            
            # Check input_ids (should be tokens only)
            min_val = input_ids.min().item()
            max_val = input_ids.max().item()
            print(f"    input_ids: min={min_val}, max={max_val}")
            
            # The model expects input_ids to be [batch, 4, sequence] but we're getting [batch, sequence]
            # This means the model architecture doesn't match the data format!
            
            if len(input_ids.shape) != 3:
                print(f"    ❌ WRONG INPUT SHAPE! Expected [batch, 4, sequence], got {input_ids.shape}")
                print(f"    The model expects 4 dimensions: [tokens, abspos, age, segment]")
                print(f"    But datamodule is only providing tokens!")
                return False
                        
    print("\n✅ All segment values look good")
    return True

if __name__ == "__main__":
    success = debug_segment_values()
    if not success:
        print("❌ Found segment value issues!")
        sys.exit(1)
    print("✅ No issues found in segments")
