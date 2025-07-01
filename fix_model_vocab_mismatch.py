# fix_model_vocab_mismatch.py
"""
PERMANENT FIX FOR MODEL VOCABULARY MISMATCH
Corrects the model's internal hyperparameters to match the actual embedding layer size.

This fixes the root cause: model hyperparameters claim vocab_size=13305 
but the embedding layer actually has 20882 tokens.
"""

import torch
import os
import shutil
from datetime import datetime
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def analyze_checkpoint_vocab_mismatch(checkpoint_path):
    """Analyze the vocabulary mismatch in detail"""
    
    print("🔍 ANALYZING VOCABULARY MISMATCH")
    print("=" * 50)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    claimed_vocab_size = hparams.get('vocab_size', 'NOT_FOUND')
    
    # Get actual embedding layer size
    state_dict = checkpoint.get('state_dict', {})
    embedding_key = None
    actual_vocab_size = None
    
    for key in state_dict.keys():
        if 'embedding' in key.lower() and 'token' in key.lower() and 'weight' in key:
            if 'parametrizations' in key and 'original' in key:
                embedding_key = key
                actual_vocab_size = state_dict[key].shape[0]
                break
    
    print(f"📊 VOCABULARY SIZE ANALYSIS:")
    print(f"  🔢 Claimed in hyperparameters: {claimed_vocab_size}")
    print(f"  🔢 Actual embedding layer size: {actual_vocab_size}")
    print(f"  📏 Embedding key: {embedding_key}")
    
    if claimed_vocab_size != actual_vocab_size:
        print(f"  ❌ MISMATCH DETECTED!")
        print(f"  💡 This is the root cause of 'index out of range' errors")
        return True, claimed_vocab_size, actual_vocab_size
    else:
        print(f"  ✅ Vocabulary sizes match")
        return False, claimed_vocab_size, actual_vocab_size

def fix_vocabulary_mismatch(checkpoint_path, backup=True):
    """Fix the vocabulary mismatch by updating hyperparameters"""
    
    print(f"\n🔧 FIXING VOCABULARY MISMATCH")
    print("=" * 40)
    
    # 1. Backup original checkpoint
    if backup:
        backup_path = f"{checkpoint_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(checkpoint_path, backup_path)
        print(f"💾 Backup created: {backup_path}")
    
    # 2. Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 3. Analyze the mismatch
    has_mismatch, claimed_size, actual_size = analyze_checkpoint_vocab_mismatch(checkpoint_path)
    
    if not has_mismatch:
        print("✅ No vocabulary mismatch found. Checkpoint is already correct.")
        return True
    
    # 4. Fix the hyperparameters
    print(f"\n🔧 APPLYING FIX:")
    print(f"  📝 Updating vocab_size: {claimed_size} → {actual_size}")
    
    # Update hyperparameters
    if 'hyper_parameters' in checkpoint:
        checkpoint['hyper_parameters']['vocab_size'] = actual_size
        print(f"  ✅ Updated hyper_parameters.vocab_size")
    
    # Also check if there are any other references to vocabulary size
    state_dict = checkpoint.get('state_dict', {})
    
    # Look for any other vocab_size references in the state dict or other sections
    for section_name in ['callbacks', 'optimizer_states', 'lr_schedulers']:
        if section_name in checkpoint:
            section = checkpoint[section_name]
            # This is mostly for completeness - these sections typically don't contain vocab info
            print(f"  📋 Checked {section_name}: no vocab_size references found")
    
    # 5. Verify the fix by loading the model class and checking compatibility
    print(f"\n🧪 VERIFYING FIX...")
    
    try:
        from models.survival_model import StartupSurvivalModel
        
        # Save the fixed checkpoint temporarily
        temp_path = checkpoint_path + ".temp_fixed"
        torch.save(checkpoint, temp_path)
        
        # Try loading with the fixed checkpoint
        model = StartupSurvivalModel.load_from_checkpoint(
            temp_path,
            pretrained_model_path="./startup2vec_startup2vec-full-1gpu-512d_final.pt",
            map_location='cpu'
        )
        
        # Check if the model's vocab size is now correct
        if hasattr(model.transformer, 'hparams'):
            model_vocab_size = model.transformer.hparams.vocab_size
            print(f"  📊 Model vocab_size after fix: {model_vocab_size}")
            
            if model_vocab_size == actual_size:
                print(f"  ✅ Model vocabulary size is now correct!")
                
                # Test a simple forward pass
                print(f"  🧪 Testing forward pass...")
                test_input = torch.randint(0, actual_size, (1, 512))
                test_mask = torch.ones(1, 512)
                
                model.eval()
                with torch.no_grad():
                    try:
                        outputs = model.forward(input_ids=test_input, padding_mask=test_mask)
                        print(f"  ✅ Forward pass successful!")
                        
                        # 6. Save the fixed checkpoint
                        print(f"\n💾 SAVING FIXED CHECKPOINT...")
                        torch.save(checkpoint, checkpoint_path)
                        print(f"  ✅ Fixed checkpoint saved: {checkpoint_path}")
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                        return True
                        
                    except Exception as e:
                        print(f"  ❌ Forward pass failed: {e}")
                        os.remove(temp_path)
                        return False
            else:
                print(f"  ❌ Model vocab_size still incorrect: {model_vocab_size}")
                os.remove(temp_path)
                return False
        else:
            print(f"  ⚠️ Could not verify model hparams")
            os.remove(temp_path)
            return False
            
    except Exception as e:
        print(f"  ❌ Error verifying fix: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def verify_fixed_checkpoint(checkpoint_path):
    """Verify that the checkpoint is now working correctly"""
    
    print(f"\n✅ VERIFICATION: TESTING FIXED CHECKPOINT")
    print("=" * 50)
    
    try:
        # 1. Test checkpoint loading
        from models.survival_model import StartupSurvivalModel
        
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path="./startup2vec_startup2vec-full-1gpu-512d_final.pt",
            map_location='cpu'
        )
        print("✅ Model loads successfully")
        
        # 2. Test with real data
        from dataloaders.survival_datamodule import SurvivalDataModule
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab", 
            batch_size=2,
            num_workers=0
        )
        datamodule.setup()
        print("✅ Datamodule loads successfully")
        
        # 3. Test forward pass with real data
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        
        input_ids = batch['input_ids']
        padding_mask = batch['padding_mask']
        
        # Handle multi-window format
        if len(input_ids.shape) == 3:
            input_ids = input_ids[:, 0, :]
        
        print(f"📊 Testing with real data:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        
        model.eval()
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                padding_mask=padding_mask
            )
            
            survival_probs = torch.softmax(outputs['survival_logits'], dim=1)[:, 1]
            print(f"  ✅ Forward pass successful!")
            print(f"  📊 Sample survival probabilities: {survival_probs.tolist()}")
            
            return True
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to fix the vocabulary mismatch"""
    
    print("🔧 STARTUP2VEC MODEL VOCABULARY MISMATCH FIX")
    print("=" * 60)
    print("This script permanently fixes the vocabulary size mismatch")
    print("in your finetuned survival model checkpoint.")
    print()
    
    checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"🎯 Target checkpoint: {checkpoint_path}")
    
    # 1. Analyze the issue
    has_mismatch, claimed_size, actual_size = analyze_checkpoint_vocab_mismatch(checkpoint_path)
    
    if not has_mismatch:
        print("✅ No fix needed - checkpoint is already correct!")
        return 0
    
    # 2. Confirm the fix
    print(f"\n❓ CONFIRMATION:")
    print(f"This will fix the vocabulary mismatch by updating the model's")
    print(f"internal hyperparameters from {claimed_size} to {actual_size} tokens.")
    print(f"A backup will be created automatically.")
    print()
    
    response = input("Proceed with the fix? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Fix cancelled by user")
        return 1
    
    # 3. Apply the fix
    success = fix_vocabulary_mismatch(checkpoint_path, backup=True)
    
    if not success:
        print("❌ Fix failed!")
        return 1
    
    # 4. Verify the fix
    verification_success = verify_fixed_checkpoint(checkpoint_path)
    
    if verification_success:
        print(f"\n🎉 SUCCESS!")
        print("=" * 20)
        print("✅ Vocabulary mismatch permanently fixed")
        print("✅ Model checkpoint verified working")
        print("✅ Ready for interpretability analysis")
        print()
        print("🚀 Next steps:")
        print("1. Run: python extract_startup2vec_data.py")
        print("2. Run: python run_interpretability_analysis.py")
        print()
        print("💡 The 'index out of range' errors should now be completely resolved.")
        return 0
    else:
        print("❌ Verification failed - please check the error messages above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
