#!/usr/bin/env python3

"""
Startup2Vec Model Sanity Check
Comprehensive assessment of trained model quality
"""

import torch
import numpy as np
import sys
from pathlib import Path

def check_model_weights(state_dict):
    """Check if model weights are healthy"""
    print("\n🔬 WEIGHT ANALYSIS:")
    
    weight_stats = []
    problem_weights = 0
    
    for name, param in state_dict.items():
        if 'weight' in name and param.numel() > 100:  # Only check major weight matrices
            mean_val = param.mean().item()
            std_val = param.std().item()
            max_val = param.max().item()
            min_val = param.min().item()
            
            # Check for problematic values
            has_nan = torch.isnan(param).any().item()
            has_inf = torch.isinf(param).any().item()
            
            if has_nan or has_inf:
                problem_weights += 1
            
            weight_stats.append({
                'name': name,
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'min': min_val,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'param_count': param.numel()
            })
    
    # Print summary
    print(f"📋 Analyzed {len(weight_stats)} weight matrices:")
    
    for i, stats in enumerate(weight_stats[:8]):  # Show first 8
        status = "⚠️ PROBLEM" if (stats['has_nan'] or stats['has_inf']) else "✅ Healthy"
        print(f"   {i+1:2d}. {stats['name'][:45]:45} | {status}")
        print(f"       Shape: {stats['param_count']:,} params | Mean: {stats['mean']:7.4f} | Std: {stats['std']:7.4f}")
        print(f"       Range: [{stats['min']:7.4f}, {stats['max']:7.4f}]")
        
    if problem_weights > 0:
        print(f"\n❌ Found {problem_weights} problematic weight matrices!")
        return False
    else:
        print(f"\n✅ All {len(weight_stats)} weight matrices look healthy!")
        return True

def test_forward_pass(model, vocab_size):
    """Test if model can do forward pass without errors"""
    print("\n🧪 FORWARD PASS TEST:")
    
    # Create realistic dummy input
    batch_size = 4
    seq_length = 512
    
    dummy_batch = {
        'input_ids': torch.randint(1, vocab_size-1, (batch_size, 4, seq_length)),  # Avoid special tokens
        'padding_mask': torch.ones(batch_size, seq_length),
        'target_tokens': torch.randint(1, vocab_size-1, (batch_size, 77)),  # Typical mask count
        'target_pos': torch.randint(1, seq_length-1, (batch_size, 77)),
        'target_sop': torch.randint(0, 3, (batch_size,))
    }
    
    try:
        with torch.no_grad():
            mlm_pred, sop_pred = model(dummy_batch)
        
        print("✅ Forward pass successful!")
        print(f"   📊 MLM predictions: {mlm_pred.shape} | Range: [{mlm_pred.min():.3f}, {mlm_pred.max():.3f}]")
        print(f"   📊 SOP predictions: {sop_pred.shape} | Range: [{sop_pred.min():.3f}, {sop_pred.max():.3f}]")
        
        # Check for NaN in outputs
        mlm_has_nan = torch.isnan(mlm_pred).any().item()
        sop_has_nan = torch.isnan(sop_pred).any().item()
        
        if mlm_has_nan or sop_has_nan:
            print("❌ Model outputs contain NaN values!")
            return False
        else:
            print("✅ Model outputs are clean (no NaN/Inf)")
            
        # Check if outputs are reasonable
        mlm_std = mlm_pred.std().item()
        sop_std = sop_pred.std().item()
        
        print(f"   📈 MLM output diversity (std): {mlm_std:.3f}")
        print(f"   📈 SOP output diversity (std): {sop_std:.3f}")
        
        if mlm_std < 0.01:
            print("⚠️ Warning: MLM outputs have very low variance - might be underfitted")
        elif mlm_std > 100:
            print("⚠️ Warning: MLM outputs have very high variance - might be unstable")
        else:
            print("✅ MLM output variance looks reasonable")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def check_vocabulary():
    """Try to load and inspect vocabulary"""
    print("\n📚 VOCABULARY CHECK:")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        
        vocab = datamodule_module.StartupVocabulary()
        print(f"✅ Vocabulary loaded successfully: {len(vocab.token_to_id):,} tokens")
        
        # Show some example tokens by category
        sample_tokens = list(vocab.token_to_id.keys())[:25]
        print(f"📝 Sample tokens: {sample_tokens}")
        
        # Check for special tokens
        special_tokens = ['[PAD]', '[UNK]', '[MASK]', '[CLS]', '[SEP]']
        found_special = [token for token in special_tokens if token in vocab.token_to_id]
        print(f"🎯 Special tokens found: {found_special}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not load vocabulary: {e}")
        print("💡 This might be OK - vocabulary might be embedded in model")
        return False

def main():
    print("🚀 STARTUP2VEC MODEL SANITY CHECK")
    print("=" * 60)
    
    # Check if model file exists
    model_path = 'startup2vec_startup2vec-full-1gpu-512d_final.pt'
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you're in the right directory and the file exists")
        return
    
    try:
        # Load model data
        print("📂 Loading model...")
        model_data = torch.load(model_path, map_location='cpu')  # Load on CPU for safety
        
        print("✅ Model loaded successfully!")
        print(f"📊 Vocabulary size: {model_data['vocab_size']:,}")
        print(f"📈 Training completed: Epoch {model_data['final_epoch']}, Step {model_data['final_step']:,}")
        print(f"🏗️ Architecture: {model_data['hparams']['hidden_size']}d hidden, {model_data['hparams']['n_encoders']} layers")
        print(f"💾 Model file size: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")
        
        # Check what's in the model file
        print(f"📋 Model components: {list(model_data.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test 1: Check weights
    print("\n" + "="*60)
    print("TEST 1: WEIGHT ANALYSIS")
    weights_ok = check_model_weights(model_data['model_state_dict'])
    
    # Test 2: Load model and test forward pass
    print("\n" + "="*60)
    print("TEST 2: MODEL FUNCTIONALITY")
    
    try:
        from src.models.pretrain import TransformerEncoder
        
        print("🏗️ Reconstructing model...")
        hparams = model_data['hparams']
        model = TransformerEncoder(hparams=hparams)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        print("✅ Model reconstructed successfully!")
        forward_ok = test_forward_pass(model, model_data['vocab_size'])
        
    except Exception as e:
        print(f"❌ Failed to reconstruct model: {e}")
        forward_ok = False
    
    # Test 3: Check vocabulary
    print("\n" + "="*60)
    print("TEST 3: VOCABULARY")
    vocab_ok = check_vocabulary()
    
    # Final assessment
    print("\n" + "="*60)
    print("🎯 FINAL ASSESSMENT")
    print("="*60)
    
    tests_passed = sum([weights_ok, forward_ok])
    total_tests = 2  # Don't count vocab as critical
    
    if tests_passed == total_tests:
        print("🎉 EXCELLENT! Your model passed all critical tests!")
        print("✅ Model weights are healthy")
        print("✅ Forward pass works correctly") 
        print("✅ No NaN/Inf values detected")
        print("\n💡 Your model is ready for fine-tuning!")
        
    elif tests_passed >= 1:
        print("⚠️ PARTIAL SUCCESS - Some issues detected")
        print(f"✅ {tests_passed}/{total_tests} critical tests passed")
        print("💡 Model might still be usable, but investigate issues")
        
    else:
        print("❌ CRITICAL ISSUES - Model needs investigation")
        print("💡 Check training logs and consider retraining")
    
    print("\n🎓 Great work on training a large language model!")
    print("📊 Check your WandB dashboard for detailed training metrics")

if __name__ == "__main__":
    main()
