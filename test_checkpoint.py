#!/usr/bin/env python3

import torch
from src.models.pretrain import TransformerEncoder

def test_checkpoint(checkpoint_path):
    print(f"🔍 Testing checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully!")
        
        # Check what's in the checkpoint
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            hparams = checkpoint['hparams']
        else:
            print("❌ No hyperparameters found in checkpoint")
            return False
            
        print(f"🏗️ Model architecture: {hparams.get('hidden_size', 'unknown')}d hidden")
        print(f"📊 Vocab size: {hparams.get('vocab_size', 'unknown')}")
        
        # Try to load the model
        model = TransformerEncoder(hparams=hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print("✅ Model loaded from checkpoint successfully!")
        
        # Quick forward pass test
        batch_size = 2
        seq_length = 512
        vocab_size = hparams['vocab_size']
        
        dummy_batch = {
            'input_ids': torch.randint(1, vocab_size-1, (batch_size, 4, seq_length)),
            'padding_mask': torch.ones(batch_size, seq_length),
            'target_tokens': torch.randint(1, vocab_size-1, (batch_size, 77)),
            'target_pos': torch.randint(1, seq_length-1, (batch_size, 77)),
            'target_sop': torch.randint(0, 3, (batch_size,))
        }
        
        with torch.no_grad():
            mlm_pred, sop_pred = model(dummy_batch)
            
        print("✅ Forward pass successful!")
        print(f"   MLM predictions: {mlm_pred.shape}")
        print(f"   SOP predictions: {sop_pred.shape}")
        
        # Check for NaN
        if torch.isnan(mlm_pred).any() or torch.isnan(sop_pred).any():
            print("❌ Model outputs contain NaN!")
            return False
        else:
            print("✅ Model outputs are clean!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing checkpoint: {e}")
        return False

if __name__ == "__main__":
    # Test the latest checkpoint
    success = test_checkpoint("checkpoints/last-v2.ckpt")
    
    if success:
        print("\n🎉 CHECKPOINT IS WORKING!")
        print("💡 Your model is ready for fine-tuning!")
    else:
        print("\n⚠️ Checkpoint has issues, trying another...")
        # Try the explicit epoch checkpoint
        success = test_checkpoint("checkpoints/startup2vec-full-1gpu-512d-epoch=02-step=032517.ckpt")
        
        if success:
            print("\n🎉 ALTERNATIVE CHECKPOINT WORKS!")
        else:
            print("\n❌ Need to investigate further")
