#!/usr/bin/env python3

import torch
from src.models.pretrain import TransformerEncoder

def debug_model_input():
    print("🔍 DEBUGGING MODEL INPUT REQUIREMENTS")
    
    # Load checkpoint
    checkpoint = torch.load("checkpoints/last-v2.ckpt", map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    # Load model
    model = TransformerEncoder(hparams=hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    print(f"📊 Expected vocab size: {hparams['vocab_size']}")
    print(f"🔧 Max length: {hparams.get('max_length', 'unknown')}")
    
    # Let's check what the actual data looks like by loading the datamodule
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        
        # Create datamodule to see actual data format
        datamodule = datamodule_module.StartupDataModule(
            batch_size=2,
            num_workers=1,
            max_length=512,
            mask_ratio=0.15
        )
        datamodule.setup()
        
        # Get a real batch
        train_loader = datamodule.train_dataloader()
        real_batch = next(iter(train_loader))
        
        print("✅ Got real data batch!")
        print(f"📋 Real batch keys: {list(real_batch.keys())}")
        
        for key, value in real_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"      min={value.min().item()}, max={value.max().item()}")
        
        # Try forward pass with real data
        print("\n🧪 Testing with REAL data:")
        with torch.no_grad():
            mlm_pred, sop_pred = model(real_batch)
            
        print("✅ Forward pass with real data SUCCESSFUL!")
        print(f"   MLM predictions: {mlm_pred.shape}")
        print(f"   SOP predictions: {sop_pred.shape}")
        
        # Check for NaN
        if torch.isnan(mlm_pred).any() or torch.isnan(sop_pred).any():
            print("❌ Model outputs contain NaN!")
        else:
            print("✅ Model outputs are clean!")
            print("🎉 YOUR MODEL IS WORKING PERFECTLY!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error with real data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_model_input()
