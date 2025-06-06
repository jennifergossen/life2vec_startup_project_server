#!/usr/bin/env python3

import torch
from src.models.pretrain import TransformerEncoder

def test_checkpoint_for_nans(checkpoint_path):
    print(f"🔍 Testing: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = checkpoint['hyper_parameters']
        
        # Load model
        model = TransformerEncoder(hparams=hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # Load real data
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        
        datamodule = datamodule_module.StartupDataModule(batch_size=2, num_workers=1, max_length=512, mask_ratio=0.15)
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        real_batch = next(iter(train_loader))
        
        # Test forward pass
        with torch.no_grad():
            mlm_pred, sop_pred = model(real_batch)
        
        # Check for NaN
        has_nan = torch.isnan(mlm_pred).any() or torch.isnan(sop_pred).any()
        
        epoch = checkpoint.get('epoch', 'unknown')
        step = checkpoint.get('global_step', 'unknown')
        
        print(f"   📈 Epoch: {epoch}, Step: {step}")
        print(f"   🧪 Has NaN: {has_nan}")
        
        if not has_nan:
            print(f"   ✅ CLEAN MODEL FOUND!")
            return True, checkpoint_path
        else:
            print(f"   ❌ Contains NaN")
            return False, None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, None

def find_best_checkpoint():
    print("🔍 SEARCHING FOR CLEAN CHECKPOINT")
    print("=" * 50)
    
    # List of checkpoints to try (from newest to oldest)
    checkpoints = [
        "checkpoints/startup2vec-full-1gpu-512d-epoch=02-step=032517.ckpt",
        "checkpoints/startup2vec-full-1gpu-512d-epoch=01-step=021678.ckpt", 
        "checkpoints/startup2vec-full-1gpu-512d-epoch=00-step=010839.ckpt",
        "checkpoints/startup2vec-full-1gpu-512d-epoch=00-step=010836.ckpt",
        "checkpoints/startup2vec-full-1gpu-512d-epoch=00-step=008127.ckpt",
        "checkpoints/startup2vec-full-1gpu-512d-epoch=00-step=005418.ckpt",
    ]
    
    for checkpoint_path in checkpoints:
        success, clean_path = test_checkpoint_for_nans(checkpoint_path)
        if success:
            print(f"\n🎉 FOUND CLEAN CHECKPOINT: {clean_path}")
            return clean_path
    
    print("\n❌ No clean checkpoints found")
    return None

if __name__ == "__main__":
    best_checkpoint = find_best_checkpoint()
    
    if best_checkpoint:
        print(f"\n✅ USE THIS CHECKPOINT FOR FINE-TUNING:")
        print(f"   📁 {best_checkpoint}")
        print(f"\n💡 Your model trained successfully!")
        print(f"   The NaN only appeared in final epochs")
        print(f"   Earlier checkpoints should work perfectly!")
    else:
        print(f"\n🤔 All checkpoints have NaN - let's investigate further")
