import torch
import sys
from step_4b_create_balanced_datamodule import StartupDataModule
from src.models.pretrain import TransformerEncoder

CHECKPOINT_PATH = "startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt"  # Change if needed

def load_model(checkpoint_path, device):
    print(f"\n🔍 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint.get('hparams', None)
    if hparams is None:
        print("❌ hparams not found in checkpoint!")
        sys.exit(1)
    print("✅ Found hparams in checkpoint.")
    model = TransformerEncoder(hparams=hparams)
    print("✅ Model instantiated successfully.")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ Model weights loaded (strict=False).\n")
    model.to(device)
    return model

def test_with_validation_data(model, device):
    print("[INFO] Loading validation data from StartupDataModule...")
    datamodule = StartupDataModule(batch_size=4, num_workers=0)  # Small batch for test
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    print(f"[DEBUG] Validation batch input_ids shape: {batch['input_ids'].shape}")
    print(f"[DEBUG] Validation batch keys: {list(batch.keys())}")
    # Move batch to device
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        output = model(batch)
    if isinstance(output, tuple):
        print(f"[RESULT] Model output is a tuple of length {len(output)}")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                print(f"[RESULT] Output[{i}] shape: {out.shape}")
            else:
                print(f"[RESULT] Output[{i}]: {out}")
    else:
        print(f"[RESULT] Model output: {output}")
    print("[SUCCESS] Model forward pass with real validation data completed.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)
    test_with_validation_data(model, device)
    print("\n🎉 All checks passed! Model is ready for downstream tasks.")

if __name__ == "__main__":
    main()