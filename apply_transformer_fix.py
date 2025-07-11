# apply_transformer_fix.py
"""
Apply the transformer.py fix to handle embedding tuple output properly
"""

import os
import shutil
from datetime import datetime

def apply_transformer_fix():
    """Apply the transformer fix"""
    
    print("�� APPLYING TRANSFORMER.PY FIX")
    print("=" * 40)
    
    transformer_file = "transformer/transformer.py"
    
    if not os.path.exists(transformer_file):
        print(f"❌ Transformer file not found: {transformer_file}")
        return False
    
    # Create backup
    backup_file = f"{transformer_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(transformer_file, backup_file)
    print(f"💾 Backup created: {backup_file}")
    
    # The fixed content
    fixed_content = '''import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from src.transformer.embeddings import Embeddings
from src.transformer.transformer_utils import ScaleNorm, l2_norm, Center, Swish, gelu, gelu_new, swish
from src.transformer.modules import EncoderLayer
import logging

log = logging.getLogger(__name__)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
    "tanh": torch.tanh,
}


class Transformer(nn.Module):
    def __init__(self, hparams):
        """Encoder part of the life2vec model"""
        super(Transformer, self).__init__()

        self.hparams = hparams
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams)
        # Initialize the Encoder Blocks
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )

    def forward(self, x, padding_mask):
        """Forward pass - FIXED to handle embedding tuple properly"""
        # Handle both tuple and tensor returns from embedding
        emb_output = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        
        # Extract embeddings (first element if tuple, otherwise use directly)
        if isinstance(emb_output, tuple):
            x, _ = emb_output
        else:
            x = emb_output
            
        for layer in self.encoders:
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
        return x

    def forward_finetuning(self, x, padding_mask=None):
        """Forward pass for finetuning - FIXED to handle embedding tuple properly"""
        # Handle both tuple and tensor returns from embedding
        emb_output = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        
        # Extract embeddings (first element if tuple, otherwise use directly)
        if isinstance(emb_output, tuple):
            x, _ = emb_output
        else:
            x = emb_output

        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)

        return x

    def get_sequence_embedding(self, x):
        """Get only embeddings - FIXED to handle embedding tuple properly"""
        # Handle both tuple and tensor returns from embedding
        emb_output = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        
        # Return the full embedding output (might be tuple or tensor)
        return emb_output

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            log.info("Redrawing projections for the encoder layers (manually)")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()


class MaskedLanguageModel(nn.Module):
    """Masked Language Model (MLM) Decoder (for pretraining)"""

    def __init__(self, hparams, embedding, act: str = "tanh"):
        super(MaskedLanguageModel, self).__init__()
        self.hparams = hparams
        self.act = ACT2FN[act]
        self.dropout = nn.Dropout(p=self.hparams.emb_dropout)

        self.V = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.g = nn.Parameter(torch.tensor([hparams.hidden_size**0.5]))
        self.out = nn.Linear(
            self.hparams.hidden_size,
            self.hparams.vocab_size,
            bias=False
        )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            ignore_index = torch.LongTensor([0, 4, 5, 6, 7, 8])
            log.info("(MLM Decoder) centering: true normalisation: %s" %
                     hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(
                ignore_index=ignore_index, norm=hparams.norm_output_emb))

    def batched_index_select(self, x, dim, indx):
        """Gather the embeddings of tokens that we should make prediction on"""
        indx_ = indx.unsqueeze(2).expand(
            indx.size(0), indx.size(1), x.size(-1))
        return x.gather(dim, indx_)

    def forward(self, logits, batch):
        indx = batch["target_pos"].long()
        logits = self.dropout(self.batched_index_select(logits, 1, indx))
        logits = self.dropout(l2_norm(self.act(self.V(logits))))
        return self.g * self.out(logits)


class SOP_Decoder(nn.Module):
    """Sequence Order Decoder (for pretraining)"""

    def __init__(self, hparams):
        super(SOP_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = hparams.cls_num_targs
        p = hparams.dc_dropout

        self.in_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act = ACT2FN["swish"]
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""
        x = self.dropout(self.norm(self.act(self.in_layer(x))))
        return self.out_layer(x)
'''
    
    # Write the fixed content
    with open(transformer_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"✅ Fixed transformer.py written")
    
    return True

def test_fixed_transformer():
    """Test if the fixed transformer works"""
    
    print("\n🧪 TESTING FIXED TRANSFORMER")
    print("=" * 35)
    
    try:
        # Import and test the fixed transformer
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from models.survival_model import StartupSurvivalModel
        
        checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
        
        # Force reload the transformer module
        import importlib
        if 'transformer.transformer' in sys.modules:
            importlib.reload(sys.modules['transformer.transformer'])
        
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path="./startup2vec_startup2vec-full-1gpu-512d_final.pt",
            map_location='cpu'
        )
        model.eval()
        
        print("✅ Model loaded successfully")
        
        # Test with simple input
        test_input = torch.zeros(1, 512, dtype=torch.long)
        test_mask = torch.ones(1, 512)
        
        print("🧪 Testing forward pass...")
        
        with torch.no_grad():
            outputs = model.forward(input_ids=test_input, padding_mask=test_mask)
            print("✅ Forward pass successful!")
            print(f"📊 Output keys: {outputs.keys()}")
            
            if 'survival_logits' in outputs:
                survival_logits = outputs['survival_logits']
                print(f"📊 Survival logits shape: {survival_logits.shape}")
                
                survival_probs = torch.softmax(survival_logits, dim=1)
                print(f"📊 Survival probabilities: {survival_probs}")
                
        # Test with real data
        print("\n🔬 Testing with real data...")
        
        from dataloaders.survival_datamodule import SurvivalDataModule
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab", 
            batch_size=2,
            num_workers=0
        )
        datamodule.setup()
        
        val_loader = datamodule.val_dataloader()
        real_batch = next(iter(val_loader))
        
        real_input_ids = real_batch['input_ids']
        real_padding_mask = real_batch['padding_mask']
        
        # Handle multi-window format
        if len(real_input_ids.shape) == 3:
            real_input_ids = real_input_ids[:, 0, :]
        
        with torch.no_grad():
            outputs = model.forward(input_ids=real_input_ids, padding_mask=real_padding_mask)
            print("✅ Real data forward pass successful!")
            
            survival_logits = outputs['survival_logits']
            survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
            print(f"📊 Sample survival probabilities: {survival_probs.tolist()}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The transformer fix is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to apply and test the fix"""
    
    print("🚀 TRANSFORMER.PY FIX APPLICATION")
    print("=" * 50)
    print("This will fix the embedding tuple handling in transformer.py")
    print()
    
    # Apply the fix
    success = apply_transformer_fix()
    
    if not success:
        print("❌ Failed to apply fix")
        return 1
    
    # Test the fix
    test_success = test_fixed_transformer()
    
    if test_success:
        print("\n✅ SUCCESS!")
        print("=" * 20)
        print("🔧 Transformer.py has been fixed")
        print("🧪 All tests pass")
        print("🚀 Ready for interpretability analysis")
        print()
        print("🎯 Next steps:")
        print("1. Run: python extract_startup2vec_data.py")
        print("2. Run: python run_interpretability_analysis.py")
        
        return 0
    else:
        print("\n❌ Fix applied but tests failed")
        print("Check the error messages above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
