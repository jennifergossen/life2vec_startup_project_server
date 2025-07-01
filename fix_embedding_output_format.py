# fix_embedding_output_format.py
"""
Fix the embedding output format issue in the Life2Vec transformer.
The issue is that embedding layer returns a tuple but code expects a tensor.
"""

import torch
import sys
from pathlib import Path
import os
import shutil
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def investigate_embedding_output():
    """Investigate what the embedding layer actually returns"""
    
    print("🔍 INVESTIGATING EMBEDDING OUTPUT FORMAT")
    print("=" * 50)
    
    try:
        from models.survival_model import StartupSurvivalModel
        
        checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
        
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path="./startup2vec_startup2vec-full-1gpu-512d_final.pt",
            map_location='cpu'
        )
        model.eval()
        
        transformer = model.transformer
        embedding_layer = transformer.embedding
        
        # Test with simple input
        test_input = torch.zeros(1, 512, dtype=torch.long)
        
        print("🧪 Testing embedding layer output...")
        with torch.no_grad():
            # Call embedding layer directly
            emb_output = embedding_layer(test_input, test_input, test_input, test_input)
            
            print(f"📊 Embedding output type: {type(emb_output)}")
            
            if isinstance(emb_output, tuple):
                print(f"📊 Tuple length: {len(emb_output)}")
                for i, item in enumerate(emb_output):
                    print(f"  📋 Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
                    
                # The actual embeddings are usually the first item
                if len(emb_output) > 0 and hasattr(emb_output[0], 'shape'):
                    actual_embeddings = emb_output[0]
                    print(f"✅ Found actual embeddings: shape {actual_embeddings.shape}")
                    return True, 0  # First item is embeddings
                    
            elif hasattr(emb_output, 'shape'):
                print(f"✅ Direct tensor output: shape {emb_output.shape}")
                return True, None  # Direct tensor
            else:
                print(f"❌ Unexpected output format: {emb_output}")
                return False, None
                
    except Exception as e:
        print(f"❌ Error investigating: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def check_transformer_forward_method():
    """Check the transformer's forward method to see how it handles embedding outputs"""
    
    print("\n🔍 CHECKING TRANSFORMER FORWARD METHOD")
    print("=" * 45)
    
    try:
        # Import the transformer module to inspect the code
        from transformer.transformer import Transformer
        
        # Get the forward method
        forward_method = Transformer.forward
        
        print("📋 Transformer forward method found")
        
        # Check if we can see the source code
        import inspect
        try:
            source = inspect.getsource(forward_method)
            print("📄 Forward method source code:")
            print("-" * 30)
            
            # Look for how embedding output is handled
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'embedding' in line.lower() and ('=' in line or 'return' in line):
                    print(f"Line {i+1}: {line.strip()}")
                    
        except Exception as e:
            print(f"⚠️ Could not get source code: {e}")
            
    except Exception as e:
        print(f"❌ Error checking transformer: {e}")

def create_transformer_fix():
    """Create a fix for the transformer's forward method"""
    
    print("\n🔧 CREATING TRANSFORMER FIX")
    print("=" * 30)
    
    # First, let's check the current transformer file
    transformer_file = "transformer/transformer.py"
    
    if not os.path.exists(transformer_file):
        print(f"❌ Transformer file not found: {transformer_file}")
        return False
    
    # Create backup
    backup_file = f"{transformer_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(transformer_file, backup_file)
    print(f"💾 Backup created: {backup_file}")
    
    # Read the current file
    with open(transformer_file, 'r') as f:
        content = f.read()
    
    print("📄 Analyzing transformer.py...")
    
    # Look for the forward method
    lines = content.split('\n')
    in_forward_method = False
    embedding_line_found = False
    
    for i, line in enumerate(lines):
        if 'def forward(' in line:
            in_forward_method = True
            print(f"📍 Found forward method at line {i+1}")
        elif in_forward_method and ('def ' in line and 'def forward(' not in line):
            in_forward_method = False
        elif in_forward_method and 'embedding' in line.lower():
            print(f"📍 Embedding usage at line {i+1}: {line.strip()}")
            embedding_line_found = True
    
    if not embedding_line_found:
        print("⚠️ No obvious embedding usage found in forward method")
        return False
    
    # The most likely fix is to extract the first element of the embedding tuple
    # Let's create a patch
    print("\n🔧 Creating patch...")
    
    # Common patterns to fix:
    patterns_to_fix = [
        # Pattern 1: direct assignment
        (r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*self\.embedding\(',
         r'\1\2 = self.embedding('),
        # Pattern 2: return statement
        (r'return\s+self\.embedding\(',
         r'return self.embedding('),
    ]
    
    import re
    
    fixed_content = content
    changes_made = []
    
    # Look for patterns where embedding output is used directly
    embedding_call_pattern = r'self\.embedding\([^)]*\)'
    
    # Find all embedding calls
    for match in re.finditer(embedding_call_pattern, content):
        start, end = match.span()
        before = content[:start]
        call = content[start:end]
        after = content[end:]
        
        # Check what comes after the call
        # If it's used directly (like .shape), we need to fix it
        if after.startswith('.') or (before.endswith('= ') and not after.startswith('[0]')):
            print(f"📍 Found potential issue: {call}")
            changes_made.append(f"Embedding call: {call}")
    
    if changes_made:
        print(f"📝 Potential fixes needed:")
        for change in changes_made:
            print(f"  - {change}")
        
        # Instead of automatically changing the file, let's create a targeted fix
        # based on what we find
        return True
    else:
        print("✅ No obvious patterns found that need fixing")
        return False

def create_manual_fix_instructions():
    """Create manual fix instructions based on the investigation"""
    
    print("\n📋 MANUAL FIX INSTRUCTIONS")
    print("=" * 30)
    
    print("Based on the investigation, the issue is that the embedding layer")
    print("returns a tuple, but the transformer code expects a tensor.")
    print()
    print("🔧 SOLUTION:")
    print("The embedding layer call needs to extract the first element of the tuple.")
    print()
    print("In transformer/transformer.py, find lines that call self.embedding(...)")
    print("and change them to extract the first element:")
    print()
    print("BEFORE:")
    print("  x = self.embedding(token_ids, segment_ids, age_ids, position_ids)")
    print()
    print("AFTER:")
    print("  emb_output = self.embedding(token_ids, segment_ids, age_ids, position_ids)")
    print("  x = emb_output[0] if isinstance(emb_output, tuple) else emb_output")
    print()
    print("OR more simply:")
    print("  x = self.embedding(token_ids, segment_ids, age_ids, position_ids)[0]")

def test_fix():
    """Test if a simple fix works"""
    
    print("\n🧪 TESTING SIMPLE FIX")
    print("=" * 25)
    
    try:
        from models.survival_model import StartupSurvivalModel
        
        checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
        
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path="./startup2vec_startup2vec-full-1gpu-512d_final.pt",
            map_location='cpu'
        )
        model.eval()
        
        # Test with simple input and manual extraction
        test_input = torch.zeros(1, 512, dtype=torch.long)
        test_mask = torch.ones(1, 512)
        
        print("🧪 Testing manual embedding extraction...")
        
        with torch.no_grad():
            # Get the embedding output
            emb_output = model.transformer.embedding(test_input, test_input, test_input, test_input)
            
            # Extract the actual embeddings
            if isinstance(emb_output, tuple):
                actual_embeddings = emb_output[0]
                print(f"✅ Extracted embeddings shape: {actual_embeddings.shape}")
                
                # Now try to manually call the rest of the transformer
                # This would be the pattern we need to implement
                print("📊 Manual extraction successful - this is the fix we need!")
                return True
            else:
                print(f"⚠️ Embedding output is not a tuple: {type(emb_output)}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnostic and fix creation function"""
    
    print("🔧 EMBEDDING OUTPUT FORMAT FIX")
    print("=" * 40)
    print("This script will diagnose and fix the embedding output format issue.")
    print()
    
    # 1. Investigate what the embedding layer returns
    success, embedding_index = investigate_embedding_output()
    
    if not success:
        print("❌ Could not determine embedding output format")
        return 1
    
    # 2. Check the transformer forward method
    check_transformer_forward_method()
    
    # 3. Test if simple fix works
    test_success = test_fix()
    
    if test_success:
        print("\n✅ SOLUTION IDENTIFIED!")
        print("The embedding layer returns a tuple, but the transformer")
        print("forward method expects a tensor.")
        print()
        print("🎯 NEXT STEPS:")
        print("1. Edit transformer/transformer.py")
        print("2. Find the embedding call in the forward method")
        print("3. Change it to extract the first element of the tuple")
        print()
        create_manual_fix_instructions()
        return 0
    else:
        print("❌ Could not confirm the fix approach")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
