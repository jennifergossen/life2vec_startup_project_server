#!/usr/bin/env python3

"""
Quick fix script to add bounds checking to the datamodule
This will patch step_4_create_datamodule.py to add safety checks
"""

import re

def fix_datamodule_bounds():
    """Add bounds checking to the datamodule"""
    
    # Read the current datamodule file
    with open("step_4_create_datamodule.py", "r") as f:
        content = f.read()
    
    # Check if already patched
    if "# BOUNDS SAFETY CHECK" in content:
        print("✅ Bounds checking already added to datamodule")
        return
    
    # Find the encode_document method and add bounds checking
    # Look for the token_ids assignment
    pattern = r'(token_ids = np\.array\([^)]+\))'
    
    replacement = '''token_ids = np.array([self.vocab.token2index.get(token, self.vocab.token2index["[UNK]"]) for token in tokens])
        
        # BOUNDS SAFETY CHECK - ensure all token IDs are valid
        vocab_size = len(self.vocab.token2index)
        invalid_mask = (token_ids >= vocab_size) | (token_ids < 0)
        if invalid_mask.any():
            print(f"⚠️ Found {invalid_mask.sum()} invalid token IDs, replacing with [UNK]")
            token_ids[invalid_mask] = self.vocab.token2index["[UNK]"]'''
    
    # Apply the fix
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added bounds checking to token_ids assignment")
    else:
        # Alternative approach - add the check after any token_ids creation
        # Look for return statements in encode_document
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            # If we see a token_ids assignment, add bounds check after it
            if 'token_ids' in line and 'np.array' in line and 'token2index' in line:
                new_lines.extend([
                    '',
                    '        # BOUNDS SAFETY CHECK - ensure all token IDs are valid',
                    '        vocab_size = len(self.vocab.token2index)',
                    '        invalid_mask = (token_ids >= vocab_size) | (token_ids < 0)',
                    '        if invalid_mask.any():',
                    '            print(f"⚠️ Found {invalid_mask.sum()} invalid token IDs, replacing with [UNK]")',
                    '            token_ids[invalid_mask] = self.vocab.token2index["[UNK]"]',
                ])
        
        content = '\n'.join(new_lines)
        print("✅ Added bounds checking after token_ids assignments")
    
    # Write the fixed file
    with open("step_4_create_datamodule.py", "w") as f:
        f.write(content)
    
    print("💾 Updated step_4_create_datamodule.py with bounds checking")
    print("🔄 Please re-run the training script")

if __name__ == "__main__":
    fix_datamodule_bounds()
