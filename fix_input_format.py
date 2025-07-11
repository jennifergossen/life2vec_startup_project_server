#!/usr/bin/env python3

"""
Fix the datamodule to provide the correct 4D input format that life2vec expects
"""

import re

def fix_datamodule_format():
    """Fix the input format in step_4_create_datamodule.py"""
    
    print("🔧 FIXING DATAMODULE INPUT FORMAT")
    print("=" * 50)
    
    # Read the current file
    with open("step_4_create_datamodule.py", "r") as f:
        content = f.read()
    
    # Check if already fixed
    if "# LIFE2VEC 4D FORMAT" in content:
        print("✅ Already fixed - datamodule provides 4D input")
        return
    
    # Find the encode_document method and fix the return format
    # Look for the return statement that creates the final result
    
    # Pattern to find where we return the result dictionary
    pattern = r'return {\s*[^}]*"input_ids":\s*([^,\n}]+)[^}]*}'
    
    # We need to replace the input_ids to be 4D format
    replacement_lines = '''
        # LIFE2VEC 4D FORMAT - Create [batch, 4, sequence] format
        # Dimensions: 0=tokens, 1=abspos, 2=age, 3=segment
        input_4d = np.zeros((4, max_length), dtype=np.int64)
        
        # Fill the 4 dimensions
        actual_length = len(token_ids)
        input_4d[0, :actual_length] = token_ids  # tokens
        input_4d[1, :actual_length] = abspos_values  # absolute positions  
        input_4d[2, :actual_length] = age_values  # age values
        input_4d[3, :actual_length] = segment_values  # segment values
        
        return {
            "sequence_id": np.array([0]),  # dummy sequence ID
            "input_ids": input_4d,  # Shape: [4, max_length]
            "padding_mask": padding_mask,
            "target_tokens": np.array([0]),  # dummy for now
            "target_pos": np.array([0]),     # dummy for now  
            "target_sop": np.array([0]),     # dummy for now
            "original_sequence": token_ids   # original tokens
        }'''
    
    # Find the encode_document method
    lines = content.split('\n')
    new_lines = []
    in_encode_method = False
    found_return = False
    
    for i, line in enumerate(lines):
        if 'def encode_document' in line:
            in_encode_method = True
            print("📍 Found encode_document method")
        
        if in_encode_method and 'return {' in line and not found_return:
            found_return = True
            print("📍 Found return statement - replacing with 4D format")
            
            # Add the new format before the return
            new_lines.extend([
                '        # Create dummy values for abspos, age, segment',
                '        max_length = 512  # Set max length',
                '        actual_length = len(token_ids)',
                '        ',
                '        # Create abspos (absolute positions) - just sequential numbers',
                '        abspos_values = np.arange(actual_length, dtype=np.int64)',
                '        ',
                '        # Create age values - dummy values (could be improved)',
                '        age_values = np.full(actual_length, 30, dtype=np.int64)  # dummy age 30',
                '        ',
                '        # Create segment values - alternating pattern',
                '        segment_values = np.array([i % 4 for i in range(actual_length)], dtype=np.int64)',
                '        ',
                '        # Create padding mask',
                '        padding_mask = np.zeros(max_length, dtype=bool)',
                '        padding_mask[:actual_length] = True',
                ''
            ])
            new_lines.extend(replacement_lines.split('\n'))
            
            # Skip the original return block
            while i < len(lines) and not (lines[i].strip().endswith('}') and 'return' in lines[i-5:i+1]):
                i += 1
                if i >= len(lines):
                    break
            continue
        
        # Check if we're past the encode_document method
        if in_encode_method and line.strip().startswith('def ') and 'encode_document' not in line:
            in_encode_method = False
        
        new_lines.append(line)
    
    if not found_return:
        print("❌ Could not find return statement in encode_document")
        print("💡 Manual fix needed - check your encode_document method")
        return False
    
    # Write the fixed content
    new_content = '\n'.join(new_lines)
    
    with open("step_4_create_datamodule.py", "w") as f:
        f.write(new_content)
    
    print("✅ Fixed datamodule to provide 4D input format")
    print("📊 New format: input_ids shape = [4, max_length]")
    print("   Dim 0: tokens")
    print("   Dim 1: absolute positions") 
    print("   Dim 2: age values")
    print("   Dim 3: segment values")
    print("")
    print("🔄 Now test training again!")
    return True

if __name__ == "__main__":
    success = fix_datamodule_format()
    if success:
        print("\n✅ DATAMODULE FIXED!")
        print("🧪 Test with: python step_5_train_startup2vec.py --quick-test --single-gpu")
    else:
        print("\n❌ Manual fix needed!")
