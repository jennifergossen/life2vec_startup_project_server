#!/usr/bin/env python3
"""
Update Interpretability Scripts to Use New Checkpoint
Updates all interpretability scripts to use the new finetuned checkpoint path.
"""

import re
from pathlib import Path

def update_checkpoint_in_file(file_path, old_checkpoint, new_checkpoint):
    """Update checkpoint path in a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the checkpoint path
    updated_content = content.replace(old_checkpoint, new_checkpoint)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated {file_path}")

def main():
    """Update all interpretability scripts"""
    
    # Old checkpoint (current)
    old_checkpoint = "survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"
    
    # New checkpoint (will be created after retraining)
    new_checkpoint = "survival_checkpoints_FIXED/finetune-v2/best-epoch=XX-val/balanced_acc=X.XXXX.ckpt"
    
    # Files to update
    interpretability_files = [
        "interp_02_data_contribution_analysis.py",
        "interp_03_visual_exploration.py", 
        "interp_04_local_explainability.py",
        "interp_05_global_explainability.py"
    ]
    
    print("🎯 UPDATING INTERPRETABILITY SCRIPTS")
    print("=" * 50)
    print(f"Old checkpoint: {old_checkpoint}")
    print(f"New checkpoint: {new_checkpoint}")
    print()
    
    for file_path in interpretability_files:
        if Path(file_path).exists():
            update_checkpoint_in_file(file_path, old_checkpoint, new_checkpoint)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n🎉 All interpretability scripts updated!")
    print("\n📝 NOTE: You'll need to update the new checkpoint path manually")
    print("   after finetuning completes, or run this script again with the")
    print("   actual checkpoint path.")

if __name__ == "__main__":
    main() 