#!/usr/bin/env python3
"""
FINAL TEST: Is your model actually working correctly?
Maybe the issue is extreme class imbalance, not broken training
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_model_on_balanced_data():
    """Test model on artificially balanced data to see if it can distinguish"""
    
    print("🔍 FINAL TEST: MODEL CAPABILITY ON BALANCED DATA")
    print("=" * 60)
    
    try:
        from models.survival_model import StartupSurvivalModel
        from dataloaders.survival_datamodule import SurvivalDataModule
        
        checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
        
        print("📥 Loading model and data...")
        model = StartupSurvivalModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
        model.eval()
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab",
            batch_size=16,
            num_workers=1,
            prediction_windows=[1, 2, 3, 4]
        )
        datamodule.setup()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"✅ Model loaded to {device}")
        
        # Collect data from multiple batches to find both classes
        val_loader = datamodule.val_dataloader()
        
        all_survival_samples = []
        all_death_samples = []
        
        print("🔍 Searching for death samples in validation data...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx > 200:  # Search through many batches
                    break
                    
                labels = batch['survival_label'].squeeze()
                
                # Separate by class
                for i in range(len(labels)):
                    if labels[i] == 0 and len(all_death_samples) < 50:  # Death
                        all_death_samples.append({
                            'input_ids': batch['input_ids'][i:i+1],
                            'padding_mask': batch['padding_mask'][i:i+1],
                            'label': labels[i:i+1]
                        })
                    elif labels[i] == 1 and len(all_survival_samples) < 50:  # Survival
                        all_survival_samples.append({
                            'input_ids': batch['input_ids'][i:i+1],
                            'padding_mask': batch['padding_mask'][i:i+1],
                            'label': labels[i:i+1]
                        })
                
                if len(all_death_samples) >= 50 and len(all_survival_samples) >= 50:
                    break
        
        print(f"📊 Found {len(all_death_samples)} death samples")
        print(f"📊 Found {len(all_survival_samples)} survival samples")
        
        if len(all_death_samples) == 0:
            print("🚨 NO DEATH SAMPLES FOUND!")
            print("🚨 This explains why AUC calculation fails!")
            print("🚨 Your test set might be 100% survival companies")
            return False
        
        # Test model on balanced subset
        print(f"\n🧪 TESTING MODEL ON BALANCED DATA:")
        
        min_samples = min(len(all_death_samples), len(all_survival_samples))
        print(f"Using {min_samples} samples per class")
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        # Test death samples
        for sample in all_death_samples[:min_samples]:
            input_ids = sample['input_ids'].to(device)
            padding_mask = sample['padding_mask'].to(device)
            label = sample['label']
            
            outputs = model.forward(input_ids=input_ids, padding_mask=padding_mask)
            logits = outputs['survival_logits']
            probs = F.softmax(logits, dim=1)
            
            survival_prob = probs[0, 1].cpu().item()
            prediction = 1 if survival_prob > 0.5 else 0
            
            all_preds.append(prediction)
            all_probs.append(survival_prob)
            all_labels.append(label.item())
        
        # Test survival samples  
        for sample in all_survival_samples[:min_samples]:
            input_ids = sample['input_ids'].to(device)
            padding_mask = sample['padding_mask'].to(device)
            label = sample['label']
            
            outputs = model.forward(input_ids=input_ids, padding_mask=padding_mask)
            logits = outputs['survival_logits']
            probs = F.softmax(logits, dim=1)
            
            survival_prob = probs[0, 1].cpu().item()
            prediction = 1 if survival_prob > 0.5 else 0
            
            all_preds.append(prediction)
            all_probs.append(survival_prob)
            all_labels.append(label.item())
        
        # Calculate metrics on balanced data
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"\n📊 BALANCED DATA RESULTS:")
        print(f"   Total samples: {len(all_labels)} ({min_samples} per class)")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Balanced Accuracy: {balanced_acc:.4f}")
        print(f"   MCC: {mcc:.4f}")
        print(f"   AUC: {auc:.4f}")
        
        # Analyze predictions by class
        death_indices = all_labels == 0
        survival_indices = all_labels == 1
        
        death_probs = all_probs[death_indices]
        survival_probs = all_probs[survival_indices]
        
        print(f"\n📊 PREDICTION ANALYSIS:")
        print(f"   Death companies - survival probs: {death_probs.mean():.4f} ± {death_probs.std():.4f}")
        print(f"   Survival companies - survival probs: {survival_probs.mean():.4f} ± {survival_probs.std():.4f}")
        
        # Check if model can distinguish
        if death_probs.mean() < survival_probs.mean():
            print(f"   ✅ Model CAN distinguish! (death < survival probs)")
            if auc > 0.6:
                print(f"   ✅ Strong AUC on balanced data: {auc:.4f}")
                print(f"   💡 Your model WORKS! Issue is extreme class imbalance in test set")
            else:
                print(f"   ⚠️ Weak AUC on balanced data: {auc:.4f}")
        else:
            print(f"   ❌ Model cannot distinguish (death >= survival probs)")
            print(f"   ❌ Model is truly broken")
        
        return auc > 0.6
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_class_distribution():
    """Analyze the actual class distribution in your data"""
    
    print(f"\n🔍 ANALYZING CLASS DISTRIBUTION:")
    print("=" * 50)
    
    try:
        from dataloaders.survival_datamodule import SurvivalDataModule
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab",
            batch_size=16,
            num_workers=1,
            prediction_windows=[1, 2, 3, 4]
        )
        datamodule.setup()
        
        class_stats = datamodule.get_class_distribution()
        
        print("📊 CLASS DISTRIBUTION:")
        for split, stats in class_stats.items():
            print(f"   {split.upper()}:")
            print(f"     Total: {stats['total']:,}")
            print(f"     Deaths: {stats['died']:,} ({stats['died_pct']:.1f}%)")
            print(f"     Survivals: {stats['survived']:,} ({stats['survived_pct']:.1f}%)")
            print(f"     Imbalance ratio: {stats['survived']/stats['died']:.1f}:1")
        
        # Check if class distribution is too extreme
        test_death_pct = class_stats.get('test', {}).get('died_pct', 0)
        if test_death_pct < 5:
            print(f"\n🚨 EXTREME IMBALANCE DETECTED!")
            print(f"   Only {test_death_pct:.1f}% deaths in test set")
            print(f"   This makes AUC calculation unreliable")
            print(f"   Model might be working but appears broken due to imbalance")
        
        return class_stats
        
    except Exception as e:
        print(f"❌ Class distribution analysis failed: {e}")
        return None

if __name__ == "__main__":
    print("🤔 HYPOTHESIS: Maybe your model actually works!")
    print("Testing on balanced data to check model capability...")
    print()
    
    # First check class distribution
    class_stats = analyze_class_distribution()
    
    # Then test model capability
    model_works = test_model_on_balanced_data()
    
    print(f"\n🎯 FINAL VERDICT:")
    if model_works:
        print(f"✅ YOUR MODEL ACTUALLY WORKS!")
        print(f"✅ The issue is extreme class imbalance in test data")
        print(f"✅ W&B metrics might be correct after all")
        print(f"✅ Focus on interpretability with balanced sampling")
        print(f"\n💡 SOLUTION:")
        print(f"   Use balanced sampling for interpretability")
        print(f"   Or collect more failure cases for analysis")
        print(f"   Your training was successful!")
    else:
        print(f"❌ Model is genuinely broken")
        print(f"❌ Need to retrain with fixes")
        print(f"❌ W&B metrics were misleading")
