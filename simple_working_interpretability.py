#!/usr/bin/env python3
"""
SIMPLE WORKING INTERPRETABILITY ANALYSIS
Focus on getting basic interpretability working with your model
"""

import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def simple_model_test():
    """Simple test to see if we can load and run the model"""
    print("🔍 SIMPLE MODEL TEST")
    print("="*50)
    
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    print(f"Pretrained exists: {os.path.exists(pretrained_path)}")
    
    try:
        from models.survival_model import StartupSurvivalModel
        print("✅ Successfully imported StartupSurvivalModel")
        
        # Try loading the model
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path=pretrained_path,
            map_location='cpu'
        )
        print("✅ Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        # Try loading datamodule
        from dataloaders.survival_datamodule import SurvivalDataModule
        print("✅ Successfully imported SurvivalDataModule")
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab",
            batch_size=8,  # Small batch for testing
            num_workers=0  # No multiprocessing for debugging
        )
        datamodule.setup()
        print("✅ Datamodule setup successfully")
        
        # Get a single batch for testing
        val_loader = datamodule.val_dataloader()
        print(f"Validation loader created with {len(val_loader)} batches")
        
        # Test with one batch
        batch = next(iter(val_loader))
        print(f"✅ Got batch with keys: {list(batch.keys())}")
        
        input_ids = batch['input_ids']
        padding_mask = batch['padding_mask']
        survival_labels = batch['survival_label']
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Padding mask shape: {padding_mask.shape}")
        print(f"Labels shape: {survival_labels.shape}")
        print(f"Sample labels: {survival_labels[:5]}")
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                padding_mask=padding_mask
            )
            print(f"✅ Model forward pass successful")
            print(f"Output keys: {list(outputs.keys())}")
            
            survival_logits = outputs['survival_logits']
            survival_probs = torch.softmax(survival_logits, dim=1)
            
            print(f"Logits shape: {survival_logits.shape}")
            print(f"Sample logits: {survival_logits[:3]}")
            print(f"Sample probs: {survival_probs[:3]}")
            
        return model, datamodule, True
        
    except Exception as e:
        print(f"❌ Error in simple test: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def run_minimal_evaluation(model, datamodule, num_batches=50):
    """Run minimal evaluation to get interpretability data"""
    print(f"\n🎯 RUNNING MINIMAL EVALUATION ({num_batches} batches)")
    print("="*50)
    
    val_loader = datamodule.val_dataloader()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_embeddings = []
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
                
            try:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                survival_labels = batch['survival_label'].to(device)
                
                # Forward pass
                outputs = model.forward(
                    input_ids=input_ids,
                    padding_mask=padding_mask
                )
                
                # Extract predictions
                survival_logits = outputs['survival_logits']
                survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]  # Probability of survival
                survival_preds = torch.argmax(survival_logits, dim=1)
                
                # Get embeddings (company representations)
                transformer_output = outputs['transformer_output']
                company_embeddings = transformer_output[:, 0, :]  # [CLS] token embedding
                
                # Store results
                all_predictions.extend(survival_preds.cpu().numpy())
                all_probabilities.extend(survival_probs.cpu().numpy())
                all_labels.extend(survival_labels.squeeze().cpu().numpy())
                all_embeddings.extend(company_embeddings.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{num_batches}", end='\r')
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
    
    print(f"\n✅ Processed {len(all_predictions):,} samples")
    
    return (np.array(all_predictions), 
            np.array(all_probabilities), 
            np.array(all_labels), 
            np.array(all_embeddings))

def analyze_and_visualize(predictions, probabilities, labels, embeddings):
    """Create analysis and visualizations"""
    print(f"\n📊 CREATING ANALYSIS AND VISUALIZATIONS")
    print("="*50)
    
    # Basic metrics
    accuracy = (predictions == labels).mean()
    
    # Handle AUC calculation safely
    try:
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probabilities)
        else:
            auc = float('nan')
            print("⚠️ Only one class in labels, cannot calculate AUC")
    except Exception as e:
        auc = float('nan')
        print(f"⚠️ AUC calculation failed: {e}")
    
    print(f"Total samples: {len(predictions):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    # Prediction distribution
    print(f"\nPrediction statistics:")
    print(f"  Probability mean: {probabilities.mean():.4f}")
    print(f"  Probability std: {probabilities.std():.4f}")
    print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Compare with target confusion matrix
    target_cm = np.array([[197840, 624], [7249, 9105]])
    total_target = target_cm.sum()
    total_current = cm.sum()
    ratio = total_target / total_current if total_current > 0 else 0
    
    print(f"\nComparison with target confusion matrix:")
    print(f"  Target total samples: {total_target:,}")
    print(f"  Current total samples: {total_current:,}")
    print(f"  Ratio: {ratio:.2f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Probability Distribution
    plt.subplot(2, 3, 2)
    plt.hist(probabilities, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Survival Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    
    # 3. Probabilities by True Label
    plt.subplot(2, 3, 3)
    if len(np.unique(labels)) > 1:
        for label in np.unique(labels):
            label_probs = probabilities[labels == label]
            plt.hist(label_probs, bins=20, alpha=0.7, 
                    label=f'True Label {label}', density=True)
        plt.title('Probabilities by True Label')
        plt.xlabel('Survival Probability')
        plt.ylabel('Density')
        plt.legend()
    
    # 4. ROC Curve (if possible)
    plt.subplot(2, 3, 4)
    if not np.isnan(auc):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probabilities)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'AUC not available', ha='center', va='center')
        plt.title('ROC Curve (N/A)')
    
    # 5. Embedding PCA (if we have enough samples)
    plt.subplot(2, 3, 5)
    if len(embeddings) > 10:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=labels, cmap='RdYlBu', alpha=0.6, s=20)
        plt.colorbar(scatter)
        plt.title('Company Embeddings (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    # 6. Prediction Confidence
    plt.subplot(2, 3, 6)
    confidence_bins = ['Low (<0.3)', 'Medium (0.3-0.7)', 'High (>0.7)']
    low_conf = (probabilities < 0.3).sum()
    med_conf = ((probabilities >= 0.3) & (probabilities <= 0.7)).sum()
    high_conf = (probabilities > 0.7).sum()
    
    plt.bar(confidence_bins, [low_conf, med_conf, high_conf])
    plt.title('Prediction Confidence Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('simple_interpretability_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualizations to 'simple_interpretability_analysis.png'")
    
    # Save data for further analysis
    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'embeddings': embeddings,
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(predictions)
        }
    }
    
    with open('simple_interpretability_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Saved results to 'simple_interpretability_results.pkl'")
    
    return results

def main():
    """Main function for simple interpretability analysis"""
    print("🚀 SIMPLE WORKING INTERPRETABILITY ANALYSIS")
    print("="*60)
    print("Focus: Get basic interpretability working with your model")
    
    # Step 1: Simple model test
    model, datamodule, success = simple_model_test()
    
    if not success:
        print("\n❌ Simple model test failed. Please check:")
        print("  1. Model checkpoint exists and is valid")
        print("  2. Pretrained model exists")
        print("  3. All dependencies are installed")
        print("  4. Data files are accessible")
        return 1
    
    # Step 2: Run minimal evaluation
    try:
        predictions, probabilities, labels, embeddings = run_minimal_evaluation(
            model, datamodule, num_batches=100  # Start with 100 batches
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Analyze and visualize
    try:
        results = analyze_and_visualize(predictions, probabilities, labels, embeddings)
        
        print(f"\n🎉 SIMPLE INTERPRETABILITY ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(predictions):,} samples")
        print(f"📈 Results saved for further analysis")
        print(f"📊 Check 'simple_interpretability_analysis.png' for visualizations")
        
        # Give next steps
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Check the visualizations to understand model behavior")
        print(f"  2. If results look good, increase num_batches for more data")
        print(f"  3. Use 'simple_interpretability_results.pkl' for further analysis")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
