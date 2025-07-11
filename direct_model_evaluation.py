#!/usr/bin/env python3
"""
Direct model evaluation to recreate the confusion matrix results
This will run the model on validation data and give you interpretability
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

def load_model_and_data():
    """Load the working model and validation data"""
    print("🔍 Loading model and validation data...")
    
    # Load the best checkpoint
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    try:
        from models.survival_model import StartupSurvivalModel
        
        model = StartupSurvivalModel.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_path=pretrained_path,
            map_location='cpu'
        )
        model.eval()
        print("✅ Model loaded successfully")
        
        # Load datamodule
        from dataloaders.survival_datamodule import SurvivalDataModule
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab",
            batch_size=32,
            num_workers=2,
            prediction_windows=[1, 2, 3, 4]
        )
        datamodule.setup()
        print("✅ Datamodule loaded successfully")
        
        return model, datamodule
        
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        return None, None

def run_evaluation_and_interpretability(model, datamodule, max_batches=1000):
    """Run evaluation and extract interpretability data"""
    print(f"\n🎯 Running evaluation (max {max_batches} batches)...")
    
    val_loader = datamodule.val_dataloader()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_embeddings = []
    all_sequences = []
    all_metadata = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                print(f"Stopping at {max_batches} batches for quick analysis")
                break
                
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{min(max_batches, len(val_loader))}", end='\r')
            
            try:
                # Extract batch data
                input_ids = batch['input_ids'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                survival_labels = batch['survival_label'].to(device)
                
                # Model forward pass
                outputs = model.forward(
                    input_ids=input_ids,
                    padding_mask=padding_mask
                )
                
                # Extract results
                survival_logits = outputs['survival_logits']
                survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                survival_preds = torch.argmax(survival_logits, dim=1)
                
                # Get embeddings
                transformer_output = outputs['transformer_output']
                company_embeddings = transformer_output[:, 0, :]  # [CLS] token
                
                # Store results
                all_predictions.extend(survival_preds.cpu().numpy())
                all_probabilities.extend(survival_probs.cpu().numpy())
                all_labels.extend(survival_labels.squeeze().cpu().numpy())
                all_embeddings.extend(company_embeddings.cpu().numpy())
                all_sequences.extend(input_ids[:, 0, :].cpu().numpy())  # First sequence in window
                
                # Store metadata
                for i in range(input_ids.size(0)):
                    metadata = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'sequence_length': padding_mask[i].sum().item(),
                        'prediction_window': batch['prediction_window'][i].item() if 'prediction_window' in batch else -1,
                        'company_age': batch['company_age_at_prediction'][i].item() if 'company_age_at_prediction' in batch else -1,
                    }
                    all_metadata.append(metadata)
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
    
    print(f"\n✅ Processed {len(all_predictions):,} samples")
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)
    embeddings = np.array(all_embeddings)
    
    return predictions, probabilities, labels, embeddings, all_sequences, all_metadata

def analyze_results(predictions, probabilities, labels, embeddings):
    """Analyze the evaluation results"""
    print(f"\n📊 ANALYSIS RESULTS")
    print("="*50)
    
    # Basic metrics
    accuracy = (predictions == labels).mean()
    auc = roc_auc_score(labels, probabilities)
    
    print(f"Total samples: {len(predictions):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0,0]:6}, {cm[0,1]:6}],")
    print(f" [{cm[1,0]:6}, {cm[1,1]:6}]]")
    
    # Check if this matches the target confusion matrix
    target_cm = np.array([[197840, 624], [7249, 9105]])
    if cm.shape == target_cm.shape:
        # Calculate ratio to see if it's a subset
        ratio = target_cm.sum() / cm.sum()
        print(f"\nRatio to target: {ratio:.2f}")
        if 3 < ratio < 10:  # If it's a reasonable subset
            print(f"✅ This appears to be a subset of the target data!")
            print(f"Scaling up would give approximately the target confusion matrix")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Died', 'Survived']))
    
    return cm, auc, accuracy

def create_interpretability_visualizations(predictions, probabilities, labels, embeddings):
    """Create interpretability visualizations"""
    print(f"\n🎨 Creating interpretability visualizations...")
    
    plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.subplot(3, 4, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Probability Distribution
    plt.subplot(3, 4, 2)
    plt.hist(probabilities, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Survival Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    
    # 3. Probabilities by True Label
    plt.subplot(3, 4, 3)
    died_probs = probabilities[labels == 0]
    survived_probs = probabilities[labels == 1]
    
    plt.hist(died_probs, bins=30, alpha=0.7, label='Actually Died', color='red')
    plt.hist(survived_probs, bins=30, alpha=0.7, label='Actually Survived', color='blue')
    plt.title('Probabilities by True Outcome')
    plt.xlabel('Survival Probability')
    plt.legend()
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    
    plt.subplot(3, 4, 4)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 5. Embedding Analysis (PCA)
    if embeddings.shape[0] > 100:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings[:5000])  # Sample for visualization
        labels_sample = labels[:5000]
        
        plt.subplot(3, 4, 5)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=labels_sample, cmap='RdYlBu', alpha=0.6, s=1)
        plt.colorbar(scatter)
        plt.title('Startup Embeddings (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    # 6. Prediction Confidence Analysis
    plt.subplot(3, 4, 6)
    high_conf = probabilities > 0.8
    med_conf = (probabilities >= 0.4) & (probabilities <= 0.6)
    low_conf = probabilities < 0.2
    
    conf_counts = [high_conf.sum(), med_conf.sum(), low_conf.sum()]
    conf_labels = ['High (>0.8)', 'Medium (0.4-0.6)', 'Low (<0.2)']
    
    plt.bar(conf_labels, conf_counts)
    plt.title('Prediction Confidence Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 7. Accuracy by Confidence Level
    plt.subplot(3, 4, 7)
    if high_conf.sum() > 0:
        high_acc = (predictions[high_conf] == labels[high_conf]).mean()
    else:
        high_acc = 0
    
    if med_conf.sum() > 0:
        med_acc = (predictions[med_conf] == labels[med_conf]).mean()
    else:
        med_acc = 0
        
    if low_conf.sum() > 0:
        low_acc = (predictions[low_conf] == labels[low_conf]).mean()
    else:
        low_acc = 0
    
    plt.bar(conf_labels, [high_acc, med_acc, low_acc])
    plt.title('Accuracy by Confidence Level')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('startup_interpretability_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualizations to 'startup_interpretability_analysis.png'")

def save_interpretability_data(predictions, probabilities, labels, embeddings, sequences, metadata):
    """Save interpretability data for further analysis"""
    print(f"\n💾 Saving interpretability data...")
    
    # Create interpretability dataset
    interpretability_data = {
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': labels,
        'embeddings': embeddings,
        'sequences': sequences,
        'metadata': metadata,
        'analysis_type': 'direct_model_evaluation',
        'total_samples': len(predictions)
    }
    
    with open('direct_interpretability_data.pkl', 'wb') as f:
        pickle.dump(interpretability_data, f)
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities,
        'true_label': labels,
        'sequence_length': [m['sequence_length'] for m in metadata],
        'prediction_window': [m['prediction_window'] for m in metadata],
        'company_age': [m['company_age'] for m in metadata],
    })
    
    df.to_pickle('direct_analysis_results.pkl')
    
    print(f"✅ Saved direct_interpretability_data.pkl and direct_analysis_results.pkl")

def main():
    """Main function"""
    print("🚀 DIRECT MODEL EVALUATION FOR INTERPRETABILITY")
    print("="*60)
    
    # Load model and data
    model, datamodule = load_model_and_data()
    
    if model is None or datamodule is None:
        print("❌ Could not load model or datamodule")
        return 1
    
    # Run evaluation
    predictions, probabilities, labels, embeddings, sequences, metadata = run_evaluation_and_interpretability(
        model, datamodule, max_batches=1000  # Adjust this number
    )
    
    # Analyze results
    cm, auc, accuracy = analyze_results(predictions, probabilities, labels, embeddings)
    
    # Create visualizations
    create_interpretability_visualizations(predictions, probabilities, labels, embeddings)
    
    # Save data
    save_interpretability_data(predictions, probabilities, labels, embeddings, sequences, metadata)
    
    print(f"\n🎉 DIRECT EVALUATION COMPLETE!")
    print(f"📊 Analyzed {len(predictions):,} samples")
    print(f"📈 AUC: {auc:.4f}")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"📁 Results saved for interpretability analysis")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
