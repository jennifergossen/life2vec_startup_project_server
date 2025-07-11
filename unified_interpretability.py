#!/usr/bin/env python3
"""
UNIFIED FULL-SCALE INTERPRETABILITY ANALYSIS
Combines working model evaluation with comprehensive interpretability
"""

import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class StartupInterpretabilityAnalyzer:
    """Unified class for comprehensive startup survival interpretability"""
    
    def __init__(self, checkpoint_path, pretrained_path, output_dir="interpretability_results"):
        self.checkpoint_path = checkpoint_path
        self.pretrained_path = pretrained_path
        self.output_dir = output_dir
        self.model = None
        self.datamodule = None
        
        # Results storage
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.embeddings = None
        self.sequences = None
        self.metadata = None
        
        # Analysis results
        self.performance_results = None
        self.embedding_results = None
        self.interpretability_data = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model_and_data(self):
        """Load the trained model and validation data"""
        print("🔍 Loading model and validation data...")
        
        try:
            from models.survival_model import StartupSurvivalModel
            from dataloaders.survival_datamodule import SurvivalDataModule
            
            # Load model
            self.model = StartupSurvivalModel.load_from_checkpoint(
                self.checkpoint_path,
                pretrained_model_path=self.pretrained_path,
                map_location='cpu'
            )
            self.model.eval()
            print("✅ Model loaded successfully")
            
            # Load datamodule
            self.datamodule = SurvivalDataModule(
                corpus_name="startup_corpus",
                vocab_name="startup_vocab",
                batch_size=32,
                num_workers=2,
                prediction_windows=[1, 2, 3, 4]
            )
            self.datamodule.setup()
            print("✅ Datamodule loaded successfully")
            
            val_loader = self.datamodule.val_dataloader()
            print(f"📊 Validation set has {len(val_loader):,} batches")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_full_validation_data(self, max_batches=None, save_every=500):
        """Extract complete validation data with progress tracking"""
        print(f"\\n🎯 EXTRACTING FULL VALIDATION DATA")
        print("="*60)
        
        if max_batches is None:
            max_batches = len(self.datamodule.val_dataloader())
        
        print(f"📊 Processing up to {max_batches:,} batches")
        print(f"💾 Saving progress every {save_every} batches")
        
        val_loader = self.datamodule.val_dataloader()
        
        # Storage lists
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        all_sequences = []
        all_metadata = []
        
        # Progress tracking
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        print(f"Using device: {device}")
        
        start_time = time.time()
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                # Progress reporting
                if batch_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = batch_idx / elapsed if elapsed > 0 else 0
                    eta = (max_batches - batch_idx) / rate if rate > 0 else 0
                    
                    print(f"Progress: {batch_idx:,}/{max_batches:,} "
                          f"({batch_idx/max_batches*100:.1f}%) | "
                          f"Samples: {processed_samples:,} | "
                          f"Rate: {rate:.1f} batch/s | "
                          f"ETA: {eta/60:.1f}min", end='\\r')
                
                try:
                    # Extract batch data
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    # Model forward pass
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        padding_mask=padding_mask
                    )
                    
                    # Extract predictions and embeddings
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    # Get company embeddings
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]  # [CLS] token
                    
                    # Store results (move to CPU for memory efficiency)
                    batch_predictions = survival_preds.cpu().numpy()
                    batch_probs = survival_probs.cpu().numpy()
                    batch_labels = survival_labels.squeeze().cpu().numpy()
                    batch_embeddings = company_embeddings.cpu().numpy()
                    batch_sequences = input_ids[:, 0, :].cpu().numpy()  # First sequence
                    
                    all_predictions.extend(batch_predictions)
                    all_probabilities.extend(batch_probs)
                    all_labels.extend(batch_labels)
                    all_embeddings.extend(batch_embeddings)
                    all_sequences.extend(batch_sequences)
                    
                    # Store metadata
                    for i in range(input_ids.size(0)):
                        metadata = {
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'sequence_length': padding_mask[i].sum().item(),
                            'prediction_window': batch['prediction_window'][i].item() if 'prediction_window' in batch else -1,
                            'company_age': batch['company_age_at_prediction'][i].item() if 'company_age_at_prediction' in batch else -1,
                            'sequence_id': batch['sequence_id'][i].item() if 'sequence_id' in batch else -1,
                        }
                        all_metadata.append(metadata)
                    
                    processed_samples += len(batch_predictions)
                    
                    # Save intermediate results
                    if batch_idx > 0 and batch_idx % save_every == 0:
                        self._save_intermediate_results(
                            all_predictions, all_probabilities, all_labels,
                            all_embeddings, all_sequences, all_metadata,
                            batch_idx
                        )
                    
                except Exception as e:
                    print(f"\\nError in batch {batch_idx}: {e}")
                    continue
        
        print(f"\\n✅ Extraction complete!")
        print(f"📊 Total samples: {processed_samples:,}")
        print(f"⏱️ Processing time: {(time.time() - start_time)/60:.1f} minutes")
        
        # Store results
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.labels = np.array(all_labels)
        self.embeddings = np.array(all_embeddings)
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        return True
    
    def analyze_model_performance(self):
        """Comprehensive model performance analysis"""
        print(f"\\n📊 MODEL PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Basic metrics
        accuracy = (self.predictions == self.labels).mean()
        
        # Handle AUC calculation
        try:
            if len(np.unique(self.labels)) > 1:
                auc = roc_auc_score(self.labels, self.probabilities)
            else:
                auc = float('nan')
        except:
            auc = float('nan')
        
        # Distribution analysis
        survival_rate = np.mean(self.labels)
        pred_survival_rate = np.mean(self.probabilities)
        
        print(f"📈 Overall Performance:")
        print(f"  Total samples: {len(self.predictions):,}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Actual survival rate: {survival_rate:.2%}")
        print(f"  Predicted survival rate: {pred_survival_rate:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        print(f"\\nConfusion Matrix:")
        print(cm)
        
        # Compare with target
        target_cm = np.array([[197840, 624], [7249, 9105]])
        if cm.shape == target_cm.shape:
            ratio = target_cm.sum() / cm.sum()
            print(f"\\nRatio to target confusion matrix: {ratio:.2f}")
            
            if 1.5 < ratio < 10:
                print(f"✅ This appears to be a subset of the target data!")
                print(f"Scaling up would approximate the target confusion matrix")
        
        # Detailed classification report
        if len(np.unique(self.labels)) > 1:
            print(f"\\nClassification Report:")
            print(classification_report(self.labels, self.predictions, 
                                      target_names=['Died', 'Survived']))
        
        # Prediction confidence analysis
        high_conf = (self.probabilities > 0.8).sum()
        med_conf = ((self.probabilities >= 0.4) & (self.probabilities <= 0.6)).sum()
        low_conf = (self.probabilities < 0.2).sum()
        
        print(f"\\n🎯 Prediction Confidence:")
        print(f"  High confidence (>0.8): {high_conf:,} ({high_conf/len(self.probabilities)*100:.1f}%)")
        print(f"  Medium confidence (0.4-0.6): {med_conf:,} ({med_conf/len(self.probabilities)*100:.1f}%)")
        print(f"  Low confidence (<0.2): {low_conf:,} ({low_conf/len(self.probabilities)*100:.1f}%)")
        
        self.performance_results = {
            'accuracy': accuracy,
            'auc': auc,
            'survival_rate': survival_rate,
            'pred_survival_rate': pred_survival_rate,
            'confusion_matrix': cm,
            'total_samples': len(self.predictions),
            'confidence_analysis': {
                'high_conf': high_conf,
                'med_conf': med_conf,
                'low_conf': low_conf
            }
        }
    
    def analyze_embeddings(self, sample_size=5000):
        """Comprehensive embedding space analysis"""
        print(f"\\n🧠 EMBEDDING SPACE ANALYSIS")
        print("="*50)
        
        # Sample for analysis if too large
        if len(self.embeddings) > sample_size:
            print(f"📊 Sampling {sample_size:,} startups for embedding analysis...")
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            embeddings_sample = self.embeddings[indices]
            probs_sample = self.probabilities[indices]
            labels_sample = self.labels[indices]
        else:
            embeddings_sample = self.embeddings
            probs_sample = self.probabilities
            labels_sample = self.labels
        
        # 1. PCA Analysis
        print(f"\\n📊 Principal Component Analysis:")
        pca = PCA(n_components=min(50, embeddings_sample.shape[1]))
        pca_embeddings = pca.fit_transform(embeddings_sample)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"  Embedding dimension: {embeddings_sample.shape[1]}")
        print(f"  First 5 components: {explained_variance[:5].sum():.1%} variance")
        print(f"  First 10 components: {explained_variance[:10].sum():.1%} variance")
        print(f"  Components for 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
        
        # 2. Clustering Analysis
        print(f"\\n🔄 Clustering Analysis:")
        cluster_embeddings = pca_embeddings[:, :20]  # Use first 20 PCs
        
        # Find optimal clusters
        best_k = 5
        silhouette_scores = []
        
        for k in range(2, min(11, len(embeddings_sample)//100)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_embeddings)
                wcss = kmeans.inertia_
                silhouette_scores.append((k, wcss))
            except:
                continue
        
        if silhouette_scores:
            # Simple elbow method
            best_k = min(silhouette_scores, key=lambda x: x[1])[0]
            best_k = min(best_k, 8)  # Cap for interpretability
        
        print(f"  Optimal clusters: {best_k}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_embeddings)
        
        print(f"  Cluster analysis:")
        cluster_results = []
        for i in range(best_k):
            mask = cluster_labels == i
            if mask.sum() > 0:
                cluster_acc = ((probs_sample[mask] > 0.5) == labels_sample[mask]).mean()
                cluster_survival = labels_sample[mask].mean()
                cluster_pred_rate = probs_sample[mask].mean()
                cluster_size = mask.sum()
                
                cluster_results.append({
                    'cluster': i,
                    'size': cluster_size,
                    'accuracy': cluster_acc,
                    'survival_rate': cluster_survival,
                    'pred_rate': cluster_pred_rate
                })
                
                print(f"    Cluster {i}: {cluster_size:4,} startups | "
                      f"Acc: {cluster_acc:.2%} | "
                      f"Survival: {cluster_survival:.2%}")
        
        self.embedding_results = {
            'pca_variance': explained_variance[:20].tolist(),
            'optimal_clusters': best_k,
            'cluster_results': cluster_results,
            'sample_size': len(embeddings_sample)
        }
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        plt.subplot(3, 4, 1)
        cm = self.performance_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. Probability Distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.probabilities, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Survival Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        
        # 3. Probabilities by True Label
        plt.subplot(3, 4, 3)
        if len(np.unique(self.labels)) > 1:
            died_probs = self.probabilities[self.labels == 0]
            survived_probs = self.probabilities[self.labels == 1]
            
            plt.hist(died_probs, bins=30, alpha=0.7, label='Actually Died', color='red', density=True)
            plt.hist(survived_probs, bins=30, alpha=0.7, label='Actually Survived', color='blue', density=True)
            plt.title('Probabilities by True Outcome')
            plt.xlabel('Survival Probability')
            plt.ylabel('Density')
            plt.legend()
        
        # 4. ROC Curve
        plt.subplot(3, 4, 4)
        if not np.isnan(self.performance_results['auc']):
            fpr, tpr, _ = roc_curve(self.labels, self.probabilities)
            plt.plot(fpr, tpr, label=f'ROC (AUC = {self.performance_results["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        
        # 5. Embedding PCA (sample)
        plt.subplot(3, 4, 5)
        if len(self.embeddings) > 100:
            sample_size = min(2000, len(self.embeddings))
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(self.embeddings[indices])
            
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=self.labels[indices], cmap='RdYlBu', alpha=0.6, s=10)
            plt.colorbar(scatter)
            plt.title('Company Embeddings (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        
        # 6. Confidence Analysis
        plt.subplot(3, 4, 6)
        conf_data = self.performance_results['confidence_analysis']
        labels = ['High (>0.8)', 'Medium (0.4-0.6)', 'Low (<0.2)']
        values = [conf_data['high_conf'], conf_data['med_conf'], conf_data['low_conf']]
        
        plt.bar(labels, values)
        plt.title('Prediction Confidence Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 7. Sequence Length Distribution
        plt.subplot(3, 4, 7)
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        plt.hist(seq_lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Sequence Length Distribution')
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        
        # 8. Performance by Company Age
        plt.subplot(3, 4, 8)
        ages = [m['company_age'] for m in self.metadata if m['company_age'] > 0]
        if len(ages) > 100:
            plt.hist(ages, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Company Age Distribution')
            plt.xlabel('Company Age (years)')
            plt.ylabel('Count')
        
        # 9. Prediction Window Analysis
        plt.subplot(3, 4, 9)
        windows = [m['prediction_window'] for m in self.metadata if m['prediction_window'] > 0]
        if len(windows) > 0:
            unique_windows, counts = np.unique(windows, return_counts=True)
            plt.bar(unique_windows, counts)
            plt.title('Prediction Window Distribution')
            plt.xlabel('Prediction Window')
            plt.ylabel('Count')
        
        # 10. PCA Explained Variance
        plt.subplot(3, 4, 10)
        if self.embedding_results:
            variance = self.embedding_results['pca_variance']
            plt.plot(range(1, len(variance)+1), np.cumsum(variance), 'bo-')
            plt.title('PCA Cumulative Explained Variance')
            plt.xlabel('Principal Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.grid(True)
        
        # 11. Cluster Analysis
        plt.subplot(3, 4, 11)
        if self.embedding_results and 'cluster_results' in self.embedding_results:
            cluster_results = self.embedding_results['cluster_results']
            cluster_ids = [r['cluster'] for r in cluster_results]
            cluster_accs = [r['accuracy'] for r in cluster_results]
            
            plt.bar(cluster_ids, cluster_accs)
            plt.title('Accuracy by Cluster')
            plt.xlabel('Cluster ID')
            plt.ylabel('Accuracy')
        
        # 12. Model Calibration
        plt.subplot(3, 4, 12)
        # Create calibration plot
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (self.probabilities > bin_lower) & (self.probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = self.labels[in_bin].mean()
                avg_confidence_in_bin = self.probabilities[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        if bin_confidences and bin_accuracies:
            plt.plot(bin_confidences, bin_accuracies, 'bo-', label='Model')
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            plt.title(f'Calibration Plot (ECE: {ece:.3f})')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, 'comprehensive_interpretability.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comprehensive visualizations saved to {viz_path}")
        
        # Save as PDF too for high quality
        pdf_path = os.path.join(self.output_dir, 'comprehensive_interpretability.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"✅ High-quality PDF saved to {pdf_path}")
        
        plt.close()
    
    def save_complete_results(self):
        """Save all results for further analysis"""
        print(f"\\n💾 SAVING COMPLETE RESULTS")
        print("="*40)
        
        # Create complete interpretability dataset
        self.interpretability_data = {
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'true_labels': self.labels,
            'embeddings': self.embeddings,
            'sequences': self.sequences,
            'metadata': self.metadata,
            'performance_results': self.performance_results,
            'embedding_results': self.embedding_results,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_checkpoint': self.checkpoint_path,
            'total_samples': len(self.predictions)
        }
        
        # Save main results
        results_path = os.path.join(self.output_dir, 'complete_interpretability_data.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.interpretability_data, f)
        print(f"✅ Complete data saved to {results_path}")
        
        # Save DataFrame for easy analysis
        df = pd.DataFrame({
            'prediction': self.predictions,
            'probability': self.probabilities,
            'true_label': self.labels,
            'sequence_length': [m['sequence_length'] for m in self.metadata],
            'prediction_window': [m['prediction_window'] for m in self.metadata],
            'company_age': [m['company_age'] for m in self.metadata],
            'sequence_id': [m['sequence_id'] for m in self.metadata],
        })
        
        df_path = os.path.join(self.output_dir, 'interpretability_results.csv')
        df.to_csv(df_path, index=False)
        print(f"✅ CSV results saved to {df_path}")
        
        # Save summary report
        self._save_summary_report()
    
    def _save_intermediate_results(self, predictions, probabilities, labels, embeddings, sequences, metadata, batch_idx):
        """Save intermediate results during processing"""
        intermediate_data = {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'true_labels': np.array(labels),
            'embeddings': np.array(embeddings),
            'sequences': sequences,
            'metadata': metadata,
            'batch_processed': batch_idx,
            'partial_data': True
        }
        
        intermediate_path = os.path.join(self.output_dir, f'intermediate_batch_{batch_idx}.pkl')
        with open(intermediate_path, 'wb') as f:
            pickle.dump(intermediate_data, f)
    
    def _save_summary_report(self):
        """Save a comprehensive summary report"""
        report_path = os.path.join(self.output_dir, 'interpretability_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE STARTUP2VEC INTERPRETABILITY ANALYSIS\\n")
            f.write("=" * 80 + "\\n\\n")
            
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Model Checkpoint: {self.checkpoint_path}\\n")
            f.write(f"Total Samples Analyzed: {len(self.predictions):,}\\n\\n")
            
            # Performance Summary
            f.write("📊 MODEL PERFORMANCE SUMMARY\\n")
            f.write("-" * 40 + "\\n")
            perf = self.performance_results
            f.write(f"Overall Accuracy: {perf['accuracy']:.4f}\\n")
            f.write(f"AUC-ROC: {perf['auc']:.4f}\\n")
            f.write(f"Actual Survival Rate: {perf['survival_rate']:.2%}\\n")
            f.write(f"Predicted Survival Rate: {perf['pred_survival_rate']:.2%}\\n\\n")
            
            # Confusion Matrix
            f.write("Confusion Matrix:\\n")
            cm = perf['confusion_matrix']
            f.write(f"[[{cm[0,0]:6}, {cm[0,1]:6}],\\n")
            f.write(f" [{cm[1,0]:6}, {cm[1,1]:6}]]\\n\\n")
            
            # Embedding Analysis
            if self.embedding_results:
                f.write("🧠 EMBEDDING ANALYSIS\\n")
                f.write("-" * 40 + "\\n")
                emb = self.embedding_results
                f.write(f"Embedding Dimension: {self.embeddings.shape[1]}\\n")
                f.write(f"Optimal Clusters: {emb['optimal_clusters']}\\n")
                f.write(f"PCA Variance (first 5): {sum(emb['pca_variance'][:5]):.1%}\\n\\n")
            
            # Key Insights
            f.write("💡 KEY INSIGHTS\\n")
            f.write("-" * 40 + "\\n")
            
            if perf['survival_rate'] > 0.95:
                f.write("1. Dataset is heavily imbalanced with very high survival rate\\n")
                f.write("2. Model performance should be evaluated with this context\\n")
            
            if not np.isnan(perf['auc']) and perf['auc'] > 0.6:
                f.write("3. Model shows decent discriminative ability despite class imbalance\\n")
            
            if self.embedding_results and self.embedding_results['optimal_clusters'] > 1:
                f.write("4. Startup embeddings form distinct clusters\\n")
            
            f.write("\\n📁 OUTPUT FILES\\n")
            f.write("-" * 40 + "\\n")
            f.write("- complete_interpretability_data.pkl: All analysis data\\n")
            f.write("- interpretability_results.csv: Results in CSV format\\n")
            f.write("- comprehensive_interpretability.png/pdf: Visualizations\\n")
            f.write("- interpretability_summary_report.txt: This summary\\n")
        
        print(f"✅ Summary report saved to {report_path}")
    
    def run_complete_analysis(self, max_batches=None):
        """Run the complete interpretability analysis pipeline"""
        print("🚀 COMPREHENSIVE STARTUP2VEC INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        print("Full-scale analysis with model evaluation and interpretability")
        print()
        
        # Step 1: Load model and data
        if not self.load_model_and_data():
            print("❌ Failed to load model and data")
            return False
        
        # Step 2: Extract validation data
        print(f"\\nStep 2: Extracting validation data...")
        if not self.extract_full_validation_data(max_batches=max_batches):
            print("❌ Failed to extract validation data")
            return False
        
        # Step 3: Analyze performance
        print(f"\\nStep 3: Analyzing model performance...")
        self.analyze_model_performance()
        
        # Step 4: Analyze embeddings
        print(f"\\nStep 4: Analyzing embedding space...")
        self.analyze_embeddings()
        
        # Step 5: Create visualizations
        print(f"\\nStep 5: Creating comprehensive visualizations...")
        self.create_comprehensive_visualizations()
        
        # Step 6: Save all results
        print(f"\\nStep 6: Saving complete results...")
        self.save_complete_results()
        
        print(f"\\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Successfully analyzed {len(self.predictions):,} startup samples")
        print(f"📁 All results saved to '{self.output_dir}' directory")
        print(f"🎯 Check interpretability_summary_report.txt for overview")
        
        return True

def main():
    """Main function to run comprehensive interpretability analysis"""
    
    # Configuration
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    output_dir = "comprehensive_interpretability_results"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not os.path.exists(pretrained_path):
        print(f"❌ Pretrained model not found: {pretrained_path}")
        return 1
    
    # Create analyzer
    analyzer = StartupInterpretabilityAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir=output_dir
    )
    
    # Ask user for batch limit
    print("🔧 CONFIGURATION")
    print("="*40)
    print("How many batches to process?")
    print("  - 100: Quick test (~3,200 samples)")
    print("  - 1000: Medium analysis (~32,000 samples)")
    print("  - 5000: Large analysis (~160,000 samples)")
    print("  - ALL: Complete dataset (~1.5M samples, may take hours)")
    
    choice = input("Enter number or 'ALL': ").strip().upper()
    
    if choice == 'ALL':
        max_batches = None
        print("🎯 Running complete analysis on full dataset")
    else:
        try:
            max_batches = int(choice)
            print(f"🎯 Running analysis on {max_batches} batches")
        except ValueError:
            print("❌ Invalid input, using default 1000 batches")
            max_batches = 1000
    
    # Run analysis
    try:
        success = analyzer.run_complete_analysis(max_batches=max_batches)
        
        if success:
            print("\\n✅ SUCCESS! Comprehensive interpretability analysis completed")
            return 0
        else:
            print("\\n❌ Analysis failed")
            return 1
            
    except KeyboardInterrupt:
        print("\\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
