# interp_03_visual_exploration.py
#!/usr/bin/env python3
"""
STARTUP2VEC VISUAL EXPLORATION - Script 3/5
Visual exploration with startup arch of life visualizations and GPU support
"""

import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
import gc
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not available. Install with: pip install umap-learn")

# Add LIME/SHAP imports and a function for local explainability
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    LIME_SHAP_AVAILABLE = True
except ImportError:
    LIME_SHAP_AVAILABLE = False
    print("[WARN] LIME/SHAP not installed. Install with: pip install shap lime")

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class StartupVisualExplorer:
    """Visual exploration for startup survival predictions"""
    
    def __init__(self, checkpoint_path, pretrained_path, output_dir="visual_exploration_results"):
        self.checkpoint_path = checkpoint_path
        self.pretrained_path = pretrained_path
        self.output_dir = output_dir
        self.model = None
        self.datamodule = None
        
        # Core data
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.embeddings = None
        self.sequences = None
        self.metadata = None
        
        # Vocabulary
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
        # Visualization results
        self.embedding_2d = None
        self.embedding_3d = None
        self.clusters = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_and_data(self):
        """Load model and data with GPU memory management"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        try:
            # Use the fixed model
            from models.survival_model import FixedStartupSurvivalModel
            self.model = FixedStartupSurvivalModel.load_from_checkpoint(
                self.checkpoint_path,
                pretrained_model_path=self.pretrained_path,
                map_location='cpu'
            )
            self.model.eval()
            print("✅ Model loaded successfully")
            from dataloaders.survival_datamodule import SurvivalDataModule
            self.datamodule = SurvivalDataModule(
                corpus_name="startup_corpus",
                vocab_name="startup_vocab",
                batch_size=16,
                num_workers=1,
                prediction_windows=[1, 2, 3, 4]
            )
            self.datamodule.setup()
            print("✅ Datamodule loaded successfully")
            self._extract_vocabulary()
            return True
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            return False
    
    def _extract_vocabulary(self):
        """Extract vocabulary for visualization"""
        try:
            if hasattr(self.datamodule, 'vocabulary'):
                self.vocab_to_idx = self.datamodule.vocabulary.token2index
                self.idx_to_vocab = self.datamodule.vocabulary.index2token
                print(f"✅ Vocabulary extracted: {len(self.vocab_to_idx):,} tokens")
            else:
                print("⚠️ Could not extract vocabulary")
        except Exception as e:
            print(f"⚠️ Vocabulary parsing failed: {e}")
    
    def extract_data_for_visualization(self, target_batches=500, balanced_sampling=False):
        """Extract data specifically for visualization"""
        print(f"\n🎯 EXTRACTING DATA FOR VISUAL EXPLORATION")
        print("="*60)
        
        if balanced_sampling:
            return self._extract_balanced_data_for_viz(target_batches)
        else:
            return self._extract_standard_data_for_viz(target_batches)
    
    def _extract_standard_data_for_viz(self, target_batches):
        """Standard data extraction for visualization"""
        val_loader = self.datamodule.val_dataloader()
        max_batches = min(target_batches, len(val_loader)) if target_batches > 0 else len(val_loader)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        all_sequences = []
        all_metadata = []
        
        # GPU handling
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔍 Using device: {device}")
        
        try:
            if torch.cuda.is_available():
                self.clear_cuda_cache()
            self.model = self.model.to(device)
            print(f"✅ Model loaded to {device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ CUDA OOM! Falling back to CPU...")
                self.clear_cuda_cache()
                device = 'cpu'
                self.model = self.model.to(device)
            else:
                raise e
        
        print(f"Processing {max_batches:,} batches...")
        successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if target_batches > 0 and batch_idx >= max_batches:
                    break
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{max_batches} (successful: {successful_batches})", end='\r')
                    
                    if device == 'cuda' and batch_idx % 100 == 0:
                        self.clear_cuda_cache()
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    outputs = self.model.forward(input_ids=input_ids, padding_mask=padding_mask)
                    
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]
                    
                    # Store results
                    all_predictions.extend(survival_preds.cpu().numpy())
                    all_probabilities.extend(survival_probs.cpu().numpy())
                    all_labels.extend(survival_labels.squeeze().cpu().numpy())
                    all_embeddings.extend(company_embeddings.cpu().numpy())
                    all_sequences.extend(input_ids[:, 0, :].cpu().numpy())
                    
                    # Extract metadata for visualization
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_viz_metadata(batch, i, input_ids[i, 0, :])
                        all_metadata.append(metadata)
                    
                    successful_batches += 1
                    
                    # Clear GPU memory
                    del input_ids, padding_mask, survival_labels, outputs
                    del survival_logits, survival_probs, survival_preds, transformer_output, company_embeddings
                    
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ CUDA OOM at batch {batch_idx}, continuing...")
                        self.clear_cuda_cache()
                        continue
                    else:
                        print(f"\nError in batch {batch_idx}: {e}")
                        continue
        
        print(f"\n✅ Data extraction complete: {len(all_predictions):,} samples")
        
        if len(all_predictions) == 0:
            return False
        
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.labels = np.array(all_labels)
        self.embeddings = np.array(all_embeddings)
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        return True
    
    def _extract_balanced_data_for_viz(self, target_batches):
        """Extract balanced data for visualization"""
        val_loader = self.datamodule.val_dataloader()
        
        survival_data = {'predictions': [], 'probabilities': [], 'labels': [], 
                        'embeddings': [], 'sequences': [], 'metadata': []}
        failure_data = {'predictions': [], 'probabilities': [], 'labels': [], 
                       'embeddings': [], 'sequences': [], 'metadata': []}
        
        target_per_class = target_batches * 8
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            if torch.cuda.is_available():
                self.clear_cuda_cache()
            self.model = self.model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                device = 'cpu'
                self.model = self.model.to(device)
        
        print(f"Collecting balanced samples for visualization (target: {target_per_class} per class)...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if (len(survival_data['labels']) >= target_per_class and 
                    len(failure_data['labels']) >= target_per_class):
                    break
                
                if batch_idx % 50 == 0:
                    survived = len(survival_data['labels'])
                    failed = len(failure_data['labels'])
                    print(f"  Batch {batch_idx}: {survived} survived, {failed} failed", end='\r')
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    outputs = self.model.forward(input_ids=input_ids, padding_mask=padding_mask)
                    
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]
                    
                    for i in range(input_ids.size(0)):
                        true_label = survival_labels[i].squeeze().item()
                        
                        sample_data = {
                            'prediction': survival_preds[i].cpu().numpy(),
                            'probability': survival_probs[i].cpu().numpy(),
                            'label': true_label,
                            'embedding': company_embeddings[i].cpu().numpy(),
                            'sequence': input_ids[i, 0, :].cpu().numpy(),
                            'metadata': self._extract_viz_metadata(batch, i, input_ids[i, 0, :])
                        }
                        
                        if true_label == 1 and len(survival_data['labels']) < target_per_class:
                            for key in survival_data.keys():
                                survival_data[key].append(sample_data[key.rstrip('s')])
                        elif true_label == 0 and len(failure_data['labels']) < target_per_class:
                            for key in failure_data.keys():
                                failure_data[key].append(sample_data[key.rstrip('s')])
                    
                    del input_ids, padding_mask, survival_labels, outputs
                    
                except Exception as e:
                    continue
        
        # Combine balanced data
        min_samples = min(len(survival_data['labels']), len(failure_data['labels']))
        print(f"\n✅ Balanced sampling complete: {min_samples} per class")
        
        if min_samples == 0:
            return False
        
        self.predictions = np.concatenate([
            np.array(survival_data['predictions'][:min_samples]),
            np.array(failure_data['predictions'][:min_samples])
        ])
        
        self.probabilities = np.concatenate([
            np.array(survival_data['probabilities'][:min_samples]),
            np.array(failure_data['probabilities'][:min_samples])
        ])
        
        self.labels = np.concatenate([
            np.array(survival_data['labels'][:min_samples]),
            np.array(failure_data['labels'][:min_samples])
        ])
        
        self.embeddings = np.vstack([
            np.array(survival_data['embeddings'][:min_samples]),
            np.array(failure_data['embeddings'][:min_samples])
        ])
        
        self.sequences = (survival_data['sequences'][:min_samples] + 
                         failure_data['sequences'][:min_samples])
        
        self.metadata = (survival_data['metadata'][:min_samples] + 
                        failure_data['metadata'][:min_samples])
        
        return True
    
    def _extract_viz_metadata(self, batch, sample_idx, sequence):
        """Extract metadata for visualization"""
        try:
            base_metadata = {
                'sample_idx': sample_idx,
                'sequence_length': (sequence > 0).sum().item(),
                'company_age': batch['company_age_at_prediction'][sample_idx].item() if 'company_age_at_prediction' in batch else 2,
            }
            
            # Parse characteristics for visualization
            viz_characteristics = self._parse_viz_characteristics(sequence)
            base_metadata.update(viz_characteristics)
            
            return base_metadata
        except Exception as e:
            return {
                'sample_idx': sample_idx, 'sequence_length': 0, 'company_age': 2,
                'country': 'Unknown', 'industry': 'Unknown', 'size_category': 'Unknown',
                'funding_stage': 'Unknown', 'region': 'Unknown'
            }
    
    def _parse_viz_characteristics(self, sequence):
        """Parse characteristics specifically for visualization"""
        characteristics = {
            'country': 'Unknown', 'industry': 'Unknown', 'size_category': 'Unknown',
            'funding_stage': 'Unknown', 'region': 'Unknown'
        }
        
        try:
            clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
            
            for token_id in clean_sequence:
                token_str = self.idx_to_vocab.get(int(token_id), "")
                
                # Country and region
                if token_str.startswith('COUNTRY_'):
                    characteristics['country'] = token_str.replace('COUNTRY_', '')
                    # Map to regions for visualization
                    if characteristics['country'] in ['USA', 'CAN']:
                        characteristics['region'] = 'North America'
                    elif characteristics['country'] in ['GBR', 'DEU', 'FRA', 'ESP', 'ITA']:
                        characteristics['region'] = 'Europe'
                    elif characteristics['country'] in ['IND', 'CHN', 'JPN', 'SGP']:
                        characteristics['region'] = 'Asia'
                    else:
                        characteristics['region'] = 'Other'
                
                # Industry
                elif token_str.startswith('INDUSTRY_'):
                    characteristics['industry'] = token_str.replace('INDUSTRY_', '')
                elif token_str.startswith('CATEGORY_'):
                    characteristics['industry'] = token_str.replace('CATEGORY_', '')
                
                # Company size
                elif token_str.startswith('EMPLOYEE_'):
                    employee_str = token_str.replace('EMPLOYEE_', '')
                    if '1-10' in employee_str:
                        characteristics['size_category'] = 'Micro'
                    elif '11-50' in employee_str:
                        characteristics['size_category'] = 'Small'
                    elif '51-' in employee_str:
                        characteristics['size_category'] = 'Medium'
                    else:
                        characteristics['size_category'] = 'Large'
                
                # Funding stage
                elif 'INV_' in token_str and 'TYPE_' in token_str:
                    if any(stage in token_str.lower() for stage in ['seed', 'angel']):
                        characteristics['funding_stage'] = 'Early'
                    elif any(stage in token_str.lower() for stage in ['series_a', 'series_b']):
                        characteristics['funding_stage'] = 'Growth'
                    elif any(stage in token_str.lower() for stage in ['series_c', 'late']):
                        characteristics['funding_stage'] = 'Late'
        
        except Exception as e:
            pass
        
        return characteristics
    
    def create_embeddings_projections(self):
        """Create 2D and 3D projections of embeddings"""
        print("\n🗺️ Creating embedding projections...")
        
        projections = {}
        
        # 1. PCA projection (always available)
        print("  📊 Creating PCA projection...")
        pca_2d = PCA(n_components=2, random_state=42)
        pca_embedding_2d = pca_2d.fit_transform(self.embeddings)
        projections['pca_2d'] = pca_embedding_2d
        
        pca_3d = PCA(n_components=3, random_state=42)
        pca_embedding_3d = pca_3d.fit_transform(self.embeddings)
        projections['pca_3d'] = pca_embedding_3d
        
        # 2. TSNE projection
        print("  🎯 Creating t-SNE projection...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)//4))
            tsne_embedding_2d = tsne.fit_transform(self.embeddings)
            projections['tsne_2d'] = tsne_embedding_2d
        except Exception as e:
            print(f"    ⚠️ t-SNE failed: {e}")
        
        # 3. UMAP projection (if available)
        if UMAP_AVAILABLE:
            print("  🌐 Creating UMAP projection...")
            try:
                umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                umap_embedding_2d = umap_2d.fit_transform(self.embeddings)
                projections['umap_2d'] = umap_embedding_2d
                
                umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
                umap_embedding_3d = umap_3d.fit_transform(self.embeddings)
                projections['umap_3d'] = umap_embedding_3d
                
                # Use UMAP as primary 2D projection
                self.embedding_2d = umap_embedding_2d
                self.embedding_3d = umap_embedding_3d
            except Exception as e:
                print(f"    ⚠️ UMAP failed: {e}")
                # Fallback to PCA
                self.embedding_2d = pca_embedding_2d
                self.embedding_3d = pca_embedding_3d
        else:
            # Use PCA as fallback
            self.embedding_2d = pca_embedding_2d
            self.embedding_3d = pca_embedding_3d
        
        print(f"    ✅ Created {len(projections)} embedding projections")
        return projections
    
    def create_startup_arch_of_life(self):
        """Create the main 'Startup Arch of Life' visualization"""
        print("\n🏗️ Creating Startup Arch of Life visualization...")
        
        try:
            # Create comprehensive arch visualization
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            fig.suptitle('STARTUP ARCH OF LIFE - Comprehensive View', fontsize=18, fontweight='bold', y=0.95)
            
            # Plot 1: Survival Probability
            scatter1 = axes[0, 0].scatter(
                self.embedding_2d[:, 0], self.embedding_2d[:, 1],
                c=self.probabilities, cmap='RdYlGn', alpha=0.7, s=20
            )
            axes[0, 0].set_title('Survival Probability', fontsize=14, fontweight='bold')
            plt.colorbar(scatter1, ax=axes[0, 0], label='Probability')
            axes[0, 0].set_xlabel('Dimension 1')
            axes[0, 0].set_ylabel('Dimension 2')
            
            # Plot 2: True Labels
            colors = ['#ff4444' if label == 0 else '#44ff44' for label in self.labels]
            axes[0, 1].scatter(
                self.embedding_2d[:, 0], self.embedding_2d[:, 1],
                c=colors, alpha=0.7, s=20
            )
            axes[0, 1].set_title('True Labels', fontsize=14, fontweight='bold')
            axes[0, 1].text(0.02, 0.98, 'Red=Failed, Green=Survived', transform=axes[0, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            axes[0, 1].set_xlabel('Dimension 1')
            axes[0, 1].set_ylabel('Dimension 2')
            
            # Plot 3: Industry Categories
            industries = [m.get('industry', 'Unknown') for m in self.metadata]
            unique_industries = [ind for ind in Counter(industries).most_common(8)]  # Top 8 industries
            industry_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_industries)))
            
            for i, (industry, count) in enumerate(unique_industries):
                mask = np.array([ind == industry for ind in industries])
                if mask.sum() > 0:
                    axes[0, 2].scatter(
                        self.embedding_2d[mask, 0], self.embedding_2d[mask, 1],
                        c=[industry_colors[i]], label=f'{industry} ({count})', alpha=0.7, s=15
                    )
            axes[0, 2].set_title('Industry Categories', fontsize=14, fontweight='bold')
            axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0, 2].set_xlabel('Dimension 1')
            axes[0, 2].set_ylabel('Dimension 2')
            
            # Plot 4: Company Size
            sizes = [m.get('size_category', 'Unknown') for m in self.metadata]
            size_color_map = {'Micro': '#1f77b4', 'Small': '#ff7f0e', 'Medium': '#2ca02c', 'Large': '#d62728', 'Unknown': '#7f7f7f'}
            
            for size_cat, color in size_color_map.items():
                mask = np.array([s == size_cat for s in sizes])
                if mask.sum() > 0:
                    axes[1, 0].scatter(
                        self.embedding_2d[mask, 0], self.embedding_2d[mask, 1],
                        c=color, label=f'{size_cat} ({mask.sum()})', alpha=0.7, s=15
                    )
            axes[1, 0].set_title('Company Size Categories', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].set_xlabel('Dimension 1')
            axes[1, 0].set_ylabel('Dimension 2')
            
            # Plot 5: Funding Stage
            funding_stages = [m.get('funding_stage', 'Unknown') for m in self.metadata]
            stage_color_map = {'Early': '#add8e6', 'Growth': '#4169e1', 'Late': '#000080', 'Unknown': '#7f7f7f'}
            
            for stage, color in stage_color_map.items():
                mask = np.array([s == stage for s in funding_stages])
                if mask.sum() > 0:
                    axes[1, 1].scatter(
                        self.embedding_2d[mask, 0], self.embedding_2d[mask, 1],
                        c=color, label=f'{stage} ({mask.sum()})', alpha=0.7, s=15
                    )
            axes[1, 1].set_title('Funding Stages', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].set_xlabel('Dimension 1')
            axes[1, 1].set_ylabel('Dimension 2')
            
            # Plot 6: Geographic Regions
            regions = [m.get('region', 'Unknown') for m in self.metadata]
            region_color_map = {
                'North America': '#ff0000', 'Europe': '#0000ff', 'Asia': '#00ff00', 
                'Other': '#ff8c00', 'Unknown': '#7f7f7f'
            }
            
            for region, color in region_color_map.items():
                mask = np.array([r == region for r in regions])
                if mask.sum() > 0:
                    axes[1, 2].scatter(
                        self.embedding_2d[mask, 0], self.embedding_2d[mask, 1],
                        c=color, label=f'{region} ({mask.sum()})', alpha=0.7, s=15
                    )
            axes[1, 2].set_title('Geographic Regions', fontsize=14, fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].set_xlabel('Dimension 1')
            axes[1, 2].set_ylabel('Dimension 2')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save comprehensive arch visualization
            arch_path = os.path.join(self.output_dir, "plots", "startup_arch_of_life.png")
            plt.savefig(arch_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Startup Arch of Life saved to: {arch_path}")
            
            return {
                'arch_path': arch_path,
                'industry_distribution': Counter(industries),
                'size_distribution': Counter(sizes),
                'funding_distribution': Counter(funding_stages),
                'region_distribution': Counter(regions)
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not create Startup Arch visualization: {e}")
            return {}
    
    def create_clustering_analysis(self):
        """Create clustering analysis visualization"""
        print("\n🔗 Creating clustering analysis...")
        
        try:
            # Perform clustering
            n_clusters = min(8, max(3, len(np.unique(self.labels)) * 4))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            self.clusters = cluster_labels
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Plot clusters
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                if mask.sum() > 0:
                    cluster_survival_rate = self.labels[mask].mean()
                    
                    plt.scatter(
                        self.embedding_2d[mask, 0], self.embedding_2d[mask, 1],
                        c=[colors[cluster_id]], 
                        label=f'Cluster {cluster_id} (n={mask.sum()}, SR={cluster_survival_rate:.2f})',
                        alpha=0.7, s=25
                    )
            
            plt.title('Startup Clustering Analysis\n(Clusters with survival rates)', fontsize=16, fontweight='bold')
            plt.xlabel('Dimension 1', fontsize=12)
            plt.ylabel('Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save plot
            cluster_path = os.path.join(self.output_dir, "plots", "clustering_analysis.png")
            plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Clustering analysis saved to: {cluster_path}")
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                if mask.sum() > 0:
                    cluster_stats[f'cluster_{cluster_id}'] = {
                        'count': mask.sum(),
                        'survival_rate': self.labels[mask].mean(),
                        'avg_probability': self.probabilities[mask].mean(),
                        'dominant_industry': Counter([self.metadata[i]['industry'] for i in np.where(mask)[0] if i < len(self.metadata)]).most_common(1)[0][0] if mask.sum() > 0 else 'Unknown'
                    }
            
            return {
                'cluster_path': cluster_path,
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not create clustering analysis: {e}")
            return {}
    
    def create_survival_heatmap(self):
        """Create survival probability heatmap"""
        print("\n🔥 Creating survival probability heatmap...")
        
        try:
            # Create grid for heatmap
            x_min, x_max = self.embedding_2d[:, 0].min(), self.embedding_2d[:, 0].max()
            y_min, y_max = self.embedding_2d[:, 1].min(), self.embedding_2d[:, 1].max()
            
            # Extend bounds slightly
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            # Create meshgrid
            grid_resolution = 50
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, grid_resolution),
                np.linspace(y_min, y_max, grid_resolution)
            )
            
            # Calculate survival probability for each grid point using nearest neighbors
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(20, len(self.embedding_2d)), metric='euclidean')
            nn.fit(self.embedding_2d)
            
            distances, indices = nn.kneighbors(grid_points)
            
            # Calculate weighted average survival probability
            grid_survival = np.zeros(len(grid_points))
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                # Use inverse distance weighting
                weights = 1 / (dists + 1e-8)  # Add small epsilon
                weights = weights / weights.sum()
                grid_survival[i] = np.average(self.probabilities[idxs], weights=weights)
            
            grid_survival = grid_survival.reshape(xx.shape)
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            
            # Plot heatmap
            heatmap = plt.contourf(xx, yy, grid_survival, levels=20, cmap='RdYlGn', alpha=0.8)
            plt.colorbar(heatmap, label='Survival Probability')
            
            # Overlay actual points
            scatter = plt.scatter(
                self.embedding_2d[:, 0], self.embedding_2d[:, 1],
                c=self.probabilities, cmap='RdYlGn', 
                s=15, alpha=0.6, edgecolors='black', linewidth=0.5
            )
            
            plt.title('Startup Survival Probability Heatmap\n(Interpolated survival probability across embedding space)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Dimension 1', fontsize=12)
            plt.ylabel('Dimension 2', fontsize=12)
            
            # Save plot
            heatmap_path = os.path.join(self.output_dir, "plots", "survival_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Survival heatmap saved to: {heatmap_path}")
            
            return {
                'heatmap_path': heatmap_path,
                'grid_survival': grid_survival,
                'grid_coordinates': (xx, yy)
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not create survival heatmap: {e}")
            return {}
    
    def create_3d_visualization(self):
        """Create 3D visualization of embedding space"""
        print("\n🌐 Creating 3D visualization...")
        
        try:
            # Create 3D plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color by survival probability
            scatter = ax.scatter(
                self.embedding_3d[:, 0], 
                self.embedding_3d[:, 1], 
                self.embedding_3d[:, 2],
                c=self.probabilities,
                cmap='RdYlGn',
                alpha=0.6,
                s=25
            )
            
            plt.colorbar(scatter, label='Survival Probability', shrink=0.5)
            ax.set_title('Startup Embedding Space - 3D View\n(Colored by Survival Probability)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            
            # Save plot
            viz_3d_path = os.path.join(self.output_dir, "plots", "embedding_3d.png")
            plt.savefig(viz_3d_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ 3D visualization saved to: {viz_3d_path}")
            
            return {
                '3d_path': viz_3d_path,
                'embedding_3d': self.embedding_3d
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not create 3D visualization: {e}")
            return {}
    
    def create_comparison_plots(self):
        """Create comparison plots for different projection methods"""
        print("\n📊 Creating projection comparison plots...")
        
        try:
            projections = self.create_embeddings_projections()
            
            # Create comparison figure
            n_projections = len(projections)
            if n_projections == 0:
                return {}
            
            # Filter to 2D projections only
            proj_2d = {k: v for k, v in projections.items() if '2d' in k}
            n_proj_2d = len(proj_2d)
            
            if n_proj_2d == 0:
                return {}
            
            fig, axes = plt.subplots(1, n_proj_2d, figsize=(6*n_proj_2d, 6))
            if n_proj_2d == 1:
                axes = [axes]
            
            fig.suptitle('Embedding Projection Comparison\n(All colored by survival probability)', 
                        fontsize=16, fontweight='bold')
            
            for idx, (method, embedding) in enumerate(proj_2d.items()):
                scatter = axes[idx].scatter(
                    embedding[:, 0], embedding[:, 1],
                    c=self.probabilities, cmap='RdYlGn', alpha=0.7, s=20
                )
                axes[idx].set_title(f'{method.upper().replace("_", " ")}', fontsize=14)
                axes[idx].set_xlabel('Dimension 1')
                axes[idx].set_ylabel('Dimension 2')
                plt.colorbar(scatter, ax=axes[idx], label='Survival Probability')
            
            plt.tight_layout()
            
            # Save comparison plot
            comparison_path = os.path.join(self.output_dir, "plots", "projection_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Projection comparison saved to: {comparison_path}")
            
            return {
                'comparison_path': comparison_path,
                'projections': projections
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not create comparison plots: {e}")
            return {}
    
    def run_visual_exploration(self):
        """Run comprehensive visual exploration analysis"""
        print("\n" + "="*70)
        print("🎨 VISUAL EXPLORATION ANALYSIS")
        print("="*70)
        
        exploration_results = {}
        
        # 1. Create embedding projections
        print("\n🗺️ Creating embedding projections...")
        projections = self.create_embeddings_projections()
        exploration_results['projections'] = projections
        
        # 2. Create Startup Arch of Life (main visualization)
        arch_result = self.create_startup_arch_of_life()
        exploration_results['startup_arch'] = arch_result
        
        # 3. Create clustering analysis
        clustering_result = self.create_clustering_analysis()
        exploration_results['clustering'] = clustering_result
        
        # 4. Create survival heatmap
        heatmap_result = self.create_survival_heatmap()
        exploration_results['heatmap'] = heatmap_result
        
        # 5. Create 3D visualization
        viz_3d_result = self.create_3d_visualization()
        exploration_results['3d_viz'] = viz_3d_result
        
        # 6. Create comparison plots
        comparison_result = self.create_comparison_plots()
        exploration_results['comparison'] = comparison_result
        
        # Save results
        self._save_exploration_results(exploration_results)
        
        return exploration_results
    
    def _save_exploration_results(self, exploration_results):
        """Save visual exploration results"""
        # Save as pickle
        results_path = os.path.join(self.output_dir, "visual_exploration_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({
                'exploration_results': exploration_results,
                'predictions': self.predictions,
                'probabilities': self.probabilities,
                'labels': self.labels,
                'embeddings': self.embeddings,
                'embedding_2d': self.embedding_2d,
                'embedding_3d': self.embedding_3d,
                'clusters': self.clusters,
                'metadata': self.metadata
            }, f)
        
        # Save as text report
        report_path = os.path.join(self.output_dir, "visual_exploration_report.txt")
        with open(report_path, 'w') as f:
            f.write("STARTUP2VEC VISUAL EXPLORATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples analyzed: {len(self.predictions):,}\n")
            f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VISUALIZATIONS CREATED:\n")
            f.write("-" * 30 + "\n")
            
            if 'startup_arch' in exploration_results and exploration_results['startup_arch']:
                f.write(f"✅ Startup Arch of Life: {exploration_results['startup_arch'].get('arch_path', 'N/A')}\n")
            
            if 'clustering' in exploration_results and exploration_results['clustering']:
                f.write(f"✅ Clustering Analysis: {exploration_results['clustering'].get('cluster_path', 'N/A')}\n")
                if 'cluster_stats' in exploration_results['clustering']:
                    f.write("\nCluster Statistics:\n")
                    for cluster_id, stats in exploration_results['clustering']['cluster_stats'].items():
                        f.write(f"  {cluster_id}: {stats['count']} companies, "
                               f"survival rate: {stats['survival_rate']:.3f}, "
                               f"dominant industry: {stats['dominant_industry']}\n")
            
            if 'heatmap' in exploration_results and exploration_results['heatmap']:
                f.write(f"✅ Survival Heatmap: {exploration_results['heatmap'].get('heatmap_path', 'N/A')}\n")
            
            if '3d_viz' in exploration_results and exploration_results['3d_viz']:
                f.write(f"✅ 3D Visualization: {exploration_results['3d_viz'].get('3d_path', 'N/A')}\n")
            
            if 'comparison' in exploration_results and exploration_results['comparison']:
                f.write(f"✅ Projection Comparison: {exploration_results['comparison'].get('comparison_path', 'N/A')}\n")
            
            f.write("\nDISTRIBUTIONS:\n")
            f.write("-" * 20 + "\n")
            
            if 'startup_arch' in exploration_results and exploration_results['startup_arch']:
                arch_data = exploration_results['startup_arch']
                if 'industry_distribution' in arch_data:
                    f.write("Top Industries:\n")
                    for industry, count in arch_data['industry_distribution'].most_common(5):
                        f.write(f"  {industry}: {count}\n")
                
                if 'region_distribution' in arch_data:
                    f.write("\nRegions:\n")
                    for region, count in arch_data['region_distribution'].items():
                        f.write(f"  {region}: {count}\n")
            
        print(f"\n✅ Visual exploration results saved to:")
        print(f"  📊 Data: {results_path}")
        print(f"  📋 Report: {report_path}")
        print(f"  🎨 Plots: {self.output_dir}/plots/")
    
    def run_complete_exploration(self, target_batches=500, balanced_sampling=False):
        """Run complete visual exploration analysis"""
        print("🚀 STARTUP2VEC VISUAL EXPLORATION")
        print("=" * 80)
        print("Visual exploration with startup arch of life and comprehensive visualizations")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data for visualization
        if not self.extract_data_for_visualization(target_batches, balanced_sampling):
            return False
        
        # Run visual exploration
        exploration_results = self.run_visual_exploration()
        
        print(f"\n🎉 VISUAL EXPLORATION COMPLETE!")
        print(f"📁 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to: {self.output_dir}/")
        print(f"🎨 Visualizations saved to: {self.output_dir}/plots/")
        
        return exploration_results

def explain_with_lime_shap(model, dataloader, num_samples=10):
    if not LIME_SHAP_AVAILABLE:
        print("LIME/SHAP not available.")
        return
    print("\n[INFO] Running LIME/SHAP explanations on a few samples...")
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    padding_mask = batch['padding_mask']
    sample_input = input_ids[0].cpu().numpy().flatten()
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.long).view(1, *input_ids.shape[1:])
        with torch.no_grad():
            logits = model(input_ids=x_tensor, padding_mask=padding_mask[0:1])['survival_logits']
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    explainer = LimeTabularExplainer([sample_input], mode='classification')
    exp = explainer.explain_instance(sample_input, predict_fn, num_features=10)
    print("LIME explanation:")
    print(exp.as_list())
    explainer_shap = shap.Explainer(predict_fn, [sample_input])
    shap_values = explainer_shap([sample_input])
    print("SHAP values:")
    print(shap_values.values)

def main():
    """Main function for visual exploration"""
    print("🔧 STARTUP2VEC VISUAL EXPLORATION")
    print("="*70)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 CUDA Available: {gpu_count} GPU(s)")
    else:
        print("❌ CUDA not available - will use CPU")
    print()
    print("🎯 VISUAL EXPLORATION FEATURES:")
    print("✅ Startup Arch of Life visualization (6-panel comprehensive view)")
    print("✅ Multiple embedding projections (PCA, t-SNE, UMAP if available)")
    print("✅ Clustering analysis with survival rates")
    print("✅ Survival probability heatmap")
    print("✅ 3D embedding space visualization")
    print("✅ Projection method comparison")
    print("✅ GPU memory management with CPU fallback")
    print("✅ Balanced sampling option")
    print()
    explorer = StartupVisualExplorer(
        checkpoint_path="survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt",
        pretrained_path="startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt",
        output_dir="visual_exploration_results"
    )
    # Always run on the full dataset (all batches)
    target_batches = 0
    balanced_sampling = False
    # Run analysis
    start_time = time.time()
    success = explorer.run_complete_exploration(
        target_batches=target_batches,
        balanced_sampling=balanced_sampling
    )
    end_time = time.time()
    if success:
        print(f"\n🎉 SUCCESS! Visual exploration completed in {end_time-start_time:.1f} seconds")
        print("\n🎨 VISUALIZATIONS CREATED:")
        print("  • startup_arch_of_life.png - Main comprehensive view")
        print("  • clustering_analysis.png - Startup clusters with survival rates")
        print("  • survival_heatmap.png - Interpolated survival probability")
        print("  • embedding_3d.png - 3D embedding space")
        print("  • projection_comparison.png - Different projection methods")
        print("\n💡 NEXT STEPS:")
        print("  1. View the plots in visual_exploration_results/plots/")
        print("  2. Check visual_exploration_report.txt for analysis summary")
        print("  3. Use visual_exploration_results.pkl for further analysis")
        print("  4. Run the next script: 04_local_explainability.py")
        return 0
    else:
        print(f"\n❌ Visual exploration failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
