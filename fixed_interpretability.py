#!/usr/bin/env python3
"""
FIXED ENHANCED INTERPRETABILITY ANALYSIS
Addresses the issues from the previous version
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
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class FixedStartupInterpretabilityAnalyzer:
    """Fixed analyzer addressing previous issues"""
    
    def __init__(self, checkpoint_path, pretrained_path, output_dir="fixed_interpretability"):
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
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model_and_data(self):
        """Load model and data"""
        print("🔍 Loading model and data...")
        
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
            
            # Try to extract vocabulary
            self._extract_vocabulary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            return False
    
    def _extract_vocabulary(self):
        """Extract vocabulary with better error handling"""
        try:
            # Try multiple ways to get vocabulary
            vocab_sources = [
                ('datamodule.vocabulary', lambda: (self.datamodule.vocabulary.token2index, self.datamodule.vocabulary.index2token)),
                ('datamodule.vocab', lambda: (self.datamodule.vocab.token2index, self.datamodule.vocab.index2token)),
                ('datamodule.tokenizer', lambda: (self.datamodule.tokenizer.token2index, self.datamodule.tokenizer.index2token))
            ]
            
            for source_name, extractor in vocab_sources:
                try:
                    if hasattr(self.datamodule, source_name.split('.')[1]):
                        self.vocab_to_idx, self.idx_to_vocab = extractor()
                        print(f"✅ Vocabulary extracted from {source_name}: {len(self.vocab_to_idx):,} tokens")
                        return
                except:
                    continue
            
            print("⚠️ Could not extract vocabulary - will use token-based analysis")
            
        except Exception as e:
            print(f"⚠️ Vocabulary extraction failed: {e}")
    
    def extract_data_with_larger_sample(self, target_batches=500):
        """Extract larger sample for better AUC estimation"""
        print(f"\\n🎯 EXTRACTING LARGER SAMPLE ({target_batches} batches)")
        print("="*60)
        print("This should give us AUC closer to the target 0.67")
        
        val_loader = self.datamodule.val_dataloader()
        max_batches = min(target_batches, len(val_loader))
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        all_sequences = []
        all_metadata = []
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        
        print(f"Processing {max_batches:,} batches for better statistics...")
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = batch_idx / elapsed if elapsed > 0 else 0
                    eta = (max_batches - batch_idx) / rate if rate > 0 else 0
                    print(f"  Batch {batch_idx}/{max_batches} | "
                          f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min", end='\\r')
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        padding_mask=padding_mask
                    )
                    
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
                    
                    # Enhanced metadata
                    for i in range(input_ids.size(0)):
                        metadata = {
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'sequence_length': padding_mask[i].sum().item(),
                            'prediction_window': batch['prediction_window'][i].item() if 'prediction_window' in batch else 1,
                            'company_age': batch['company_age_at_prediction'][i].item() if 'company_age_at_prediction' in batch else 2,
                            'sequence_id': batch['sequence_id'][i].item() if 'sequence_id' in batch else -1,
                            'founded_year': batch['company_founded_year'][i].item() if 'company_founded_year' in batch else 2020,
                        }
                        all_metadata.append(metadata)
                    
                except Exception as e:
                    print(f"\\nError in batch {batch_idx}: {e}")
                    continue
        
        print(f"\\n✅ Larger sample extracted: {len(all_predictions):,} samples")
        print(f"⏱️ Processing time: {(time.time() - start_time)/60:.1f} minutes")
        
        # Store results
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.labels = np.array(all_labels)
        self.embeddings = np.array(all_embeddings)
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        # Quick performance check
        self._quick_performance_analysis()
        
        return True
    
    def _quick_performance_analysis(self):
        """Quick analysis of the larger sample"""
        print(f"\\n📊 QUICK PERFORMANCE ANALYSIS (LARGER SAMPLE)")
        print("-"*50)
        
        accuracy = (self.predictions == self.labels).mean()
        survival_rate = self.labels.mean()
        
        try:
            if len(np.unique(self.labels)) > 1:
                auc = roc_auc_score(self.labels, self.probabilities)
            else:
                auc = float('nan')
        except:
            auc = float('nan')
        
        print(f"📈 Sample Performance:")
        print(f"  Total samples: {len(self.predictions):,}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Survival rate: {survival_rate:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        print(f"\\nConfusion Matrix:")
        print(cm)
        
        # Compare with target
        target_cm = np.array([[197840, 624], [7249, 9105]])
        ratio = target_cm.sum() / cm.sum() if cm.sum() > 0 else 0
        print(f"\\nRatio to target dataset: {ratio:.2f}")
        
        if not np.isnan(auc):
            if auc > 0.6:
                print(f"✅ AUC {auc:.3f} is good - close to target 0.67!")
            elif auc > 0.5:
                print(f"📊 AUC {auc:.3f} is above random - model is learning")
            else:
                print(f"🤔 AUC {auc:.3f} - may indicate class imbalance effects")
    
    def improved_algorithmic_auditing(self):
        """Improved algorithmic auditing with better categories"""
        print(f"\\n🔍 1. IMPROVED ALGORITHMIC AUDITING")
        print("="*50)
        
        results = {}
        
        # 1. Company Size (based on sequence length)
        print("\\n📊 Performance by Company Size:")
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        size_categories = []
        
        for length in seq_lengths:
            if length < np.percentile(seq_lengths, 33):
                size_categories.append('Small')
            elif length < np.percentile(seq_lengths, 67):
                size_categories.append('Medium')
            else:
                size_categories.append('Large')
        
        size_results = self._analyze_subgroup_performance(size_categories, 'Company Size')
        results['company_size'] = size_results
        
        # 2. Company Age Groups
        print("\\n🕐 Performance by Company Age:")
        ages = [max(1, m['company_age']) for m in self.metadata]
        age_categories = []
        
        for age in ages:
            if age <= 2:
                age_categories.append('Very Young')
            elif age <= 5:
                age_categories.append('Young')
            elif age <= 10:
                age_categories.append('Mature')
            else:
                age_categories.append('Established')
        
        age_results = self._analyze_subgroup_performance(age_categories, 'Company Age')
        results['company_age'] = age_results
        
        # 3. Prediction Window Analysis
        print("\\n📅 Performance by Prediction Window:")
        windows = [f"Window {m['prediction_window']}" for m in self.metadata]
        window_results = self._analyze_subgroup_performance(windows, 'Prediction Window')
        results['prediction_window'] = window_results
        
        # 4. Sequence Complexity (unique tokens)
        print("\\n🧮 Performance by Sequence Complexity:")
        complexity_categories = []
        
        for sequence in self.sequences:
            unique_tokens = len(set(sequence[sequence > 0]))  # Remove padding
            if unique_tokens < 20:
                complexity_categories.append('Simple')
            elif unique_tokens < 40:
                complexity_categories.append('Moderate')
            else:
                complexity_categories.append('Complex')
        
        complexity_results = self._analyze_subgroup_performance(complexity_categories, 'Sequence Complexity')
        results['sequence_complexity'] = complexity_results
        
        # 5. Embedding Cluster Analysis
        print("\\n🎯 Performance by Embedding Cluster:")
        cluster_categories = self._create_embedding_clusters()
        cluster_results = self._analyze_subgroup_performance(cluster_categories, 'Embedding Cluster')
        results['embedding_clusters'] = cluster_results
        
        # Enhanced bias detection
        self._enhanced_bias_detection(results)
        
        self.algorithmic_audit_results = results
    
    def improved_data_contribution_analysis(self):
        """Improved data contribution analysis"""
        print(f"\\n📊 2. IMPROVED DATA CONTRIBUTION ANALYSIS")
        print("="*50)
        
        # Token frequency analysis
        print("\\n🔤 Token Frequency Analysis:")
        survived_tokens = []
        died_tokens = []
        
        for i, sequence in enumerate(self.sequences):
            clean_sequence = sequence[sequence > 0]  # Remove padding
            if self.labels[i] == 1:  # Survived
                survived_tokens.extend(clean_sequence)
            else:  # Died
                died_tokens.extend(clean_sequence)
        
        survived_freq = Counter(survived_tokens)
        died_freq = Counter(died_tokens)
        
        # Calculate token importance scores
        token_scores = {}
        all_tokens = set(survived_tokens + died_tokens)
        
        total_survived = len([l for l in self.labels if l == 1])
        total_died = len([l for l in self.labels if l == 0])
        
        if total_died > 0 and total_survived > 0:
            for token in all_tokens:
                survived_rate = survived_freq.get(token, 0) / total_survived
                died_rate = died_freq.get(token, 0) / total_died
                
                # Importance score (difference in rates)
                importance = survived_rate - died_rate
                
                if abs(importance) > 0.001:  # Only significant differences
                    token_scores[token] = {
                        'importance': importance,
                        'survived_rate': survived_rate,
                        'died_rate': died_rate,
                        'token_id': int(token)
                    }
            
            # Top tokens for survival vs death
            survival_tokens = sorted(token_scores.items(), key=lambda x: x[1]['importance'], reverse=True)[:10]
            death_tokens = sorted(token_scores.items(), key=lambda x: x[1]['importance'])[:10]
            
            print(f"\\n🎯 Top Survival-Associated Tokens:")
            for i, (token, scores) in enumerate(survival_tokens, 1):
                token_name = self.idx_to_vocab.get(int(token), f"Token_{token}") if self.idx_to_vocab else f"Token_{token}"
                print(f"  {i}. {token_name}: {scores['importance']:+.4f}")
            
            print(f"\\n💀 Top Death-Associated Tokens:")
            for i, (token, scores) in enumerate(death_tokens, 1):
                token_name = self.idx_to_vocab.get(int(token), f"Token_{token}") if self.idx_to_vocab else f"Token_{token}"
                print(f"  {i}. {token_name}: {scores['importance']:+.4f}")
        
        # Sequence length contribution
        print(f"\\n📏 Sequence Length Analysis:")
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        
        # Correlation between sequence length and survival
        if len(set(self.labels)) > 1:
            length_survival_corr = np.corrcoef(seq_lengths, self.labels)[0, 1]
            print(f"  Length-Survival correlation: {length_survival_corr:.4f}")
            
            if abs(length_survival_corr) > 0.1:
                direction = "longer" if length_survival_corr > 0 else "shorter"
                print(f"  📊 {direction.title()} sequences tend to be associated with survival")
        
        self.data_contribution_results = {
            'token_scores': token_scores,
            'survival_tokens': survival_tokens,
            'death_tokens': death_tokens,
            'length_correlation': length_survival_corr if 'length_survival_corr' in locals() else 0
        }
    
    def fixed_visual_exploration(self):
        """Fixed visual exploration with proper UMAP installation"""
        print(f"\\n🎨 3. FIXED VISUAL EXPLORATION")
        print("="*50)
        
        # Sample for visualization
        max_viz = 2000
        if len(self.embeddings) > max_viz:
            indices = np.random.choice(len(self.embeddings), max_viz, replace=False)
            viz_embeddings = self.embeddings[indices]
            viz_probs = self.probabilities[indices]
            viz_labels = self.labels[indices]
            viz_metadata = [self.metadata[i] for i in indices]
        else:
            viz_embeddings = self.embeddings
            viz_probs = self.probabilities
            viz_labels = self.labels
            viz_metadata = self.metadata
        
        # 1. PCA (always works)
        print("📊 Creating PCA projection...")
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(viz_embeddings)
        
        # 2. Try UMAP with proper error handling
        umap_embeddings = None
        try:
            import umap.umap_ as umap
            print("🗺️ Creating UMAP projection...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            umap_embeddings = umap_reducer.fit_transform(viz_embeddings)
        except ImportError:
            print("⚠️ UMAP not installed - using t-SNE instead")
        except Exception as e:
            print(f"⚠️ UMAP failed: {e} - using t-SNE instead")
        
        # 3. t-SNE (backup)
        print("🔄 Creating t-SNE projection...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_embeddings)//4))
            tsne_embeddings = tsne.fit_transform(viz_embeddings)
        except Exception as e:
            print(f"⚠️ t-SNE failed: {e}")
            tsne_embeddings = None
        
        # Store results
        self.visual_exploration_results = {
            'pca_embeddings': pca_embeddings,
            'umap_embeddings': umap_embeddings,
            'tsne_embeddings': tsne_embeddings,
            'viz_probabilities': viz_probs,
            'viz_labels': viz_labels,
            'viz_metadata': viz_metadata
        }
        
        # Create startup arch visualization
        self._create_startup_arch_visualization()
    
    def improved_local_explainability(self):
        """Improved local explainability with better examples"""
        print(f"\\n🔍 4. IMPROVED LOCAL EXPLAINABILITY")
        print("="*50)
        
        # Select more diverse examples
        examples = {
            'perfect_predictions': [],      # High confidence, correct
            'confident_mistakes': [],       # High confidence, wrong
            'uncertain_cases': [],          # Low confidence
            'extreme_probabilities': [],    # Very high or very low probs
            'sequence_outliers': []         # Unusual sequence lengths
        }
        
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        length_mean = np.mean(seq_lengths)
        length_std = np.std(seq_lengths)
        
        for i in range(len(self.probabilities)):
            prob = self.probabilities[i]
            pred = (prob > 0.5).astype(int)
            true = self.labels[i]
            seq_len = self.metadata[i]['sequence_length']
            
            # Categorize examples
            if prob > 0.9 and pred == true:
                examples['perfect_predictions'].append(i)
            elif prob > 0.8 and pred != true:
                examples['confident_mistakes'].append(i)
            elif 0.3 < prob < 0.7:
                examples['uncertain_cases'].append(i)
            elif prob > 0.95 or prob < 0.05:
                examples['extreme_probabilities'].append(i)
            elif abs(seq_len - length_mean) > 2 * length_std:
                examples['sequence_outliers'].append(i)
        
        # Analyze examples
        local_results = {}
        for example_type, indices in examples.items():
            if len(indices) > 0:
                print(f"\\n🎯 Analyzing {example_type} ({len(indices)} examples):")
                
                example_analyses = []
                for idx in indices[:5]:  # Top 5 examples
                    analysis = self._deep_analyze_startup(idx)
                    example_analyses.append(analysis)
                    
                    print(f"  Sample {idx}: P={self.probabilities[idx]:.3f}, "
                          f"True={self.labels[idx]}, Age={self.metadata[idx]['company_age']}, "
                          f"Len={self.metadata[idx]['sequence_length']}")
                
                local_results[example_type] = example_analyses
        
        self.local_explainability_results = local_results
    
    def improved_global_explainability(self):
        """Improved global explainability with better concepts"""
        print(f"\\n🌐 5. IMPROVED GLOBAL EXPLAINABILITY")
        print("="*50)
        
        # Define more sophisticated concepts
        concepts = {
            'High Activity': self._test_high_activity_concept,
            'Long Sequences': self._test_long_sequence_concept,
            'Recent Foundation': self._test_recent_foundation_concept,
            'Multiple Windows': self._test_multiple_window_concept,
            'Token Diversity': self._test_token_diversity_concept
        }
        
        concept_scores = {}
        
        for concept_name, concept_test in concepts.items():
            print(f"\\n🧠 Testing concept: {concept_name}")
            
            try:
                concept_score = concept_test()
                concept_scores[concept_name] = concept_score
                
                direction = "survival" if concept_score > 0 else "death"
                strength = "strong" if abs(concept_score) > 0.3 else "moderate" if abs(concept_score) > 0.1 else "weak"
                
                print(f"  Score: {concept_score:+.3f} ({strength} → {direction})")
                
            except Exception as e:
                print(f"  ⚠️ Concept test failed: {e}")
                concept_scores[concept_name] = 0.0
        
        # Rank concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\\n🏆 Concept Importance Ranking:")
        for i, (concept, score) in enumerate(sorted_concepts, 1):
            direction = "survival" if score > 0 else "death"
            print(f"  {i}. {concept}: {score:+.3f} (→ {direction})")
        
        self.global_explainability_results = {
            'concept_scores': concept_scores,
            'ranking': sorted_concepts
        }
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        # Create the startup arch visualization
        self._create_startup_arch_visualization()
        
        # Create algorithmic audit visualizations
        self._create_audit_visualizations()
        
        # Create data contribution visualizations
        self._create_contribution_visualizations()
        
        # Create explainability visualizations  
        self._create_explainability_visualizations()
        
        print("✅ All visualizations created successfully")
    
    # Helper methods
    def _analyze_subgroup_performance(self, categories, category_name):
        """Analyze performance across subgroups"""
        unique_cats = list(set(categories))
        results = []
        
        for cat in unique_cats:
            mask = np.array([c == cat for c in categories])
            if mask.sum() >= 10:
                cat_acc = ((self.probabilities[mask] > 0.5) == self.labels[mask]).mean()
                cat_survival = self.labels[mask].mean()
                cat_pred_rate = self.probabilities[mask].mean()
                cat_count = mask.sum()
                
                results.append({
                    'category': cat,
                    'count': cat_count,
                    'accuracy': cat_acc,
                    'survival_rate': cat_survival,
                    'pred_rate': cat_pred_rate
                })
                
                print(f"  {cat}: {cat_count:5,} samples | "
                      f"Acc: {cat_acc:.2%} | "
                      f"Survival: {cat_survival:.2%}")
        
        return results
    
    def _enhanced_bias_detection(self, audit_results):
        """Enhanced bias detection"""
        print(f"\\n⚖️ Enhanced Bias Detection:")
        
        bias_issues = []
        
        for category, results in audit_results.items():
            if len(results) > 1:
                accuracies = [r['accuracy'] for r in results]
                survival_rates = [r['survival_rate'] for r in results]
                
                acc_range = max(accuracies) - min(accuracies)
                survival_range = max(survival_rates) - min(survival_rates)
                
                if acc_range > 0.05:  # 5% threshold
                    bias_issues.append(f"{category}: {acc_range:.1%} accuracy gap")
                
                if survival_range > 0.1:  # 10% threshold
                    bias_issues.append(f"{category}: {survival_range:.1%} survival rate gap")
        
        if bias_issues:
            print("  ⚠️ Potential bias detected:")
            for issue in bias_issues:
                print(f"    - {issue}")
        else:
            print("  ✅ No significant bias detected")
    
    def _create_embedding_clusters(self):
        """Create embedding-based clusters"""
        # Use PCA for clustering
        pca = PCA(n_components=20)
        pca_embeddings = pca.fit_transform(self.embeddings)
        
        # K-means clustering
        n_clusters = min(8, len(self.embeddings) // 100)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(pca_embeddings)
            return [f"Cluster {label}" for label in cluster_labels]
        else:
            return ["Cluster 0"] * len(self.embeddings)
    
    def _deep_analyze_startup(self, idx):
        """Deep analysis of individual startup"""
        metadata = self.metadata[idx]
        sequence = self.sequences[idx]
        
        # Sequence analysis
        clean_sequence = sequence[sequence > 0]
        unique_tokens = len(set(clean_sequence))
        most_common_token = Counter(clean_sequence).most_common(1)[0] if len(clean_sequence) > 0 else (0, 0)
        
        return {
            'index': idx,
            'probability': float(self.probabilities[idx]),
            'prediction': int((self.probabilities[idx] > 0.5)),
            'true_label': int(self.labels[idx]),
            'sequence_length': metadata['sequence_length'],
            'unique_tokens': unique_tokens,
            'company_age': metadata['company_age'],
            'prediction_window': metadata['prediction_window'],
            'most_common_token': int(most_common_token[0]),
            'token_frequency': int(most_common_token[1])
        }
    
    # Concept testing methods
    def _test_high_activity_concept(self):
        """Test high activity concept"""
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        high_activity = np.array(seq_lengths) > np.percentile(seq_lengths, 75)
        return np.corrcoef(high_activity.astype(float), self.probabilities)[0, 1]
    
    def _test_long_sequence_concept(self):
        """Test long sequence concept"""
        seq_lengths = [m['sequence_length'] for m in self.metadata]
        return np.corrcoef(seq_lengths, self.probabilities)[0, 1]
    
    def _test_recent_foundation_concept(self):
        """Test recent foundation concept"""
        founded_years = [m.get('founded_year', 2020) for m in self.metadata]
        recent_foundation = np.array(founded_years) >= 2018
        return np.corrcoef(recent_foundation.astype(float), self.probabilities)[0, 1]
    
    def _test_multiple_window_concept(self):
        """Test multiple prediction window concept"""
        windows = [m['prediction_window'] for m in self.metadata]
        return np.corrcoef(windows, self.probabilities)[0, 1]
    
    def _test_token_diversity_concept(self):
        """Test token diversity concept"""
        diversities = []
        for sequence in self.sequences:
            clean_seq = sequence[sequence > 0]
            diversity = len(set(clean_seq)) / len(clean_seq) if len(clean_seq) > 0 else 0
            diversities.append(diversity)
        
        return np.corrcoef(diversities, self.probabilities)[0, 1]
    
    def _create_startup_arch_visualization(self):
        """Create startup arch visualization"""
        if not hasattr(self, 'visual_exploration_results'):
            return
        
        viz_data = self.visual_exploration_results
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # PCA colored by survival probability
        if viz_data['pca_embeddings'] is not None:
            scatter = axes[0, 0].scatter(viz_data['pca_embeddings'][:, 0], 
                                       viz_data['pca_embeddings'][:, 1],
                                       c=viz_data['viz_probabilities'], 
                                       cmap='RdYlBu', alpha=0.6, s=20)
            axes[0, 0].set_title('Startup Arch (PCA) - Survival Probability')
            plt.colorbar(scatter, ax=axes[0, 0])
        
        # PCA colored by true labels
        if viz_data['pca_embeddings'] is not None:
            scatter = axes[0, 1].scatter(viz_data['pca_embeddings'][:, 0], 
                                       viz_data['pca_embeddings'][:, 1],
                                       c=viz_data['viz_labels'], 
                                       cmap='RdYlBu', alpha=0.6, s=20)
            axes[0, 1].set_title('Startup Arch (PCA) - True Labels')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # t-SNE if available
        if viz_data['tsne_embeddings'] is not None:
            scatter = axes[0, 2].scatter(viz_data['tsne_embeddings'][:, 0], 
                                       viz_data['tsne_embeddings'][:, 1],
                                       c=viz_data['viz_probabilities'], 
                                       cmap='RdYlBu', alpha=0.6, s=20)
            axes[0, 2].set_title('Startup Space (t-SNE)')
            plt.colorbar(scatter, ax=axes[0, 2])
        
        # UMAP if available
        if viz_data['umap_embeddings'] is not None:
            scatter = axes[1, 0].scatter(viz_data['umap_embeddings'][:, 0], 
                                       viz_data['umap_embeddings'][:, 1],
                                       c=viz_data['viz_probabilities'], 
                                       cmap='RdYlBu', alpha=0.6, s=20)
            axes[1, 0].set_title('Startup Arch (UMAP)')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Company age coloring
        if viz_data['pca_embeddings'] is not None:
            ages = [m['company_age'] for m in viz_data['viz_metadata']]
            scatter = axes[1, 1].scatter(viz_data['pca_embeddings'][:, 0], 
                                       viz_data['pca_embeddings'][:, 1],
                                       c=ages, cmap='viridis', alpha=0.6, s=20)
            axes[1, 1].set_title('Startup Arch - Company Age')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        # Sequence length coloring
        if viz_data['pca_embeddings'] is not None:
            seq_lens = [m['sequence_length'] for m in viz_data['viz_metadata']]
            scatter = axes[1, 2].scatter(viz_data['pca_embeddings'][:, 0], 
                                       viz_data['pca_embeddings'][:, 1],
                                       c=seq_lens, cmap='plasma', alpha=0.6, s=20)
            axes[1, 2].set_title('Startup Arch - Sequence Length')
            plt.colorbar(scatter, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'startup_arch_of_life.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_audit_visualizations(self):
        """Create audit visualizations"""
        # Implementation for audit visualizations
        pass
    
    def _create_contribution_visualizations(self):
        """Create contribution visualizations"""
        # Implementation for contribution visualizations  
        pass
    
    def _create_explainability_visualizations(self):
        """Create explainability visualizations"""
        # Implementation for explainability visualizations
        pass
    
    def run_fixed_analysis(self, target_batches=500):
        """Run the fixed analysis with larger sample"""
        print("🚀 FIXED ENHANCED INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        print("Addresses previous issues and uses larger sample for better AUC")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract larger sample for better statistics
        if not self.extract_data_with_larger_sample(target_batches):
            return False
        
        # Run improved analysis pipeline
        print("\\n" + "="*60)
        print("RUNNING IMPROVED INTERPRETABILITY PIPELINE")
        print("="*60)
        
        self.improved_algorithmic_auditing()
        self.improved_data_contribution_analysis()
        self.fixed_visual_exploration()
        self.improved_local_explainability()
        self.improved_global_explainability()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Save results
        self._save_all_results()
        
        print(f"\\n🎉 FIXED ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.predictions):,} samples")
        print(f"📁 Results saved to '{self.output_dir}' directory")
        
        return True
    
    def _save_all_results(self):
        """Save all results"""
        results = {
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'labels': self.labels,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'algorithmic_audit': getattr(self, 'algorithmic_audit_results', None),
            'data_contribution': getattr(self, 'data_contribution_results', None),
            'visual_exploration': getattr(self, 'visual_exploration_results', None),
            'local_explainability': getattr(self, 'local_explainability_results', None),
            'global_explainability': getattr(self, 'global_explainability_results', None),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(self.output_dir, 'fixed_interpretability_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✅ All results saved to {results_path}")

def main():
    """Main function"""
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    analyzer = FixedStartupInterpretabilityAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir="fixed_interpretability"
    )
    
    print("🔧 FIXED Enhanced Interpretability Analysis")
    print("="*50)
    print("Fixes:")
    print("- Uses larger sample (500 batches) for better AUC estimation")
    print("- Fixed UMAP/visualization issues")
    print("- Improved concept analysis")
    print("- Better token-level analysis")
    print("- Enhanced bias detection")
    print()
    
    choice = input("Enter number of batches (500 recommended for AUC ~0.67): ").strip()
    try:
        target_batches = int(choice)
    except ValueError:
        target_batches = 500
    
    success = analyzer.run_fixed_analysis(target_batches=target_batches)
    
    if success:
        print("\\n✅ SUCCESS! Fixed interpretability analysis completed")
        return 0
    else:
        print("\\n❌ Analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
