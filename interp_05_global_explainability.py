#!/usr/bin/env python3
"""
STARTUP2VEC GLOBAL EXPLAINABILITY - Script 5/5
Global explainability analysis using TCAV-inspired concept analysis
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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Add LIME/SHAP imports and a function for local explainability
# try:
#     import shap
#     from lime.lime_tabular import LimeTabularExplainer
#     LIME_SHAP_AVAILABLE = True
# except ImportError:
#     LIME_SHAP_AVAILABLE = False
#     print("[WARN] LIME/SHAP not installed. Install with: pip install shap lime")

# Use correct model and datamodule imports
from models.survival_model import FixedStartupSurvivalModel
from dataloaders.survival_datamodule import SurvivalDataModule

# COLOR PALETTE FOR ALL CONCEPT VISUALIZATIONS (use consistently throughout the paper)
# ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
# Assign in this order to: high_growth, tech_focused, well_funded, team_focused, b2b_oriented, consumer_focused, early_stage

# --- Color palette for concepts ---
CONCEPT_COLOR_PALETTE = [
    "#e6194b",  # high_growth
    "#3cb44b",  # tech_focused
    "#ffe119",  # well_funded
    "#4363d8",  # team_focused
    "#f58231",  # b2b_oriented
    "#911eb4",  # consumer_focused
    "#46f0f0",  # early_stage
]
CONCEPT_ORDER = [
    'high_growth', 'tech_focused', 'well_funded', 'team_focused',
    'b2b_oriented', 'consumer_focused', 'early_stage'
]
CONCEPT_COLOR_MAP = {concept: CONCEPT_COLOR_PALETTE[i] for i, concept in enumerate(CONCEPT_ORDER)}

class StartupGlobalExplainer:
    """Global explainability analysis for startup survival predictions using TCAV-inspired methods"""
    
    def __init__(self, checkpoint_path, output_dir="global_explainability_results"):
        self.checkpoint_path = checkpoint_path
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
        
        # Global explainability results
        self.concepts = None
        self.concept_vectors = None
        self.tcav_scores = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "concept_analysis"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tcav_results"), exist_ok=True)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_and_data(self):
        """Load model and data with GPU memory management"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        
        try:
            # Load model to CPU first
            print("Loading model from:", self.checkpoint_path)
            model = FixedStartupSurvivalModel.load_from_checkpoint(self.checkpoint_path, map_location='cpu')
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            self.model = model
            print(f"✅ Model loaded successfully to {device}")
            # Load datamodule (batch_size=128, num_workers=4)
            self.datamodule = SurvivalDataModule(
                corpus_name="startup_corpus",
                vocab_name="startup_vocab",
                batch_size=128,
                num_workers=4,
                prediction_windows=[1, 2, 3, 4]
            )
            self.datamodule.setup()
            print("✅ Datamodule loaded successfully")
            
            # Extract vocabulary
            self._extract_vocabulary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            return False
    
    def _extract_vocabulary(self):
        """Extract vocabulary for concept analysis"""
        try:
            if hasattr(self.datamodule, 'vocabulary'):
                self.vocab_to_idx = self.datamodule.vocabulary.token2index
                self.idx_to_vocab = self.datamodule.vocabulary.index2token
                print(f"✅ Vocabulary extracted: {len(self.vocab_to_idx):,} tokens")
            else:
                print("⚠️ Could not extract vocabulary")
        except Exception as e:
            print(f"⚠️ Vocabulary parsing failed: {e}")
    
    def extract_data_for_concepts(self, target_batches=500, balanced_sampling=False):
        """Extract data for concept analysis"""
        print(f"\n🎯 EXTRACTING DATA FOR GLOBAL CONCEPT ANALYSIS")
        print("="*60)
        
        if balanced_sampling:
            return self._extract_balanced_data_for_concepts(target_batches)
        else:
            return self._extract_standard_data_for_concepts(target_batches)
    
    def _extract_standard_data_for_concepts(self, target_batches):
        """Standard data extraction for concept analysis"""
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
        
        print(f"Processing {max_batches:,} batches for concept analysis...")
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
                    
                    # Extract metadata for concept analysis
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_concept_metadata(batch, i, input_ids[i, 0, :])
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
    
    def _extract_balanced_data_for_concepts(self, target_batches):
        """Extract balanced data for concept analysis"""
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
        
        print(f"Collecting balanced samples for concept analysis (target: {target_per_class} per class)...")
        
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
                            'metadata': self._extract_concept_metadata(batch, i, input_ids[i, 0, :])
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
        print(f"\n✅ Balanced sampling for concepts complete: {min_samples} per class")
        
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
    
    def _extract_concept_metadata(self, batch, sample_idx, sequence):
        """Extract metadata for concept analysis"""
        try:
            base_metadata = {
                'sample_idx': sample_idx,
                'sequence_length': (sequence > 0).sum().item(),
                'company_age': batch['company_age_at_prediction'][sample_idx].item() if 'company_age_at_prediction' in batch else 2,
            }
            
            # Parse characteristics for concept analysis
            concept_characteristics = self._parse_concept_characteristics(sequence)
            base_metadata.update(concept_characteristics)
            
            return base_metadata
        except Exception as e:
            return {
                'sample_idx': sample_idx, 'sequence_length': 0, 'company_age': 2,
                'is_high_growth': False, 'is_tech_focused': False, 'is_well_funded': False,
                'is_team_focused': False, 'is_b2b': False,
                'is_consumer': False, 'is_early_stage': False
            }
    
    def _parse_concept_characteristics(self, sequence):
        """Parse characteristics for concept analysis"""
        characteristics = {
            'is_high_growth': False, 'is_tech_focused': False, 'is_well_funded': False,
            'is_team_focused': False, 'is_b2b': False,
            'is_consumer': False, 'is_early_stage': False
        }
        
        try:
            clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
            
            funding_count = 0
            tech_count = 0
            team_count = 0
            
            for token_id in clean_sequence:
                token_str = self.idx_to_vocab.get(int(token_id), "")
                
                # High growth indicators
                if any(keyword in token_str.lower() for keyword in ['series_a', 'series_b', 'series_c', 'growth']):
                    characteristics['is_high_growth'] = True
                
                # Tech focused
                if any(keyword in token_str.lower() for keyword in ['tech', 'ai', 'ml', 'software', 'platform', 'api']):
                    tech_count += 1
                
                # Well funded
                if any(keyword in token_str.lower() for keyword in ['investment', 'funding', 'raised']):
                    funding_count += 1
                
                # Team focused
                if any(keyword in token_str.lower() for keyword in ['employee', 'hire', 'team', 'people']):
                    team_count += 1
                
                # B2B vs Consumer
                if any(keyword in token_str.lower() for keyword in ['b2b', 'enterprise', 'business']):
                    characteristics['is_b2b'] = True
                elif any(keyword in token_str.lower() for keyword in ['b2c', 'consumer', 'retail']):
                    characteristics['is_consumer'] = True
                
                # Early stage
                if any(keyword in token_str.lower() for keyword in ['seed', 'angel', 'pre_series']):
                    characteristics['is_early_stage'] = True
            
            # Set flags based on counts
            characteristics['is_tech_focused'] = tech_count >= 2
            characteristics['is_well_funded'] = funding_count >= 3
            characteristics['is_team_focused'] = team_count >= 2
            
        except Exception as e:
            pass
        
        return characteristics
    
    def define_startup_concepts(self):
        """Define abstract concepts for startup analysis"""
        print("\n🧠 Defining abstract startup concepts...")
        
        concepts = {
            'high_growth': {
                'description': 'Startups with rapid growth indicators (multiple funding rounds)',
                'keywords': ['series_a', 'series_b', 'series_c', 'growth', 'scale'],
                'characteristic_key': 'is_high_growth'
            },
            'tech_focused': {
                'description': 'Technology-focused startups (AI, ML, software platforms)',
                'keywords': ['tech', 'ai', 'ml', 'software', 'platform', 'api'],
                'characteristic_key': 'is_tech_focused'
            },
            'well_funded': {
                'description': 'Well-funded startups with substantial investment',
                'keywords': ['investment', 'funding', 'raised', 'venture', 'capital'],
                'characteristic_key': 'is_well_funded'
            },
            'team_focused': {
                'description': 'Startups with emphasis on team building and hiring',
                'keywords': ['employee', 'hire', 'team', 'people', 'talent'],
                'characteristic_key': 'is_team_focused'
            },
            'b2b_oriented': {
                'description': 'Business-to-business focused startups',
                'keywords': ['b2b', 'enterprise', 'business', 'corporate'],
                'characteristic_key': 'is_b2b'
            },
            'consumer_focused': {
                'description': 'Consumer-focused startups',
                'keywords': ['b2c', 'consumer', 'retail', 'marketplace'],
                'characteristic_key': 'is_consumer'
            },
            'early_stage': {
                'description': 'Early-stage startups (seed, angel funding)',
                'keywords': ['seed', 'angel', 'pre_series', 'startup'],
                'characteristic_key': 'is_early_stage'
            }
        }
        
        self.concepts = concepts
        
        print(f"    📋 Defined {len(concepts)} concepts:")
        for concept_name, concept_data in concepts.items():
            print(f"      {concept_name}: {concept_data['description']}")
        
        return concepts
    
    def compute_concept_vectors(self):
        """Compute concept activation vectors for each concept"""
        print("\n🎯 Computing concept activation vectors...")
        
        if not self.concepts:
            print("    ⚠️ No concepts defined")
            return {}
        
        concept_vectors = {}
        
        for concept_name, concept_data in self.concepts.items():
            try:
                # Find startups that match this concept
                characteristic_key = concept_data['characteristic_key']
                concept_mask = np.array([m.get(characteristic_key, False) for m in self.metadata])
                
                if concept_mask.sum() >= 10:  # Need at least 10 examples
                    # Compute average embedding for concept examples
                    concept_embeddings = self.embeddings[concept_mask]
                    concept_vector = np.mean(concept_embeddings, axis=0)
                    
                    # Compute random direction for comparison (TCAV requirement)
                    random_indices = np.random.choice(len(self.embeddings), 
                                                    size=min(concept_mask.sum(), 200), 
                                                    replace=False)
                    random_embeddings = self.embeddings[random_indices]
                    random_vector = np.mean(random_embeddings, axis=0)
                    
                    concept_vectors[concept_name] = {
                        'concept_vector': concept_vector,
                        'random_vector': random_vector,
                        'n_examples': concept_mask.sum(),
                        'example_indices': np.where(concept_mask)[0],
                        'survival_rate': self.labels[concept_mask].mean()
                    }
                    
                    print(f"    ✅ {concept_name}: {concept_mask.sum()} examples, "
                          f"survival rate: {concept_vectors[concept_name]['survival_rate']:.3f}")
                else:
                    print(f"    ⚠️ {concept_name}: insufficient examples ({concept_mask.sum()})")
                    
            except Exception as e:
                print(f"    ❌ {concept_name}: error computing vector ({e})")
        
        self.concept_vectors = concept_vectors
        return concept_vectors
    
    def compute_tcav_scores(self):
        """Compute TCAV scores for each concept"""
        print("\n📊 Computing TCAV scores...")
        if not self.concept_vectors:
            print("    ⚠️ No concept vectors available")
            return {}
        tcav_scores = {}
        for concept_name, concept_data in self.concept_vectors.items():
            try:
                concept_vector = concept_data['concept_vector']
                random_vector = concept_data['random_vector']
                # Compute directional derivatives (simplified TCAV)
                concept_similarities = []
                random_similarities = []
                for embedding in self.embeddings:
                    concept_sim = 1 - cosine(embedding, concept_vector)
                    random_sim = 1 - cosine(embedding, random_vector)
                    concept_similarities.append(concept_sim)
                    random_similarities.append(random_sim)
                concept_similarities = np.array(concept_similarities)
                random_similarities = np.array(random_similarities)
                tcav_score = np.mean(concept_similarities > random_similarities)
                # Fix: Use only the mask for this concept for survival correlation
                mask = np.array([m.get(self.concepts[concept_name]['characteristic_key'], False) for m in self.metadata])
                if mask.sum() > 0:
                    survival_correlation = np.corrcoef(concept_similarities[mask], self.labels[mask])[0, 1]
                    if np.isnan(survival_correlation):
                        survival_correlation = 0.0
                else:
                    survival_correlation = 0.0
                concept_higher_count = np.sum(concept_similarities > random_similarities)
                total_samples = len(concept_similarities)
                tcav_scores[concept_name] = {
                    'tcav_score': tcav_score,
                    'survival_correlation': survival_correlation,
                    'concept_similarities': concept_similarities,
                    'n_examples': concept_data['n_examples'],
                    'concept_survival_rate': concept_data['survival_rate'],
                    'concept_higher_count': concept_higher_count,
                    'total_samples': total_samples
                }
                print(f"    📊 {concept_name}:")
                print(f"      TCAV Score: {tcav_score:.3f}")
                print(f"      Survival Correlation: {survival_correlation:.3f}")
                print(f"      Concept Examples: {concept_data['n_examples']}")
            except Exception as e:
                print(f"    ❌ {concept_name}: error computing TCAV score ({e})")
        self.tcav_scores = tcav_scores
        return tcav_scores
    
    def rank_concepts_by_importance(self):
        """Rank concepts by importance"""
        print("\n🏆 Ranking concepts by importance...")
        
        if not self.tcav_scores:
            print("    ⚠️ No TCAV scores available")
            return []
        
        try:
            concept_rankings = []
            
            for concept_name, scores in self.tcav_scores.items():
                # Combined importance score
                tcav_score = scores.get('tcav_score', 0)
                survival_corr = abs(scores.get('survival_correlation', 0))
                n_examples = scores.get('n_examples', 0)
                
                # Weight by evidence strength (number of examples)
                evidence_weight = min(1.0, n_examples / 100)
                
                # Combined importance
                importance = (tcav_score * 0.4 + survival_corr * 0.6) * evidence_weight
                
                concept_rankings.append({
                    'concept': concept_name,
                    'importance': importance,
                    'tcav_score': tcav_score,
                    'survival_correlation': survival_corr,
                    'n_examples': n_examples,
                    'concept_survival_rate': scores.get('concept_survival_rate', 0)
                })
            
            # Sort by importance
            concept_rankings.sort(key=lambda x: x['importance'], reverse=True)
            
            print(f"    🏆 Concept Importance Ranking:")
            for i, ranking in enumerate(concept_rankings):
                print(f"      {i+1}. {ranking['concept']}: "
                      f"Importance={ranking['importance']:.3f}, "
                      f"TCAV={ranking['tcav_score']:.3f}, "
                      f"Correlation={ranking['survival_correlation']:.3f}, "
                      f"Examples={ranking['n_examples']}")
            
            return concept_rankings
            
        except Exception as e:
            print(f"    ❌ Error ranking concepts: {e}")
            return []
    
    def create_concept_visualizations(self):
        """Create visualizations for concept analysis"""
        print("\n🎨 Creating concept visualizations...")
        
        viz_results = {}
        
        try:
            # 1. Concept importance plot
            if self.tcav_scores:
                importance_plot = self._create_concept_importance_plot(self.rank_concepts_by_importance())
                viz_results['importance_plot'] = importance_plot
            
            # 2. Concept distribution plot
            if self.concept_vectors:
                distribution_plot = self._create_concept_distribution_plot()
                viz_results['distribution_plot'] = distribution_plot
            
            # 3. TCAV score comparison
            if self.tcav_scores:
                tcav_comparison = self._create_tcav_comparison_plot(self.tcav_scores)
                viz_results['tcav_comparison'] = tcav_comparison
            
        except Exception as e:
            print(f"    ⚠️ Could not create concept visualizations: {e}")
            viz_results = {'error': str(e)}
        
        return viz_results
    
    def _create_concept_importance_plot(self, rankings):
        concepts = [r['concept'] for r in rankings]
        importances = [r['importance'] for r in rankings]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(concepts, importances, color="#4363d8")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_ylabel('Importance')
        ax.set_xlabel('Concept')
        ax.set_title('Startup Concept Importance Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "concept_analysis", "concept_importance.png"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_concept_distribution_plot(self):
        try:
            concept_counts = {}
            for concept_name, concept_data in self.concept_vectors.items():
                concept_counts[concept_name] = {
                    'total_examples': concept_data['n_examples'],
                    'survival_rate': concept_data['survival_rate']
                }
            if not concept_counts:
                return None
            concepts = list(concept_counts.keys())
            total_examples = [concept_counts[c]['total_examples'] for c in concepts]
            survival_rates = [concept_counts[c]['survival_rate'] for c in concepts]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            bars1 = ax1.bar(concepts, total_examples, color="#4363d8")
            for spine in ax1.spines.values():
                spine.set_visible(False)
            ax1.grid(False)
            ax1.set_ylabel('Number of Examples')
            ax1.set_title('Concept Distribution: Number of Examples')
            ax1.tick_params(axis='x', rotation=45)
            for bar, count in zip(bars1, total_examples):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_examples)*0.01,
                        f'{count}', ha='center', va='bottom')
            bars2 = ax2.bar(concepts, survival_rates, color="#4363d8")
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.grid(False)
            ax2.set_ylabel('Survival Rate')
            ax2.set_xlabel('Concepts')
            ax2.set_title('Concept Survival Rates')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1)
            overall_survival_rate = self.labels.mean()
            ax2.axhline(y=overall_survival_rate, color='red', linestyle='--', 
                       label=f'Overall Rate ({overall_survival_rate:.3f})')
            ax2.legend()
            for bar, rate in zip(bars2, survival_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rate:.3f}', ha='center', va='bottom')
            plt.tight_layout()
            distribution_plot_path = os.path.join(self.output_dir, "concept_analysis", "concept_distributions.png")
            plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Concept distribution plot saved to: {distribution_plot_path}")
            return distribution_plot_path
        except Exception as e:
            print(f"      ⚠️ Could not create concept distribution plot: {e}")
            return None
    
    def _create_tcav_comparison_plot(self, tcav_scores):
        try:
            concepts = [c for c in CONCEPT_ORDER if c in tcav_scores]
            tcav_vals = [tcav_scores[c]['survival_correlation'] for c in concepts]
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(concepts, tcav_vals, color="#4363d8")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_ylim(-0.25, 0.05)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel('Survival Correlation')
            ax.set_xlabel('Concept')
            ax.set_title('TCAV Survival Correlation by Concept')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "tcav_results", "tcav_comparison.png"), bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"      ✅ TCAV comparison plot saved to: {os.path.join(self.output_dir, 'tcav_results', 'tcav_comparison.png')}")
            return os.path.join(self.output_dir, 'tcav_results', 'tcav_comparison.png')
        except Exception as e:
            print(f"      ⚠️ Could not create TCAV comparison plot: {e}")
            return None
    
    def run_global_explainability_analysis(self):
        """Run comprehensive global explainability analysis"""
        print("\n" + "="*70)
        print("🌍 GLOBAL EXPLAINABILITY ANALYSIS")
        print("="*70)
        
        explainability_results = {}
        
        # 1. Define Concepts
        print("\n🧠 Concept Definition:")
        concepts = self.define_startup_concepts()
        explainability_results['concepts'] = concepts
        
        # 2. Compute Concept Vectors
        print("\n🎯 Concept Vector Computation:")
        concept_vectors = self.compute_concept_vectors()
        explainability_results['concept_vectors'] = concept_vectors
        
        # 3. Compute TCAV Scores
        print("\n📊 TCAV Score Computation:")
        tcav_scores = self.compute_tcav_scores()
        explainability_results['tcav_scores'] = tcav_scores
        
        # 4. Rank Concepts by Importance
        print("\n🏆 Concept Importance Ranking:")
        concept_rankings = self.rank_concepts_by_importance()
        explainability_results['concept_rankings'] = concept_rankings
        
        # 5. Create Visualizations
        print("\n🎨 Creating Global Explainability Visualizations:")
        visualizations = self.create_concept_visualizations()
        explainability_results['visualizations'] = visualizations
        
        # 6. Generate Global Insights
        print("\n💡 Generating Global Insights:")
        global_insights = self._generate_global_insights()
        explainability_results['global_insights'] = global_insights
        
        # Save results
        self._save_global_explainability_results(explainability_results)
        
        return explainability_results
    
    def _generate_global_insights(self):
        """Generate global insights from concept analysis"""
        insights = {}
        
        try:
            print("    💡 Analyzing global patterns...")
            
            # Most important concepts
            if self.tcav_scores:
                concept_rankings = self.rank_concepts_by_importance()
                
                if concept_rankings:
                    top_concepts = concept_rankings[:3]
                    insights['top_concepts'] = [c['concept'] for c in top_concepts]
                    
                    print(f"    🏆 Top 3 most important concepts:")
                    for i, concept in enumerate(top_concepts):
                        print(f"      {i+1}. {concept['concept']}: {concept['importance']:.3f}")
                
                # Concept with highest survival correlation
                survival_rankings = sorted(concept_rankings, 
                                         key=lambda x: abs(x['survival_correlation']), 
                                         reverse=True)
                if survival_rankings:
                    insights['highest_survival_correlation'] = survival_rankings[0]
                    print(f"    📈 Highest survival correlation: {survival_rankings[0]['concept']} "
                          f"({survival_rankings[0]['survival_correlation']:.3f})")
                
                # Most reliable concepts (high evidence)
                evidence_rankings = sorted(concept_rankings, 
                                         key=lambda x: x['n_examples'], 
                                         reverse=True)
                if evidence_rankings:
                    insights['most_evidence'] = evidence_rankings[:3]
                    print(f"    📊 Most reliable concepts (by evidence):")
                    for concept in evidence_rankings[:3]:
                        print(f"      {concept['concept']}: {concept['n_examples']} examples")
            
            # Overall model insights
            overall_survival_rate = self.labels.mean()
            insights['overall_survival_rate'] = overall_survival_rate
            insights['total_samples'] = len(self.labels)
            
            print(f"    📊 Overall survival rate: {overall_survival_rate:.3f}")
            print(f"    📊 Total samples analyzed: {len(self.labels):,}")
            
            # Concept coverage analysis
            if self.concept_vectors:
                concept_coverage = {}
                for concept_name, concept_data in self.concept_vectors.items():
                    coverage = concept_data['n_examples'] / len(self.labels)
                    concept_coverage[concept_name] = coverage
                
                insights['concept_coverage'] = concept_coverage
                
                highest_coverage = max(concept_coverage.items(), key=lambda x: x[1])
                print(f"    📈 Highest concept coverage: {highest_coverage[0]} ({highest_coverage[1]:.1%})")
            
        except Exception as e:
            print(f"    ⚠️ Could not generate global insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _save_global_explainability_results(self, explainability_results):
        """Save global explainability results"""
        # Save as pickle
        results_path = os.path.join(self.output_dir, "global_explainability_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({
                'explainability_results': explainability_results,
                'predictions': self.predictions,
                'probabilities': self.probabilities,
                'labels': self.labels,
                'embeddings': self.embeddings,
                'sequences': self.sequences,
                'metadata': self.metadata,
                'concepts': self.concepts,
                'concept_vectors': self.concept_vectors,
                'tcav_scores': self.tcav_scores
            }, f)
        
        # Save as text report
        report_path = os.path.join(self.output_dir, "global_explainability_report.txt")
        with open(report_path, 'w') as f:
            f.write("STARTUP2VEC GLOBAL EXPLAINABILITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples analyzed: {len(self.predictions):,}\n")
            f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Concepts defined
            if 'concepts' in explainability_results:
                f.write("CONCEPTS DEFINED:\n")
                f.write("-" * 20 + "\n")
                for concept_name, concept_data in explainability_results['concepts'].items():
                    f.write(f"{concept_name}: {concept_data['description']}\n")
                f.write("\n")
            
            # Concept rankings
            if 'concept_rankings' in explainability_results:
                f.write("CONCEPT IMPORTANCE RANKINGS:\n")
                f.write("-" * 35 + "\n")
                for i, ranking in enumerate(explainability_results['concept_rankings']):
                    f.write(f"{i+1}. {ranking['concept']}: "
                           f"Importance={ranking['importance']:.3f}, "
                           f"TCAV={ranking['tcav_score']:.3f}, "
                           f"Correlation={ranking['survival_correlation']:.3f}, "
                           f"Examples={ranking['n_examples']}\n")
                f.write("\n")
            
            # TCAV scores
            if 'tcav_scores' in explainability_results:
                f.write("TCAV SCORES:\n")
                f.write("-" * 15 + "\n")
                for concept_name, scores in explainability_results['tcav_scores'].items():
                    f.write(f"{concept_name}:\n")
                    f.write(f"  TCAV Score: {scores['tcav_score']:.3f}\n")
                    f.write(f"  Survival Correlation: {scores['survival_correlation']:.3f}\n")
                    f.write(f"  Number of Examples: {scores['n_examples']}\n")
                    f.write(f"  Concept Survival Rate: {scores['concept_survival_rate']:.3f}\n\n")
            
            # Global insights
            if 'global_insights' in explainability_results:
                insights = explainability_results['global_insights']
                f.write("GLOBAL INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Overall survival rate: {insights.get('overall_survival_rate', 'N/A'):.3f}\n")
                f.write(f"Total samples: {insights.get('total_samples', 'N/A'):,}\n")
                
                if 'top_concepts' in insights:
                    f.write(f"Top concepts: {', '.join(insights['top_concepts'])}\n")
                
                if 'highest_survival_correlation' in insights:
                    hsc = insights['highest_survival_correlation']
                    f.write(f"Highest survival correlation: {hsc['concept']} ({hsc['survival_correlation']:.3f})\n")
        
        print(f"\n✅ Global explainability results saved to:")
        print(f"  📊 Data: {results_path}")
        print(f"  📋 Report: {report_path}")
        print(f"  🧠 Concept analysis: {self.output_dir}/concept_analysis/")
        print(f"  📊 TCAV results: {self.output_dir}/tcav_results/")
    
    def run_complete_global_explainability(self, target_batches=500, balanced_sampling=False):
        """Run complete global explainability analysis"""
        print("🚀 STARTUP2VEC GLOBAL EXPLAINABILITY")
        print("=" * 80)
        print("Global explainability analysis using TCAV-inspired concept analysis")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data for concept analysis
        if not self.extract_data_for_concepts(target_batches, balanced_sampling):
            return False
        
        # Run global explainability analysis
        explainability_results = self.run_global_explainability_analysis()
        
        print(f"\n🎉 GLOBAL EXPLAINABILITY ANALYSIS COMPLETE!")
        print(f"🧠 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to: {self.output_dir}/")
        
        return explainability_results

def main():
    """Main function for global explainability"""
    CHECKPOINT_PATH = "survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"
    
    print("🔧 STARTUP2VEC GLOBAL EXPLAINABILITY")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 CUDA Available: {gpu_count} GPU(s)")
    else:
        print("❌ CUDA not available - will use CPU")
    
    print()
    print("🎯 GLOBAL EXPLAINABILITY FEATURES:")
    print("✅ TCAV-inspired concept analysis:")
    print("   • High-growth startups")
    print("   • Tech-focused companies") 
    print("   • Well-funded ventures")
    print("   • Team-focused culture")
    print("   • B2B vs consumer orientation")
    print("   • Early-stage characteristics")
    print("✅ Concept activation vector computation")
    print("✅ TCAV score calculation")
    print("✅ Concept importance ranking")
    print("✅ Global pattern identification")
    print("✅ GPU memory management with CPU fallback")
    print("✅ Balanced sampling option")
    print()
    
    explainer = StartupGlobalExplainer(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir="global_explainability_results"
    )
    
    # Set defaults for automation
    target_batches = 0  # FULL DATA
    balanced_sampling = False
    start_time = time.time()
    success = explainer.run_complete_global_explainability(
        target_batches=target_batches,
        balanced_sampling=balanced_sampling
    )
    end_time = time.time()
    
    if success:
        print(f"\n🎉 SUCCESS! Global explainability analysis completed in {end_time-start_time:.1f} seconds")
        print("\n🧠 GLOBAL ANALYSIS CREATED:")
        print("  • Abstract concept definitions and analysis")
        print("  • TCAV-inspired concept activation vectors")
        print("  • Concept importance rankings")
        print("  • Global pattern identification")
        print("  • Concept correlation analysis")
        print("  • Survival prediction insights")
        print("\n💡 NEXT STEPS:")
        print("  1. Check global_explainability_report.txt for comprehensive findings")
        print("  2. View concept_analysis/ folder for concept visualizations")
        print("  3. Check tcav_results/ folder for TCAV analysis")
        print("  4. Use global_explainability_results.pkl for further research")
        print("  5. Compare results across all 5 interpretability scripts")
        print("\n🎯 INTERPRETABILITY SUITE COMPLETE!")
        print("   You now have comprehensive analysis from all 5 scripts:")
        print("   01_algorithmic_audit.py - Bias detection across subgroups")
        print("   02_data_contribution_analysis.py - Event type contributions")
        print("   03_visual_exploration.py - Embedding space visualization")
        print("   04_local_explainability.py - Individual explanations")
        print("   05_global_explainability.py - Abstract concept analysis")
        return 0
    else:
        print(f"\n❌ Global explainability analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
