# interp_02_data_contribution_analysis.py
#!/usr/bin/env python3
"""
STARTUP2VEC DATA CONTRIBUTION ANALYSIS - Script 2/5
Analysis of how different event types contribute to predictions with GPU support
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
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, f1_score)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class StartupDataContributionAnalyzer:
    """Data contribution analysis for startup survival predictions"""
    
    def __init__(self, checkpoint_path, pretrained_path, output_dir="data_contribution_results"):
        self.checkpoint_path = checkpoint_path
        self.pretrained_path = pretrained_path
        self.output_dir = output_dir
        self.model = None
        self.datamodule = None
        
        # Core data
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.sequences = None
        self.metadata = None
        
        # Vocabulary
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.token_categories = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_and_data(self):
        """Load model and data with GPU memory management"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        
        try:
            from models.survival_model import StartupSurvivalModel
            from dataloaders.survival_datamodule import SurvivalDataModule
            
            # Load model to CPU first
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
                batch_size=16,
                num_workers=1,
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
        """Extract vocabulary and parse token categories for event analysis"""
        try:
            if hasattr(self.datamodule, 'vocabulary'):
                self.vocab_to_idx = self.datamodule.vocabulary.token2index
                self.idx_to_vocab = self.datamodule.vocabulary.index2token
                print(f"✅ Vocabulary extracted: {len(self.vocab_to_idx):,} tokens")
            else:
                print("⚠️ Could not extract vocabulary")
                return
            
            self.token_categories = self._parse_token_categories_for_events()
            
            print(f"\n📋 Event Categories Found:")
            for category, tokens in self.token_categories.items():
                if tokens:
                    print(f"  {category}: {len(tokens)} tokens")
            
        except Exception as e:
            print(f"⚠️ Vocabulary parsing failed: {e}")
    
    def _parse_token_categories_for_events(self):
        """Parse vocabulary into event-focused categories"""
        categories = {
            # Core event types
            'funding_events': {},
            'team_events': {},
            'product_events': {},
            'education_events': {},
            'acquisition_events': {},
            'ipo_events': {},
            
            # Detailed funding analysis
            'early_funding': {},
            'growth_funding': {},
            'late_funding': {},
            'funding_amounts': {},
            
            # Company characteristics for context
            'company_info': {},
            'geographic_tokens': {},
            'temporal_tokens': {},
        }
        
        for token_str, token_id in self.vocab_to_idx.items():
            # Funding events (comprehensive)
            if (token_str.startswith('INV_') or 
                'funding' in token_str.lower() or 
                'investment' in token_str.lower()):
                categories['funding_events'][token_id] = token_str
                
                # Categorize by funding stage
                if any(stage in token_str.lower() for stage in ['seed', 'angel', 'pre_seed']):
                    categories['early_funding'][token_id] = token_str
                elif any(stage in token_str.lower() for stage in ['series_a', 'series_b']):
                    categories['growth_funding'][token_id] = token_str
                elif any(stage in token_str.lower() for stage in ['series_c', 'series_d', 'late']):
                    categories['late_funding'][token_id] = token_str
                
                # Funding amounts
                if 'AMOUNT' in token_str or 'USD' in token_str:
                    categories['funding_amounts'][token_id] = token_str
            
            # Team/People events
            elif (token_str.startswith('PPL_') or 
                  token_str.startswith('PEOPLE_') or
                  'hire' in token_str.lower() or
                  'employee' in token_str.lower()):
                categories['team_events'][token_id] = token_str
            
            # Education events
            elif token_str.startswith('EDU_'):
                categories['education_events'][token_id] = token_str
            
            # Product/Technology events
            elif (token_str.startswith('TECH_') or
                  'product' in token_str.lower() or
                  'launch' in token_str.lower()):
                categories['product_events'][token_id] = token_str
            
            # Acquisition events
            elif token_str.startswith('ACQ_'):
                categories['acquisition_events'][token_id] = token_str
            
            # IPO events
            elif token_str.startswith('IPO_'):
                categories['ipo_events'][token_id] = token_str
            
            # Company information
            elif (token_str.startswith('COUNTRY_') or 
                  token_str.startswith('INDUSTRY_') or
                  token_str.startswith('MODEL_') or
                  token_str.startswith('CATEGORY_')):
                categories['company_info'][token_id] = token_str
            
            # Geographic tokens
            elif 'CITY_' in token_str or 'COUNTRY_' in token_str:
                categories['geographic_tokens'][token_id] = token_str
            
            # Temporal tokens
            elif token_str.startswith('DAYS_'):
                categories['temporal_tokens'][token_id] = token_str
        
        return categories
    
    def extract_data_with_event_analysis(self, target_batches=500, balanced_sampling=False):
        """Extract data with event-focused metadata"""
        print(f"\n🎯 EXTRACTING DATA FOR EVENT CONTRIBUTION ANALYSIS")
        print("="*60)
        
        if balanced_sampling:
            return self._extract_balanced_data_with_events(target_batches)
        else:
            return self._extract_standard_data_with_events(target_batches)
    
    def _extract_standard_data_with_events(self, target_batches):
        """Standard data extraction with event analysis"""
        val_loader = self.datamodule.val_dataloader()
        max_batches = min(target_batches, len(val_loader)) if target_batches > 0 else len(val_loader)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
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
                    
                    # Store results
                    all_predictions.extend(survival_preds.cpu().numpy())
                    all_probabilities.extend(survival_probs.cpu().numpy())
                    all_labels.extend(survival_labels.squeeze().cpu().numpy())
                    all_sequences.extend(input_ids[:, 0, :].cpu().numpy())
                    
                    # Extract event-focused metadata
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_event_metadata(batch, i, input_ids[i, 0, :])
                        all_metadata.append(metadata)
                    
                    successful_batches += 1
                    
                    # Clear GPU memory
                    del input_ids, padding_mask, survival_labels, outputs
                    del survival_logits, survival_probs, survival_preds
                    
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
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        return True
    
    def _extract_balanced_data_with_events(self, target_batches):
        """Extract balanced data with event analysis"""
        val_loader = self.datamodule.val_dataloader()
        
        survival_data = {'predictions': [], 'probabilities': [], 'labels': [], 'sequences': [], 'metadata': []}
        failure_data = {'predictions': [], 'probabilities': [], 'labels': [], 'sequences': [], 'metadata': []}
        
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
        
        print(f"Collecting balanced samples (target: {target_per_class} per class)...")
        
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
                    
                    for i in range(input_ids.size(0)):
                        true_label = survival_labels[i].squeeze().item()
                        
                        sample_data = {
                            'prediction': survival_preds[i].cpu().numpy(),
                            'probability': survival_probs[i].cpu().numpy(),
                            'label': true_label,
                            'sequence': input_ids[i, 0, :].cpu().numpy(),
                            'metadata': self._extract_event_metadata(batch, i, input_ids[i, 0, :])
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
        
        self.sequences = (survival_data['sequences'][:min_samples] + 
                         failure_data['sequences'][:min_samples])
        
        self.metadata = (survival_data['metadata'][:min_samples] + 
                        failure_data['metadata'][:min_samples])
        
        return True
    
    def _extract_event_metadata(self, batch, sample_idx, sequence):
        """Extract event-focused metadata from sequence"""
        try:
            base_metadata = {
                'sample_idx': sample_idx,
                'sequence_length': (sequence > 0).sum().item(),
                'company_age': batch['company_age_at_prediction'][sample_idx].item() if 'company_age_at_prediction' in batch else 2,
            }
            
            # Analyze events in sequence
            event_analysis = self._analyze_sequence_events(sequence)
            base_metadata.update(event_analysis)
            
            return base_metadata
        except Exception as e:
            return {
                'sample_idx': sample_idx, 'sequence_length': 0, 'company_age': 2,
                'has_funding_events': False, 'has_team_events': False, 'has_product_events': False,
                'has_education_events': False, 'funding_event_count': 0, 'team_event_count': 0,
                'funding_ratio': 0.0, 'event_diversity': 0.0, 'funding_stage': 'Unknown'
            }
    
    def _analyze_sequence_events(self, sequence):
        """Analyze events present in sequence"""
        event_analysis = {
            'has_funding_events': False, 'has_team_events': False, 'has_product_events': False,
            'has_education_events': False, 'has_acquisition_events': False, 'has_ipo_events': False,
            'funding_event_count': 0, 'team_event_count': 0, 'product_event_count': 0,
            'education_event_count': 0, 'total_event_count': 0,
            'funding_ratio': 0.0, 'event_diversity': 0.0, 'funding_stage': 'Unknown'
        }
        
        try:
            clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
            event_analysis['total_event_count'] = len(clean_sequence)
            
            # Count different event types
            funding_tokens = []
            
            for token_id in clean_sequence:
                token_id = int(token_id)
                
                # Check each event category
                if token_id in self.token_categories['funding_events']:
                    event_analysis['has_funding_events'] = True
                    event_analysis['funding_event_count'] += 1
                    token_str = self.idx_to_vocab.get(token_id, "")
                    funding_tokens.append(token_str)
                
                if token_id in self.token_categories['team_events']:
                    event_analysis['has_team_events'] = True
                    event_analysis['team_event_count'] += 1
                
                if token_id in self.token_categories['product_events']:
                    event_analysis['has_product_events'] = True
                    event_analysis['product_event_count'] += 1
                
                if token_id in self.token_categories['education_events']:
                    event_analysis['has_education_events'] = True
                    event_analysis['education_event_count'] += 1
                
                if token_id in self.token_categories['acquisition_events']:
                    event_analysis['has_acquisition_events'] = True
                
                if token_id in self.token_categories['ipo_events']:
                    event_analysis['has_ipo_events'] = True
            
            # Calculate ratios and diversity
            total_events = event_analysis['total_event_count']
            if total_events > 0:
                event_analysis['funding_ratio'] = event_analysis['funding_event_count'] / total_events
                
                # Event type diversity (how many different event types)
                event_types = sum([
                    event_analysis['has_funding_events'],
                    event_analysis['has_team_events'],
                    event_analysis['has_product_events'],
                    event_analysis['has_education_events'],
                    event_analysis['has_acquisition_events'],
                    event_analysis['has_ipo_events']
                ])
                event_analysis['event_diversity'] = event_types / 6  # Max 6 event types
            
            # Determine funding stage from tokens
            if funding_tokens:
                if any(token for token in funding_tokens if any(stage in token.lower() for stage in ['seed', 'angel'])):
                    event_analysis['funding_stage'] = 'Early'
                elif any(token for token in funding_tokens if any(stage in token.lower() for stage in ['series_a', 'series_b'])):
                    event_analysis['funding_stage'] = 'Growth'
                elif any(token for token in funding_tokens if any(stage in token.lower() for stage in ['series_c', 'late'])):
                    event_analysis['funding_stage'] = 'Late'
                else:
                    event_analysis['funding_stage'] = 'Other'
            
        except Exception as e:
            print(f"Warning: Could not analyze sequence events: {e}")
        
        return event_analysis
    
    def run_data_contribution_analysis(self):
        """Run comprehensive data contribution analysis"""
        print("\n" + "="*70)
        print("📊 DATA CONTRIBUTION ANALYSIS")
        print("="*70)
        
        contribution_results = {}
        
        # 1. Event Type Contribution
        print("\n📈 Event Type Contribution Analysis:")
        event_contrib = self._analyze_event_type_contribution()
        contribution_results['event_types'] = event_contrib
        
        # 2. Funding vs Non-Funding Events
        print("\n💰 Funding vs Non-Funding Events:")
        funding_contrib = self._analyze_funding_vs_nonfunding()
        contribution_results['funding_vs_nonfunding'] = funding_contrib
        
        # 3. Event Frequency Analysis
        print("\n🔢 Event Frequency Analysis:")
        frequency_contrib = self._analyze_event_frequency()
        contribution_results['event_frequency'] = frequency_contrib
        
        # 4. Event Diversity Analysis
        print("\n🎲 Event Diversity Analysis:")
        diversity_contrib = self._analyze_event_diversity()
        contribution_results['event_diversity'] = diversity_contrib
        
        # 5. Funding Stage Progression
        print("\n🚀 Funding Stage Progression Analysis:")
        stage_contrib = self._analyze_funding_stage_progression()
        contribution_results['funding_stages'] = stage_contrib
        
        # Save results
        self._save_contribution_results(contribution_results)
        
        return contribution_results
    
    def _analyze_event_type_contribution(self):
        """Analyze contribution of different event types"""
        event_contrib = {}
        
        event_types = [
            ('has_funding_events', 'Funding Events'),
            ('has_team_events', 'Team Events'),
            ('has_product_events', 'Product Events'),
            ('has_education_events', 'Education Events'),
            ('has_acquisition_events', 'Acquisition Events'),
            ('has_ipo_events', 'IPO Events')
        ]
        
        for event_key, event_name in event_types:
            has_events = np.array([m.get(event_key, False) for m in self.metadata])
            no_events = ~has_events
            
            if has_events.sum() > 10 and no_events.sum() > 10:
                # With events
                with_acc = (self.predictions[has_events] == self.labels[has_events]).mean()
                with_survival = self.labels[has_events].mean()
                with_predicted = self.predictions[has_events].mean()
                
                # Without events
                without_acc = (self.predictions[no_events] == self.labels[no_events]).mean()
                without_survival = self.labels[no_events].mean()
                without_predicted = self.predictions[no_events].mean()
                
                contribution_score = abs(with_survival - without_survival)
                
                event_contrib[event_name] = {
                    'with_events': {
                        'count': has_events.sum(),
                        'accuracy': with_acc,
                        'survival_rate': with_survival,
                        'predicted_rate': with_predicted
                    },
                    'without_events': {
                        'count': no_events.sum(),
                        'accuracy': without_acc,
                        'survival_rate': without_survival,
                        'predicted_rate': without_predicted
                    },
                    'contribution_score': contribution_score,
                    'survival_difference': with_survival - without_survival
                }
                
                print(f"  {event_name}:")
                print(f"    With events: {has_events.sum()} companies, {with_survival:.3f} survival")
                print(f"    Without events: {no_events.sum()} companies, {without_survival:.3f} survival")
                print(f"    Contribution score: {contribution_score:.3f}")
                print(f"    Difference: {event_contrib[event_name]['survival_difference']:+.3f}")
        
        return event_contrib
    
    def _analyze_funding_vs_nonfunding(self):
        """Analyze funding vs non-funding event contribution"""
        funding_ratios = [m.get('funding_ratio', 0) for m in self.metadata]
        
        high_funding = np.array([ratio > 0.5 for ratio in funding_ratios])
        low_funding = ~high_funding
        
        funding_contrib = {}
        
        if high_funding.sum() > 10 and low_funding.sum() > 10:
            # High funding ratio companies
            high_acc = (self.predictions[high_funding] == self.labels[high_funding]).mean()
            high_survival = self.labels[high_funding].mean()
            
            # Low funding ratio companies
            low_acc = (self.predictions[low_funding] == self.labels[low_funding]).mean()
            low_survival = self.labels[low_funding].mean()
            
            funding_contrib = {
                'high_funding_ratio': {
                    'count': high_funding.sum(),
                    'accuracy': high_acc,
                    'survival_rate': high_survival,
                    'avg_funding_ratio': np.mean(np.array(funding_ratios)[high_funding])
                },
                'low_funding_ratio': {
                    'count': low_funding.sum(),
                    'accuracy': low_acc,
                    'survival_rate': low_survival,
                    'avg_funding_ratio': np.mean(np.array(funding_ratios)[low_funding])
                },
                'difference': high_survival - low_survival
            }
            
            print(f"  High Funding Ratio (>50%): {high_funding.sum()} companies")
            print(f"    Survival rate: {high_survival:.3f}")
            print(f"    Average funding ratio: {funding_contrib['high_funding_ratio']['avg_funding_ratio']:.3f}")
            
            print(f"  Low Funding Ratio (≤50%): {low_funding.sum()} companies")
            print(f"    Survival rate: {low_survival:.3f}")
            print(f"    Difference: {funding_contrib['difference']:+.3f}")
        
        return funding_contrib
    
    def _analyze_event_frequency(self):
        """Analyze contribution by event frequency"""
        event_counts = [m.get('total_event_count', 0) for m in self.metadata]
        
        try:
            freq_quartiles = np.percentile(event_counts, [25, 50, 75])
            frequency_contrib = {}
            
            for i, (lower, upper) in enumerate([(0, freq_quartiles[0]), 
                                               (freq_quartiles[0], freq_quartiles[1]),
                                               (freq_quartiles[1], freq_quartiles[2]), 
                                               (freq_quartiles[2], np.inf)]):
                if upper == np.inf:
                    mask = np.array([count >= lower for count in event_counts])
                    bin_name = f"Q4_High_Freq"
                else:
                    mask = np.array([lower <= count < upper for count in event_counts])
                    bin_name = f"Q{i+1}"
                
                if mask.sum() > 10:
                    subset_acc = (self.predictions[mask] == self.labels[mask]).mean()
                    subset_survival = self.labels[mask].mean()
                    
                    frequency_contrib[bin_name] = {
                        'count': mask.sum(),
                        'accuracy': subset_acc,
                        'survival_rate': subset_survival,
                        'avg_event_count': np.mean(np.array(event_counts)[mask])
                    }
                    
                    print(f"  {bin_name}: {mask.sum()} companies")
                    print(f"    Survival rate: {subset_survival:.3f}")
                    print(f"    Avg events: {frequency_contrib[bin_name]['avg_event_count']:.1f}")
            
            return frequency_contrib
        except Exception as e:
            print(f"  ⚠️ Could not analyze event frequency: {e}")
            return {}
    
    def _analyze_event_diversity(self):
        """Analyze contribution by event diversity"""
        diversities = [m.get('event_diversity', 0) for m in self.metadata]
        
        if len(diversities) > 0 and max(diversities) > 0:
            diversity_median = np.median([d for d in diversities if d > 0])
            
            high_diversity = np.array([d >= diversity_median for d in diversities])
            low_diversity = np.array([0 < d < diversity_median for d in diversities])
            
            diversity_contrib = {}
            
            if high_diversity.sum() > 10 and low_diversity.sum() > 10:
                # High diversity
                high_acc = (self.predictions[high_diversity] == self.labels[high_diversity]).mean()
                high_survival = self.labels[high_diversity].mean()
                
                # Low diversity
                low_acc = (self.predictions[low_diversity] == self.labels[low_diversity]).mean()
                low_survival = self.labels[low_diversity].mean()
                
                diversity_contrib = {
                    'high_diversity': {
                        'count': high_diversity.sum(),
                        'accuracy': high_acc,
                        'survival_rate': high_survival,
                        'avg_diversity': np.mean(np.array(diversities)[high_diversity])
                    },
                    'low_diversity': {
                        'count': low_diversity.sum(),
                        'accuracy': low_acc,
                        'survival_rate': low_survival,
                        'avg_diversity': np.mean(np.array(diversities)[low_diversity])
                    },
                    'difference': high_survival - low_survival
                }
                
                print(f"  High Diversity (≥{diversity_median:.3f}): {high_diversity.sum()} companies")
                print(f"    Survival rate: {high_survival:.3f}")
                
                print(f"  Low Diversity (<{diversity_median:.3f}): {low_diversity.sum()} companies")
                print(f"    Survival rate: {low_survival:.3f}")
                print(f"    Difference: {diversity_contrib['difference']:+.3f}")
            
            return diversity_contrib
        
        return {}
    
    def _analyze_funding_stage_progression(self):
        """Analyze contribution by funding stage progression"""
        funding_stages = [m.get('funding_stage', 'Unknown') for m in self.metadata]
        stage_contrib = {}
        
        for stage in ['Early', 'Growth', 'Late', 'Other']:
            mask = np.array([s == stage for s in funding_stages])
            
            if mask.sum() > 10:
                subset_acc = (self.predictions[mask] == self.labels[mask]).mean()
                subset_survival = self.labels[mask].mean()
                
                stage_contrib[stage] = {
                    'count': mask.sum(),
                    'accuracy': subset_acc,
                    'survival_rate': subset_survival
                }
                
                print(f"  {stage} Stage: {mask.sum()} companies")
                print(f"    Survival rate: {subset_survival:.3f}")
                print(f"    Accuracy: {subset_acc:.3f}")
        
        return stage_contrib
    
    def _save_contribution_results(self, contribution_results):
        """Save contribution analysis results"""
        # Save as pickle
        results_path = os.path.join(self.output_dir, "data_contribution_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({
                'contribution_results': contribution_results,
                'predictions': self.predictions,
                'probabilities': self.probabilities,
                'labels': self.labels,
                'metadata': self.metadata
            }, f)
        
        # Save as text report
        report_path = os.path.join(self.output_dir, "data_contribution_report.txt")
        with open(report_path, 'w') as f:
            f.write("STARTUP2VEC DATA CONTRIBUTION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples analyzed: {len(self.predictions):,}\n")
            f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, results in contribution_results.items():
                f.write(f"{category.upper()} ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                
                if category == 'event_types':
                    for event_type, data in results.items():
                        f.write(f"{event_type}:\n")
                        f.write(f"  With events: {data['with_events']['count']} companies, "
                               f"survival rate: {data['with_events']['survival_rate']:.3f}\n")
                        f.write(f"  Without events: {data['without_events']['count']} companies, "
                               f"survival rate: {data['without_events']['survival_rate']:.3f}\n")
                        f.write(f"  Contribution score: {data['contribution_score']:.3f}\n")
                        f.write(f"  Difference: {data['survival_difference']:+.3f}\n\n")
                
                elif category in ['funding_vs_nonfunding', 'event_diversity']:
                    for subcat, data in results.items():
                        if isinstance(data, dict) and 'count' in data:
                            f.write(f"  {subcat}: {data['count']} companies, "
                                   f"survival rate: {data['survival_rate']:.3f}\n")
                        elif subcat == 'difference':
                            f.write(f"  Difference: {data:+.3f}\n")
                
                f.write("\n")
        
        print(f"\n✅ Data contribution results saved to:")
        print(f"  📊 Data: {results_path}")
        print(f"  📋 Report: {report_path}")
    
    def run_complete_analysis(self, target_batches=500, balanced_sampling=False):
        """Run complete data contribution analysis"""
        print("🚀 STARTUP2VEC DATA CONTRIBUTION ANALYSIS")
        print("=" * 80)
        print("Analysis of how different event types contribute to predictions")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data with event analysis
        if not self.extract_data_with_event_analysis(target_batches, balanced_sampling):
            return False
        
        # Run data contribution analysis
        contribution_results = self.run_data_contribution_analysis()
        
        print(f"\n🎉 DATA CONTRIBUTION ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to: {self.output_dir}/")
        
        return contribution_results

def main():
    """Main function for data contribution analysis"""
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    print("🔧 STARTUP2VEC DATA CONTRIBUTION ANALYSIS")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 CUDA Available: {gpu_count} GPU(s)")
    else:
        print("❌ CUDA not available - will use CPU")
    
    print()
    print("🎯 DATA CONTRIBUTION ANALYSIS FEATURES:")
    print("✅ Event type contribution analysis:")
    print("   • Funding vs team vs product vs education events")
    print("   • Event frequency and diversity analysis")
    print("   • Funding stage progression analysis")
    print("✅ GPU memory management with CPU fallback")
    print("✅ Balanced sampling option")
    print("✅ Comprehensive event categorization")
    print()
    
    analyzer = StartupDataContributionAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir="data_contribution_results"
    )
    
    # Get user preferences
    print("🎛️ ANALYSIS OPTIONS:")
    print("1. Standard analysis (original data distribution)")
    print("2. Balanced analysis (equal success/failure samples)")
    
    choice = input("Choose analysis type (1 or 2): ").strip()
    balanced_sampling = choice == "2"
    
    batch_choice = input("Enter number of batches (0 for ALL data, 500+ recommended): ").strip()
    try:
        target_batches = int(batch_choice)
    except ValueError:
        target_batches = 500
    
    # Run analysis
    start_time = time.time()
    success = analyzer.run_complete_analysis(
        target_batches=target_batches,
        balanced_sampling=balanced_sampling
    )
    end_time = time.time()
    
    if success:
        print(f"\n🎉 SUCCESS! Data contribution analysis completed in {end_time-start_time:.1f} seconds")
        print("\n💡 NEXT STEPS:")
        print("  1. Check data_contribution_report.txt for detailed findings")
        print("  2. Use data_contribution_results.pkl for further analysis")
        print("  3. Identify which event types contribute most to survival predictions")
        print("  4. Run the next script: 03_visual_exploration.py")
        return 0
    else:
        print(f"\n❌ Data contribution analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
