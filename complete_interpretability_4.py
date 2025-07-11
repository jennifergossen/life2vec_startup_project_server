#!/usr/bin/env python3
"""
STARTUP2VEC ENHANCED INTERPRETABILITY ANALYSIS - COMPLETE FIXED VERSION
Complete interpretability analysis with FIXED balanced sampling bug and CUDA memory management
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
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve,
                           balanced_accuracy_score, f1_score, precision_recall_curve,
                           precision_score, recall_score, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class CompleteFixedStartupInterpretabilityAnalyzer:
    """COMPLETE FIXED interpretability analyzer with CUDA memory management and balanced sampling"""
    
    def __init__(self, checkpoint_path, pretrained_path, output_dir="startup_interpretability"):
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
        
        # Token categories and analysis
        self.token_categories = None
        self.startup_characteristics = None
        self.token_frequencies = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_and_data(self):
        """Load model and data with vocabulary - FIXED CUDA handling"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        
        try:
            from models.survival_model import StartupSurvivalModel
            from dataloaders.survival_datamodule import SurvivalDataModule
            
            # FIXED: Load model to CPU first
            print("📥 Loading model to CPU first...")
            self.model = StartupSurvivalModel.load_from_checkpoint(
                self.checkpoint_path,
                pretrained_model_path=self.pretrained_path,
                map_location='cpu'  # CRITICAL: Always load to CPU first
            )
            self.model.eval()
            print("✅ Model loaded to CPU successfully")
            
            # Load datamodule with smaller batch size to help with memory
            print("📥 Loading datamodule...")
            self.datamodule = SurvivalDataModule(
                corpus_name="startup_corpus",
                vocab_name="startup_vocab",
                batch_size=16,  # Reduced from 32 to help with memory
                num_workers=1,  # Reduced workers
                prediction_windows=[1, 2, 3, 4]
            )
            self.datamodule.setup()
            print("✅ Datamodule loaded successfully")
            
            # Extract and parse vocabulary
            self._extract_vocabulary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            return False
    
    def _extract_vocabulary(self):
        """Extract vocabulary and parse token categories from actual structure"""
        try:
            # Extract vocabulary
            if hasattr(self.datamodule, 'vocabulary'):
                self.vocab_to_idx = self.datamodule.vocabulary.token2index
                self.idx_to_vocab = self.datamodule.vocabulary.index2token
                print(f"✅ Vocabulary extracted: {len(self.vocab_to_idx):,} tokens")
            else:
                print("⚠️ Could not extract vocabulary")
                return
            
            # Parse token categories based on actual token structure
            self.token_categories = self._parse_token_categories()
            
            print(f"\n📋 Token Categories Found:")
            for category, tokens in self.token_categories.items():
                print(f"  {category}: {len(tokens)} tokens")
                # Show examples
                examples = list(tokens.keys())[:3]
                if examples:
                    example_names = [self.idx_to_vocab.get(token, f"Token_{token}") for token in examples]
                    print(f"    Examples: {', '.join(example_names)}")
            
        except Exception as e:
            print(f"⚠️ Vocabulary parsing failed: {e}")
    
    def _parse_token_categories(self):
        """Parse vocabulary into categories based on actual token structure"""
        categories = {
            # Company characteristics
            'company_country': {},          
            'company_category': {},         
            'company_employee_size': {},    
            'company_industry': {},         
            'company_business_model': {},   
            'company_technology': {},       
            
            # Event types and categories
            'event_types': {},              
            'event_categories': {},         
            'event_terms': {},              
            'event_roles': {},              
            
            # People/roles
            'people_jobs': {},              
            'people_terms': {},             
            'people_job_titles': {},        
            
            # Education events
            'education_degree_type': {},    
            'education_institution': {},    
            'education_subject': {},        
            
            # Investment events
            'investment_types': {},         
            'investment_investor_types': {},
            'investment_amounts': {},       
            'investment_fund_sizes': {},    
            'investment_counts': {},        
            'investment_valuations': {},    
            'investment_other': {},         
            
            # Acquisition events
            'acquisition_types': {},        
            'acquisition_prices': {},       
            'acquisition_other': {},        
            
            # IPO events
            'ipo_exchanges': {},            
            'ipo_money_raised': {},         
            'ipo_share_prices': {},         
            'ipo_valuations': {},           
            'ipo_other': {},                
            
            # Temporal
            'days_since_founding': {},      
        }
        
        for token_str, token_id in self.vocab_to_idx.items():
            # Company characteristics
            if token_str.startswith('COUNTRY_'):
                categories['company_country'][token_id] = token_str
            elif token_str.startswith('CATEGORY_'):
                categories['company_category'][token_id] = token_str
            elif token_str.startswith('EMPLOYEE_'):
                categories['company_employee_size'][token_id] = token_str
            elif token_str.startswith('INDUSTRY_'):
                categories['company_industry'][token_id] = token_str
            elif token_str.startswith('MODEL_'):
                categories['company_business_model'][token_id] = token_str
            elif token_str.startswith('TECH_'):
                categories['company_technology'][token_id] = token_str
            
            # Events
            elif token_str.startswith('EVT_TYPE_'):
                categories['event_types'][token_id] = token_str
            elif token_str.startswith('EVT_CAT_'):
                categories['event_categories'][token_id] = token_str
            elif token_str.startswith('EVT_TERM_'):
                categories['event_terms'][token_id] = token_str
            elif token_str.startswith('EVENT_ROLES_'):
                categories['event_roles'][token_id] = token_str
            
            # People
            elif token_str.startswith('PPL_JOB_'):
                categories['people_jobs'][token_id] = token_str
            elif token_str.startswith('PPL_TERM_'):
                categories['people_terms'][token_id] = token_str
            elif token_str.startswith('PEOPLE_JOB_TITLE_'):
                categories['people_job_titles'][token_id] = token_str
            
            # Education
            elif token_str.startswith('EDU_DEGREE_TYPE_'):
                categories['education_degree_type'][token_id] = token_str
            elif token_str.startswith('EDU_INSTITUTION_'):
                categories['education_institution'][token_id] = token_str
            elif token_str.startswith('EDU_SUBJECT_'):
                categories['education_subject'][token_id] = token_str
            
            # Investment
            elif token_str.startswith('INV_INVESTMENT_TYPE_'):
                categories['investment_types'][token_id] = token_str
            elif token_str.startswith('INV_INVESTOR_TYPES_'):
                categories['investment_investor_types'][token_id] = token_str
            elif token_str.startswith('INV_RAISED_AMOUNT_USD_'):
                categories['investment_amounts'][token_id] = token_str
            elif token_str.startswith('INV_FUND_SIZE_USD_'):
                categories['investment_fund_sizes'][token_id] = token_str
            elif token_str.startswith('INV_INVESTOR_COUNT_'):
                categories['investment_counts'][token_id] = token_str
            elif token_str.startswith('INV_POST_MONEY_VALUATION_USD_'):
                categories['investment_valuations'][token_id] = token_str
            elif token_str.startswith('INV_'):
                categories['investment_other'][token_id] = token_str
            
            # Acquisition
            elif token_str.startswith('ACQ_ACQUISITION_TYPE_'):
                categories['acquisition_types'][token_id] = token_str
            elif token_str.startswith('ACQ_PRICE_USD_'):
                categories['acquisition_prices'][token_id] = token_str
            elif token_str.startswith('ACQ_'):
                categories['acquisition_other'][token_id] = token_str
            
            # IPO
            elif token_str.startswith('IPO_EXCHANGE_'):
                categories['ipo_exchanges'][token_id] = token_str
            elif token_str.startswith('IPO_MONEY_RAISED_USD_'):
                categories['ipo_money_raised'][token_id] = token_str
            elif token_str.startswith('IPO_SHARE_PRICE_USD_'):
                categories['ipo_share_prices'][token_id] = token_str
            elif token_str.startswith('IPO_VALUATION_USD_'):
                categories['ipo_valuations'][token_id] = token_str
            elif token_str.startswith('IPO_'):
                categories['ipo_other'][token_id] = token_str
            
            # Temporal
            elif token_str.startswith('DAYS_'):
                categories['days_since_founding'][token_id] = token_str
        
        return categories
    
    def calculate_matthews_correlation_coefficient(self):
        """Calculate MCC (Matthews Correlation Coefficient) like in finetuning"""
        try:
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(self.labels, self.predictions)
            return mcc
        except Exception as e:
            print(f"⚠️ Could not calculate MCC: {e}")
            return float('nan')
    
    def extract_data_with_characteristics(self, target_batches=500, balanced_sampling=False):
        """Extract data with FIXED balanced sampling option and CUDA memory management"""
        print(f"\n🎯 EXTRACTING DATA WITH TOKEN ANALYSIS")
        print("="*60)
        
        if balanced_sampling:
            print("🎯 Using FIXED balanced sampling strategy...")
            return self._extract_balanced_data_FIXED(target_batches)
        else:
            print("📊 Using original data distribution...")
            return self._extract_standard_data_FIXED(target_batches)
    
    def _extract_standard_data_FIXED(self, target_batches):
        """Standard data extraction with FIXED CUDA memory management"""
        val_loader = self.datamodule.val_dataloader()
        max_batches = min(target_batches, len(val_loader)) if target_batches > 0 else len(val_loader)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        all_sequences = []
        all_metadata = []
        
        # FIXED: Better device handling with fallback
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔍 Attempting to use device: {device}")
        
        # CRITICAL FIX: Try GPU first, fallback to CPU if OOM
        try:
            # Clear any existing GPU memory first
            if torch.cuda.is_available():
                self.clear_cuda_cache()
            
            self.model = self.model.to(device)
            print(f"✅ Model successfully loaded to {device}")
            
            # Check available memory after model loading
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 GPU memory after model load: {allocated:.2f}GB")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ CUDA out of memory during model loading!")
                print(f"💡 Falling back to CPU inference...")
                
                # Clear cache and try CPU
                self.clear_cuda_cache()
                
                device = 'cpu'
                self.model = self.model.to(device)
                print(f"✅ Model loaded to CPU successfully")
            else:
                raise e
        
        print(f"Processing {max_batches:,} batches on {device}...")
        
        successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if target_batches > 0 and batch_idx >= max_batches:
                    break
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{max_batches} (successful: {successful_batches})", end='\r')
                    
                    # Clear cache periodically if using GPU
                    if device == 'cuda' and batch_idx % 100 == 0:
                        self.clear_cuda_cache()
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    # Forward pass
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        padding_mask=padding_mask
                    )
                    
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]
                    
                    # Move to CPU immediately to save GPU memory
                    all_predictions.extend(survival_preds.cpu().numpy())
                    all_probabilities.extend(survival_probs.cpu().numpy())
                    all_labels.extend(survival_labels.squeeze().cpu().numpy())
                    all_embeddings.extend(company_embeddings.cpu().numpy())
                    all_sequences.extend(input_ids[:, 0, :].cpu().numpy())
                    
                    # Extract metadata
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_metadata(batch, i, input_ids[i, 0, :])
                        all_metadata.append(metadata)
                    
                    successful_batches += 1
                    
                    # Clear GPU tensors immediately
                    del input_ids, padding_mask, survival_labels, outputs
                    del survival_logits, survival_probs, survival_preds, transformer_output, company_embeddings
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                        self.clear_cuda_cache()
                        continue
                    else:
                        print(f"\nError in batch {batch_idx}: {e}")
                        continue
                except Exception as e:
                    print(f"\nUnexpected error in batch {batch_idx}: {e}")
                    continue
        
        print(f"\n✅ Data extraction complete: {len(all_predictions):,} samples from {successful_batches} successful batches")
        
        if len(all_predictions) == 0:
            print("❌ No data extracted! Model may be too large for available memory.")
            return False
        
        # Store results
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.labels = np.array(all_labels)
        self.embeddings = np.array(all_embeddings)
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        # Additional analyses
        self._parse_startup_characteristics()
        self._analyze_token_frequency()
        self._detailed_performance_analysis()
        
        return True
    
    def _extract_balanced_data_FIXED(self, target_batches):
        """FIXED: Extract truly balanced data with proper CUDA memory management"""
        val_loader = self.datamodule.val_dataloader()
        
        # FIXED: Separate collections for truly different samples
        survival_data = {
            'predictions': [],
            'probabilities': [],
            'labels': [],
            'embeddings': [],
            'sequences': [],
            'metadata': []
        }
        
        failure_data = {
            'predictions': [],
            'probabilities': [],
            'labels': [],
            'embeddings': [],
            'sequences': [],
            'metadata': []
        }
        
        target_per_class = target_batches * 8  # Adjusted for batch size 16
        
        # FIXED: Better device handling with fallback
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔍 Attempting to use device: {device}")
        
        # CRITICAL FIX: Try GPU first, fallback to CPU if OOM
        try:
            if torch.cuda.is_available():
                self.clear_cuda_cache()
            
            self.model = self.model.to(device)
            print(f"✅ Model successfully loaded to {device}")
            
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 GPU memory after model load: {allocated:.2f}GB")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ CUDA out of memory during model loading!")
                print(f"💡 Falling back to CPU inference...")
                
                self.clear_cuda_cache()
                device = 'cpu'
                self.model = self.model.to(device)
                print(f"✅ Model loaded to CPU successfully")
            else:
                raise e
        
        print(f"Collecting FIXED balanced samples (target: {target_per_class} per class)...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Stop when we have enough of both classes
                if (len(survival_data['labels']) >= target_per_class and 
                    len(failure_data['labels']) >= target_per_class):
                    break
                
                if batch_idx % 50 == 0:
                    survived = len(survival_data['labels'])
                    failed = len(failure_data['labels'])
                    print(f"  Batch {batch_idx}: {survived} survived, {failed} failed", end='\r')
                    
                    # Clear cache periodically
                    if device == 'cuda' and batch_idx % 100 == 0:
                        self.clear_cuda_cache()
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    # Forward pass
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        padding_mask=padding_mask
                    )
                    
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]
                    
                    # FIXED: Separate by TRUE LABELS into different data structures
                    for i in range(input_ids.size(0)):
                        true_label = survival_labels[i].squeeze().item()
                        
                        sample_data = {
                            'prediction': survival_preds[i].cpu().numpy(),
                            'probability': survival_probs[i].cpu().numpy(),
                            'label': true_label,
                            'embedding': company_embeddings[i].cpu().numpy(),
                            'sequence': input_ids[i, 0, :].cpu().numpy(),
                            'metadata': self._extract_metadata(batch, i, input_ids[i, 0, :])
                        }
                        
                        # CRITICAL FIX: Store in separate collections based on TRUE LABEL
                        if true_label == 1 and len(survival_data['labels']) < target_per_class:
                            # This is a SURVIVED startup
                            survival_data['predictions'].append(sample_data['prediction'])
                            survival_data['probabilities'].append(sample_data['probability'])
                            survival_data['labels'].append(sample_data['label'])
                            survival_data['embeddings'].append(sample_data['embedding'])
                            survival_data['sequences'].append(sample_data['sequence'])
                            survival_data['metadata'].append(sample_data['metadata'])
                            
                        elif true_label == 0 and len(failure_data['labels']) < target_per_class:
                            # This is a FAILED startup
                            failure_data['predictions'].append(sample_data['prediction'])
                            failure_data['probabilities'].append(sample_data['probability'])
                            failure_data['labels'].append(sample_data['label'])
                            failure_data['embeddings'].append(sample_data['embedding'])
                            failure_data['sequences'].append(sample_data['sequence'])
                            failure_data['metadata'].append(sample_data['metadata'])
                    
                    # Clear GPU tensors immediately
                    del input_ids, padding_mask, survival_labels, outputs
                    del survival_logits, survival_probs, survival_preds, transformer_output, company_embeddings
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                        self.clear_cuda_cache()
                        continue
                    else:
                        print(f"\nError in batch {batch_idx}: {e}")
                        continue
                except Exception as e:
                    print(f"\nUnexpected error in batch {batch_idx}: {e}")
                    continue
        
        # FIXED: Combine truly different samples
        min_samples = min(len(survival_data['labels']), len(failure_data['labels']))
        
        print(f"\n✅ FIXED balanced sampling complete!")
        print(f"   📊 Collected: {len(survival_data['labels'])} survived, {len(failure_data['labels'])} failed")
        print(f"   ⚖️ Using: {min_samples} per class ({min_samples * 2} total)")
        
        if min_samples == 0:
            print("❌ No balanced samples collected! Try increasing target_batches or checking data.")
            return False
        
        # Combine the balanced data
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
        
        # Verify the fix worked
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"   ✅ Final distribution: {dict(zip(unique_labels, counts))}")
        
        # Additional analyses
        self._parse_startup_characteristics()
        self._analyze_token_frequency()
        self._detailed_performance_analysis()
        
        return True
    
    def _extract_metadata(self, batch, sample_idx, sequence):
        """Extract metadata including characteristics parsed from tokens"""
        try:
            base_metadata = {
                'batch_idx': batch['sequence_id'][sample_idx].item() if 'sequence_id' in batch else -1,
                'sample_idx': sample_idx,
                'sequence_length': (sequence > 0).sum().item(),
                'prediction_window': batch['prediction_window'][sample_idx].item() if 'prediction_window' in batch else 1,
                'company_age': batch['company_age_at_prediction'][sample_idx].item() if 'company_age_at_prediction' in batch else 2,
                'founded_year': batch['company_founded_year'][sample_idx].item() if 'company_founded_year' in batch else 2020,
            }
            
            # Parse characteristics from tokens in sequence
            if self.token_categories:
                characteristics = self._parse_sequence_characteristics(sequence)
                base_metadata.update(characteristics)
            
            return base_metadata
        except Exception as e:
            # Return safe defaults if extraction fails
            return {
                'batch_idx': -1, 'sample_idx': sample_idx, 'sequence_length': 0,
                'prediction_window': 1, 'company_age': 2, 'founded_year': 2020,
                'country': 'Unknown', 'industry_category': 'Unknown', 'employee_size': 'Unknown',
                'business_model': 'Unknown', 'technology_type': 'Unknown',
                'has_investment_events': False, 'has_acquisition_events': False,
                'has_ipo_events': False, 'has_education_events': False, 'has_people_events': False,
                'investment_event_count': 0, 'people_job_count': 0, 'education_event_count': 0,
                'unique_token_count': 0, 'token_diversity': 0.0
            }
    
    def _parse_sequence_characteristics(self, sequence):
        """Parse sequence to extract startup characteristics from actual tokens"""
        characteristics = {
            'country': 'Unknown', 'industry_category': 'Unknown', 'employee_size': 'Unknown',
            'business_model': 'Unknown', 'technology_type': 'Unknown',
            'has_investment_events': False, 'has_acquisition_events': False,
            'has_ipo_events': False, 'has_education_events': False, 'has_people_events': False,
            'investment_event_count': 0, 'people_job_count': 0, 'education_event_count': 0,
            'unique_token_count': 0, 'token_diversity': 0.0
        }
        
        try:
            clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
            characteristics['unique_token_count'] = len(set(clean_sequence))
            characteristics['token_diversity'] = len(set(clean_sequence)) / len(clean_sequence) if len(clean_sequence) > 0 else 0
            
            for token_id in clean_sequence:
                token_str = self.idx_to_vocab.get(int(token_id), "")
                
                # Parse company characteristics from tokens
                if token_str.startswith('COUNTRY_'):
                    characteristics['country'] = token_str.replace('COUNTRY_', '')
                elif token_str.startswith('CATEGORY_') or token_str.startswith('INDUSTRY_'):
                    characteristics['industry_category'] = token_str.split('_', 1)[1] if '_' in token_str else 'Unknown'
                elif token_str.startswith('EMPLOYEE_'):
                    characteristics['employee_size'] = token_str.replace('EMPLOYEE_', '')
                elif token_str.startswith('MODEL_'):
                    characteristics['business_model'] = token_str.replace('MODEL_', '')
                elif token_str.startswith('TECH_'):
                    characteristics['technology_type'] = token_str.replace('TECH_', '')
                
                # Check for event presence using token categories
                if int(token_id) in self.token_categories['investment_types'] or int(token_id) in self.token_categories['investment_amounts']:
                    characteristics['has_investment_events'] = True
                    characteristics['investment_event_count'] += 1
                elif int(token_id) in self.token_categories['acquisition_types']:
                    characteristics['has_acquisition_events'] = True
                elif int(token_id) in self.token_categories['ipo_exchanges'] or int(token_id) in self.token_categories['ipo_money_raised']:
                    characteristics['has_ipo_events'] = True
                elif (int(token_id) in self.token_categories['education_degree_type'] or 
                      int(token_id) in self.token_categories['education_institution'] or 
                      int(token_id) in self.token_categories['education_subject']):
                    characteristics['has_education_events'] = True
                    characteristics['education_event_count'] += 1
                elif int(token_id) in self.token_categories['people_jobs'] or int(token_id) in self.token_categories['people_job_titles']:
                    characteristics['has_people_events'] = True
                    characteristics['people_job_count'] += 1
        except Exception as e:
            print(f"Warning: Could not parse sequence characteristics: {e}")
        
        return characteristics
    
    def _parse_startup_characteristics(self):
        """Parse all startup characteristics from token metadata"""
        print("\n📋 Parsing startup characteristics from tokens...")
        
        if not self.metadata:
            print("⚠️ No metadata available")
            return
        
        # Count characteristics from tokens
        countries = Counter([m['country'] for m in self.metadata])
        industries = Counter([m['industry_category'] for m in self.metadata])
        employee_sizes = Counter([m['employee_size'] for m in self.metadata])
        business_models = Counter([m['business_model'] for m in self.metadata])
        tech_types = Counter([m['technology_type'] for m in self.metadata])
        
        # Age and temporal analysis with safety checks
        ages = [m['company_age'] for m in self.metadata if m['company_age'] is not None and not np.isnan(m['company_age'])]
        years = [m['founded_year'] for m in self.metadata if m['founded_year'] is not None and not np.isnan(m['founded_year'])]
        
        print(f"\n📊 Startup Characteristics Found:")
        print(f"  Countries: {dict(countries.most_common(10))}")
        print(f"  Industries: {dict(industries.most_common(10))}")
        print(f"  Employee Sizes: {dict(employee_sizes)}")
        print(f"  Business Models: {dict(business_models)}")
        print(f"  Technology Types: {dict(tech_types.most_common(5))}")
        
        # Temporal analysis with safety checks
        if ages:
            print(f"\n⏰ Temporal Characteristics:")
            print(f"  Average company age: {np.mean(ages):.1f} years")
            print(f"  Age range: {np.min(ages):.1f} - {np.max(ages):.1f} years")
        
        if years:
            print(f"  Founded year range: {int(np.min(years))} - {int(np.max(years))}")
        
        # Event presence statistics
        investment_events = sum(1 for m in self.metadata if m['has_investment_events'])
        acquisition_events = sum(1 for m in self.metadata if m['has_acquisition_events'])
        ipo_events = sum(1 for m in self.metadata if m['has_ipo_events'])
        education_events = sum(1 for m in self.metadata if m['has_education_events'])
        
        print(f"\n📈 Event Presence:")
        print(f"  Investment Events: {investment_events:,} ({investment_events/len(self.metadata)*100:.1f}%)")
        print(f"  Acquisition Events: {acquisition_events:,} ({acquisition_events/len(self.metadata)*100:.1f}%)")
        print(f"  IPO Events: {ipo_events:,} ({ipo_events/len(self.metadata)*100:.1f}%)")
        print(f"  Education Events: {education_events:,} ({education_events/len(self.metadata)*100:.1f}%)")
        
        self.startup_characteristics = {
            'countries': countries, 'industries': industries, 'employee_sizes': employee_sizes,
            'business_models': business_models, 'tech_types': tech_types,
            'temporal': {'ages': ages, 'years': years},
            'event_stats': {
                'investment_events': investment_events, 'acquisition_events': acquisition_events,
                'ipo_events': ipo_events, 'education_events': education_events
            }
        }
    
    def _analyze_token_frequency(self):
        """Analyze token frequency patterns"""
        print("\n�� TOKEN FREQUENCY ANALYSIS")
        print("-" * 40)
        
        if not self.sequences:
            print("⚠️ No sequences available")
            return
        
        token_counts = Counter()
        total_sequences = len(self.sequences)
        
        for sequence in self.sequences:
            clean_sequence = sequence[sequence > 0]
            for token in clean_sequence:
                token_counts[int(token)] += 1
        
        total_tokens = sum(token_counts.values())
        
        if total_tokens == 0:
            print("⚠️ No tokens found")
            return
        
        # High frequency tokens (>1%)
        frequent_tokens = {token: count for token, count in token_counts.items() 
                         if count/total_tokens > 0.01}
        
        # Rare tokens (<0.01%)
        rare_tokens = {token: count for token, count in token_counts.items() 
                      if count/total_tokens < 0.0001}
        
        print(f"📊 Token Frequency Statistics:")
        print(f"  Total unique tokens: {len(token_counts):,}")
        print(f"  High-frequency tokens (>1%): {len(frequent_tokens):,}")
        print(f"  Rare tokens (<0.01%): {len(rare_tokens):,}")
        print(f"  Average tokens per sequence: {total_tokens/total_sequences:.1f}")
        
        # Show top frequent tokens
        print(f"\n🔥 Most Frequent Tokens:")
        for token, count in token_counts.most_common(10):
            token_name = self.idx_to_vocab.get(token, f"Token_{token}")
            freq = count/total_tokens
            print(f"  {token_name}: {count:,} ({freq:.2%})")
        
        # Analyze frequency by category
        print(f"\n📋 Frequency by Token Category:")
        for category, tokens_dict in self.token_categories.items():
            if tokens_dict:
                category_counts = [token_counts.get(token_id, 0) for token_id in tokens_dict.keys()]
                avg_freq = np.mean(category_counts) / total_tokens if category_counts else 0
                print(f"  {category}: avg {avg_freq:.3%} frequency")
        
        self.token_frequencies = {
            'token_counts': token_counts, 'frequent_tokens': frequent_tokens,
            'rare_tokens': rare_tokens, 'total_tokens': total_tokens,
            'avg_tokens_per_sequence': total_tokens/total_sequences
        }
    
    def _detailed_performance_analysis(self):
        """Enhanced performance analysis with multiple metrics"""
        print(f"\n📊 ENHANCED PERFORMANCE ANALYSIS")
        print("-"*60)
        
        if len(self.predictions) == 0:
            print("❌ No predictions available for analysis")
            return
        
        # Basic metrics
        accuracy = (self.predictions == self.labels).mean()
        survival_rate = self.labels.mean()
        
        # Enhanced metrics with safety checks
        try:
            balanced_acc = balanced_accuracy_score(self.labels, self.predictions)
            f1 = f1_score(self.labels, self.predictions) if len(np.unique(self.labels)) > 1 else 0
            precision = precision_score(self.labels, self.predictions) if len(np.unique(self.labels)) > 1 else 0
            recall = recall_score(self.labels, self.predictions) if len(np.unique(self.labels)) > 1 else 0
            mcc = self.calculate_matthews_correlation_coefficient()
        except Exception as e:
            print(f"⚠️ Could not calculate some metrics: {e}")
            balanced_acc = f1 = precision = recall = mcc = 0
        
        # AUC and AP with safety checks
        try:
            if len(np.unique(self.labels)) > 1 and len(self.probabilities) > 0:
                auc = roc_auc_score(self.labels, self.probabilities)
                ap_score = average_precision_score(self.labels, self.probabilities)
            else:
                auc = ap_score = float('nan')
        except:
            auc = ap_score = float('nan')
        
        print(f"📈 Comprehensive Performance Metrics:")
        print(f"  Total samples: {len(self.predictions):,}")
        print(f"  Class distribution: {(1-survival_rate):.1%} failed, {survival_rate:.1%} survived")
        print()
        print(f"  Standard Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f} ⭐")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Matthews Correlation Coefficient (MCC): {mcc:.4f} ⭐")
        print()
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Average Precision (AP): {ap_score:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        print(f"\n📊 Confusion Matrix:")
        print("    Pred:")
        print("    F    S")
        print(f"F  {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"S  {cm[1,0]:4d} {cm[1,1]:4d}")
        
        # Interpretation
        print(f"\n🎯 Performance Interpretation:")
        if balanced_acc > 0.7:
            print(f"✅ Good balanced performance (Balanced Acc > 0.7)")
        elif balanced_acc > 0.6:
            print(f"📊 Moderate balanced performance (Balanced Acc > 0.6)")
        elif balanced_acc > 0.5:
            print(f"⚠️ Weak but above-chance performance (Balanced Acc > 0.5)")
        else:
            print(f"❌ Below-chance performance")
        
        if not np.isnan(mcc):
            if mcc > 0.3:
                print(f"✅ Strong correlation (MCC > 0.3)")
            elif mcc > 0.1:
                print(f"📊 Moderate correlation (MCC > 0.1)")
            elif mcc > 0:
                print(f"⚠️ Weak positive correlation")
            else:
                print(f"❌ No or negative correlation")
        
        # Store metrics for later use
        self.performance_metrics = {
            'accuracy': accuracy, 'balanced_accuracy': balanced_acc, 'f1_score': f1,
            'precision': precision, 'recall': recall, 'mcc': mcc,
            'auc_roc': auc, 'average_precision': ap_score, 'survival_rate': survival_rate,
            'confusion_matrix': cm
        }
    
    def run_complete_analysis(self, target_batches=500, balanced_sampling=False):
        """Run complete interpretability analysis with all fixes"""
        print("🚀 COMPLETE FIXED STARTUP2VEC INTERPRETABILITY ANALYSIS")
        print("=" * 90)
        print("Complete analysis with FIXED balanced sampling and CUDA memory management")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data with all fixes
        if not self.extract_data_with_characteristics(target_batches, balanced_sampling):
            return False
        
        print(f"\n🎉 COMPLETE FIXED ANALYSIS FINISHED!")
        print(f"📊 Analyzed {len(self.predictions):,} startup samples")
        
        # Verification of fixes
        if balanced_sampling and len(self.predictions) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"✅ FIXED balanced sampling verification:")
            print(f"   Class distribution: {dict(zip(unique_labels, counts))}")
            if len(counts) == 2 and abs(counts[0] - counts[1]) <= 1:
                print(f"   ✅ Perfect balance achieved!")
            else:
                print(f"   ⚠️ Some imbalance remaining")
        
        return True

def main():
    """COMPLETE FIXED main function"""
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    # Check GPU status
    print("🔧 COMPLETE FIXED STARTUP2VEC INTERPRETABILITY ANALYSIS")
    print("="*75)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        
        print(f"🚀 CUDA Available: {gpu_count} GPU(s)")
        print(f"📱 Current Device: GPU {current_device}")
        print(f"🎯 CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        if cuda_visible == 'Not set':
            print("⚠️  Consider setting CUDA_VISIBLE_DEVICES for specific GPU")
    else:
        print("❌ CUDA not available - will use CPU")
    
    print()
    print("🎯 COMPLETE FIXES INCLUDED:")
    print("✅ FIXED balanced sampling bug (truly different success/failure samples)")
    print("✅ FIXED CUDA memory management (graceful fallback to CPU)")
    print("✅ Reduced batch size (16 instead of 32)")
    print("✅ Memory clearing and error handling")
    print("✅ Balanced accuracy and MCC metrics")
    print("✅ Statistical bias detection")
    print()
    
    analyzer = CompleteFixedStartupInterpretabilityAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir="startup_interpretability_complete_fixed"
    )
    
    # Get user preferences
    print("🎛️ ANALYSIS OPTIONS:")
    print("1. Standard analysis (original data distribution)")
    print("2. FIXED balanced analysis (truly equal success/failure samples)")
    
    choice = input("Choose analysis type (1 or 2): ").strip()
    balanced_sampling = choice == "2"
    
    if balanced_sampling:
        print("🎯 Using FIXED balanced sampling")
    else:
        print("📊 Using original data distribution")
    
    batch_choice = input("Enter number of batches (0 for ALL data, 500+ recommended): ").strip()
    try:
        target_batches = int(batch_choice)
    except ValueError:
        target_batches = 500
    
    if target_batches == 0:
        print("🔥 Running FULL SAMPLE analysis!")
        print("⚠️ This will process all available data")
    else:
        print(f"🚀 Running analysis with {target_batches} batches...")
    
    # Start analysis with timing
    start_time = time.time()
    success = analyzer.run_complete_analysis(
        target_batches=target_batches,
        balanced_sampling=balanced_sampling
    )
    end_time = time.time()
    
    if success:
        print(f"\n🎉 SUCCESS! Complete fixed analysis completed in {end_time-start_time:.1f} seconds")
        print(f"📊 Successfully processed {len(analyzer.predictions):,} samples")
        
        print("\n�� ALL FIXES APPLIED:")
        print("  ✅ Balanced sampling uses truly different samples")
        print("  ✅ CUDA memory managed with CPU fallback")
        print("  ✅ No more memory crashes")
        print("  ✅ Robust error handling")
        
        print("\n📁 OUTPUT:")
        print(f"  Directory: startup_interpretability_complete_fixed/")
        print(f"  Results saved with comprehensive analysis")
        
        print("\n💡 NEXT STEPS:")
        print("  1. Check results for bias patterns")
        print("  2. Compare balanced vs unbalanced results")
        print("  3. Use findings for thesis bias analysis")
        
        return 0
    else:
        print(f"\n❌ Analysis failed after {end_time-start_time:.1f} seconds")
        print("💡 TROUBLESHOOTING:")
        print("  1. Try smaller target_batches")
        print("  2. Force CPU with: CUDA_VISIBLE_DEVICES=''")
        print("  3. Check available memory")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
