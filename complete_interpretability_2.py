#!/usr/bin/env python3
"""
STARTUP2VEC ENHANCED INTERPRETABILITY ANALYSIS
Complete interpretability analysis with improved metrics and visualizations
"""

import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
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

class EnhancedStartupInterpretabilityAnalyzer:
    """Enhanced interpretability analyzer with improved metrics and visualizations"""
    
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
    
    def load_model_and_data(self):
        """Load model and data with vocabulary"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        
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
        """Extract data with option for balanced sampling"""
        print(f"\n🎯 EXTRACTING DATA WITH TOKEN ANALYSIS")
        print("="*60)
        
        if balanced_sampling:
            print("🎯 Using balanced sampling strategy...")
            return self._extract_balanced_data(target_batches)
        else:
            print("📊 Using original data distribution...")
            return self._extract_standard_data(target_batches)
    
    def _extract_standard_data(self, target_batches):
        """Standard data extraction"""
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
        
        print(f"Processing {max_batches:,} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{max_batches}", end='\r')
                
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
                    
                    # Store results
                    all_predictions.extend(survival_preds.cpu().numpy())
                    all_probabilities.extend(survival_probs.cpu().numpy())
                    all_labels.extend(survival_labels.squeeze().cpu().numpy())
                    all_embeddings.extend(company_embeddings.cpu().numpy())
                    all_sequences.extend(input_ids[:, 0, :].cpu().numpy())
                    
                    # Extract metadata with token parsing
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_metadata(batch, i, input_ids[i, 0, :])
                        all_metadata.append(metadata)
                    
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {e}")
                    continue
        
        print(f"\n✅ Data extraction complete: {len(all_predictions):,} samples")
        
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
    
    def _extract_balanced_data(self, target_batches):
        """Extract balanced data (equal survival/failure samples)"""
        val_loader = self.datamodule.val_dataloader()
        
        survival_samples = []
        failure_samples = []
        target_per_class = target_batches * 16  # Assuming batch size ~32, aim for equal splits
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        
        print(f"Collecting balanced samples (target: {target_per_class} per class)...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if len(survival_samples) >= target_per_class and len(failure_samples) >= target_per_class:
                    break
                
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
                    
                    # Separate by class
                    for i in range(input_ids.size(0)):
                        sample_data = {
                            'prediction': survival_preds[i].cpu().numpy(),
                            'probability': survival_probs[i].cpu().numpy(),
                            'label': survival_labels[i].squeeze().cpu().numpy(),
                            'embedding': company_embeddings[i].cpu().numpy(),
                            'sequence': input_ids[i, 0, :].cpu().numpy(),
                            'metadata': self._extract_metadata(batch, i, input_ids[i, 0, :])
                        }
                        
                        if survival_labels[i].item() == 1 and len(survival_samples) < target_per_class:
                            survival_samples.append(sample_data)
                        elif survival_labels[i].item() == 0 and len(failure_samples) < target_per_class:
                            failure_samples.append(sample_data)
                
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {e}")
                    continue
        
        # Combine balanced samples
        min_samples = min(len(survival_samples), len(failure_samples))
        balanced_samples = survival_samples[:min_samples] + failure_samples[:min_samples]
        
        print(f"\n✅ Balanced sampling complete: {len(balanced_samples):,} samples ({min_samples} per class)")
        
        # Reorganize data
        self.predictions = np.array([s['prediction'] for s in balanced_samples])
        self.probabilities = np.array([s['probability'] for s in balanced_samples])
        self.labels = np.array([s['label'] for s in balanced_samples])
        self.embeddings = np.array([s['embedding'] for s in balanced_samples])
        self.sequences = [s['sequence'] for s in balanced_samples]
        self.metadata = [s['metadata'] for s in balanced_samples]
        
        # Additional analyses
        self._parse_startup_characteristics()
        self._analyze_token_frequency()
        self._detailed_performance_analysis()
        
        return True
    
    def _analyze_token_frequency(self):
        """Analyze token frequency patterns"""
        print("\n🔍 TOKEN FREQUENCY ANALYSIS")
        print("-" * 40)
        
        token_counts = Counter()
        total_sequences = len(self.sequences)
        
        for sequence in self.sequences:
            clean_sequence = sequence[sequence > 0]
            for token in clean_sequence:
                token_counts[int(token)] += 1
        
        # Calculate frequencies
        total_tokens = sum(token_counts.values())
        
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
            'token_counts': token_counts,
            'frequent_tokens': frequent_tokens,
            'rare_tokens': rare_tokens,
            'total_tokens': total_tokens,
            'avg_tokens_per_sequence': total_tokens/total_sequences
        }
    
    def _extract_metadata(self, batch, sample_idx, sequence):
        """Extract metadata including characteristics parsed from tokens"""
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
    
    def _parse_sequence_characteristics(self, sequence):
        """Parse sequence to extract startup characteristics from actual tokens"""
        characteristics = {
            # Company characteristics
            'country': 'Unknown',
            'industry_category': 'Unknown',
            'employee_size': 'Unknown',
            'business_model': 'Unknown',
            'technology_type': 'Unknown',
            
            # Event presence
            'has_investment_events': False,
            'has_acquisition_events': False,
            'has_ipo_events': False,
            'has_education_events': False,
            'has_people_events': False,
            
            # Event counts
            'investment_event_count': 0,
            'people_job_count': 0,
            'education_event_count': 0,
            
            # Sequence stats
            'unique_token_count': 0,
            'token_diversity': 0.0
        }
        
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
        
        return characteristics
    
    def _parse_startup_characteristics(self):
        """Parse all startup characteristics from token metadata"""
        print("\n📋 Parsing startup characteristics from tokens...")
        
        # Count characteristics from tokens
        countries = Counter([m['country'] for m in self.metadata])
        industries = Counter([m['industry_category'] for m in self.metadata])
        employee_sizes = Counter([m['employee_size'] for m in self.metadata])
        business_models = Counter([m['business_model'] for m in self.metadata])
        tech_types = Counter([m['technology_type'] for m in self.metadata])
        
        # Age and temporal analysis
        ages = [m['company_age'] for m in self.metadata]
        years = [m['founded_year'] for m in self.metadata]
        
        print(f"\n📊 Startup Characteristics Found:")
        print(f"  Countries: {dict(countries.most_common(10))}")
        print(f"  Industries: {dict(industries.most_common(10))}")
        print(f"  Employee Sizes: {dict(employee_sizes)}")
        print(f"  Business Models: {dict(business_models)}")
        print(f"  Technology Types: {dict(tech_types.most_common(5))}")
        
        # Temporal analysis
        print(f"\n⏰ Temporal Characteristics:")
        print(f"  Average company age: {np.mean(ages):.1f} years")
        print(f"  Age range: {np.min(ages):.1f} - {np.max(ages):.1f} years")
        print(f"  Founded year range: {np.min(years)} - {np.max(years)}")
        
        # Age-based survival analysis
        age_groups = pd.cut(ages, bins=5, labels=['Very Young', 'Young', 'Medium', 'Mature', 'Old'])
        print(f"\n📈 Survival by Age Group:")
        for group in age_groups.categories:
            mask = age_groups == group
            if mask.sum() > 10:
                survival_rate = self.labels[mask].mean()
                print(f"  {group}: {survival_rate:.2%} survival rate ({mask.sum()} companies)")
        
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
            'countries': countries,
            'industries': industries,
            'employee_sizes': employee_sizes,
            'business_models': business_models,
            'tech_types': tech_types,
            'temporal': {
                'ages': ages,
                'years': years,
                'age_groups': age_groups
            },
            'event_stats': {
                'investment_events': investment_events,
                'acquisition_events': acquisition_events,
                'ipo_events': ipo_events,
                'education_events': education_events
            }
        }
    
    def _detailed_performance_analysis(self):
        """Enhanced performance analysis with multiple metrics"""
        print(f"\n�� ENHANCED PERFORMANCE ANALYSIS")
        print("-"*60)
        
        # Basic metrics
        accuracy = (self.predictions == self.labels).mean()
        survival_rate = self.labels.mean()
        
        # Enhanced metrics
        balanced_acc = balanced_accuracy_score(self.labels, self.predictions)
        f1 = f1_score(self.labels, self.predictions)
        precision = precision_score(self.labels, self.predictions)
        recall = recall_score(self.labels, self.predictions)
        mcc = self.calculate_matthews_correlation_coefficient()
        
        # AUC and AP
        try:
            if len(np.unique(self.labels)) > 1:
                auc = roc_auc_score(self.labels, self.probabilities)
                ap_score = average_precision_score(self.labels, self.probabilities)
            else:
                auc = float('nan')
                ap_score = float('nan')
        except:
            auc = float('nan')
            ap_score = float('nan')
        
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
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'auc_roc': auc,
            'average_precision': ap_score,
            'survival_rate': survival_rate,
            'confusion_matrix': cm
        }
    
    def enhanced_bias_detection(self, audit_results):
        """Enhanced statistical bias detection"""
        print(f"\n⚖️ ENHANCED BIAS DETECTION")
        print("-" * 40)
        
        bias_issues = []
        statistical_tests = {}
        
        for category, results in audit_results.items():
            if len(results) <= 1:
                continue
            
            print(f"\n🔍 Analyzing {category}:")
            
            # Collect data for statistical testing
            groups_predictions = []
            groups_labels = []
            group_names = []
            
            accuracies = []
            survival_rates = []
            
            for result in results:
                cat_name = result['category']
                
                # Find samples for this category
                if category == 'employee_size':
                    mask = np.array([m['employee_size'] == cat_name for m in self.metadata])
                elif category == 'industry':
                    mask = np.array([m['industry_category'] == cat_name for m in self.metadata])
                elif category == 'country':
                    mask = np.array([m['country'] == cat_name for m in self.metadata])
                elif category == 'business_model':
                    mask = np.array([m['business_model'] == cat_name for m in self.metadata])
                elif category == 'technology_type':
                    mask = np.array([m['technology_type'] == cat_name for m in self.metadata])
                else:
                    continue
                
                if mask.sum() > 10:  # Minimum sample size
                    groups_predictions.append(self.probabilities[mask])
                    groups_labels.append(self.labels[mask])
                    group_names.append(cat_name)
                    
                    accuracies.append(result['accuracy'])
                    survival_rates.append(result['survival_rate'])
            
            if len(groups_predictions) > 1:
                # Statistical tests
                try:
                    # One-way ANOVA for prediction probabilities
                    f_stat, p_val = stats.f_oneway(*groups_predictions)
                    
                    # Chi-square test for survival rates
                    observed = np.array([[group.sum(), len(group) - group.sum()] for group in groups_labels])
                    if observed.min() >= 5:  # Chi-square assumption
                        chi2, chi2_p = stats.chi2_contingency(observed)[:2]
                    else:
                        chi2, chi2_p = np.nan, np.nan
                    
                    statistical_tests[category] = {
                        'anova_f': f_stat,
                        'anova_p': p_val,
                        'chi2': chi2,
                        'chi2_p': chi2_p
                    }
                    
                    print(f"  📊 Statistical Tests:")
                    print(f"    ANOVA (predictions): F={f_stat:.3f}, p={p_val:.4f}")
                    if not np.isnan(chi2_p):
                        print(f"    Chi-square (outcomes): χ²={chi2:.3f}, p={chi2_p:.4f}")
                    
                    # Significance assessment
                    if p_val < 0.001:
                        significance = "***"
                        bias_issues.append(f"{category}: highly significant bias (p<0.001)")
                    elif p_val < 0.01:
                        significance = "**"
                        bias_issues.append(f"{category}: significant bias (p<0.01)")
                    elif p_val < 0.05:
                        significance = "*"
                        bias_issues.append(f"{category}: potential bias (p<0.05)")
                    else:
                        significance = "ns"
                    
                    print(f"    Significance: {significance}")
                    
                except Exception as e:
                    print(f"    ⚠️ Statistical test failed: {e}")
            
            # Effect size analysis
            if len(accuracies) > 1:
                acc_range = max(accuracies) - min(accuracies)
                survival_range = max(survival_rates) - min(survival_rates)
                
                print(f"  📏 Effect Sizes:")
                print(f"    Accuracy range: {acc_range:.1%}")
                print(f"    Survival rate range: {survival_range:.1%}")
                
                if acc_range > 0.05:  # 5% threshold
                    bias_issues.append(f"{category}: large accuracy gap ({acc_range:.1%})")
                elif acc_range > 0.03:
                    bias_issues.append(f"{category}: moderate accuracy gap ({acc_range:.1%})")
                
                if survival_range > 0.10:  # 10% threshold
                    bias_issues.append(f"{category}: large survival rate gap ({survival_range:.1%})")
                elif survival_range > 0.05:
                    bias_issues.append(f"{category}: moderate survival rate gap ({survival_range:.1%})")
        
        # Summary
        print(f"\n📋 Bias Detection Summary:")
        if bias_issues:
            print("  ⚠️ Potential biases detected:")
            for issue in bias_issues:
                print(f"    - {issue}")
        else:
            print("  ✅ No significant biases detected")
        
        return {
            'bias_issues': bias_issues,
            'statistical_tests': statistical_tests
        }
    
    def algorithmic_auditing(self):
        """1. Enhanced Algorithmic Auditing"""
        print(f"\n🔍 1. ENHANCED ALGORITHMIC AUDITING")
        print("="*70)
        print("Examining model performance across startup subgroups with statistical testing...")
        
        results = {}
        
        # Analyze subgroups
        subgroup_analyses = [
            ('employee_size', [m['employee_size'] for m in self.metadata], 'Employee Size'),
            ('industry', [m['industry_category'] for m in self.metadata], 'Industry'),
            ('country', [m['country'] for m in self.metadata], 'Country'),
            ('business_model', [m['business_model'] for m in self.metadata], 'Business Model'),
            ('technology_type', [m['technology_type'] for m in self.metadata], 'Technology Type'),
        ]
        
        for key, categories, name in subgroup_analyses:
            print(f"\n👥 Performance by {name}:")
            
            # For countries, group smaller ones as 'Other'
            if key == 'country':
                country_counts = Counter(categories)
                major_countries = [country for country, count in country_counts.items() if count >= 50]
                categories = [country if country in major_countries else 'Other' for country in categories]
            
            subgroup_results = self._analyze_subgroup_performance(categories, name)
            results[key] = subgroup_results
        
        # Investment event analysis
        print(f"\n🎯 Performance by Investment Event Presence:")
        investment_presence = ['Has Investment Events' if m['has_investment_events'] else 'No Investment Events' 
                             for m in self.metadata]
        investment_results = self._analyze_subgroup_performance(investment_presence, 'Investment Events')
        results['investment_events'] = investment_results
        
        # Enhanced bias detection with statistical testing
        bias_analysis = self.enhanced_bias_detection(results)
        results['bias_analysis'] = bias_analysis
        
        self.algorithmic_audit_results = results
    
    def _analyze_subgroup_performance(self, categories, category_name):
        """Analyze performance across subgroups with enhanced metrics"""
        unique_cats = list(set(categories))
        results = []
        
        for cat in unique_cats:
            mask = np.array([c == cat for c in categories])
            if mask.sum() >= 20:  # Minimum sample size
                cat_preds = self.predictions[mask]
                cat_probs = self.probabilities[mask]
                cat_labels = self.labels[mask]
                
                cat_acc = (cat_preds == cat_labels).mean()
                cat_survival = cat_labels.mean()
                cat_pred_rate = cat_probs.mean()
                cat_count = mask.sum()
                
                # Enhanced metrics
                if len(np.unique(cat_labels)) > 1:
                    cat_balanced_acc = balanced_accuracy_score(cat_labels, cat_preds)
                    cat_f1 = f1_score(cat_labels, cat_preds)
                else:
                    cat_balanced_acc = cat_acc
                    cat_f1 = 0.0
                
                results.append({
                    'category': cat,
                    'count': cat_count,
                    'accuracy': cat_acc,
                    'balanced_accuracy': cat_balanced_acc,
                    'f1_score': cat_f1,
                    'survival_rate': cat_survival,
                    'pred_rate': cat_pred_rate
                })
                
                print(f"  {cat}: {cat_count:5,} samples | "
                      f"Acc: {cat_acc:.2%} | "
                      f"Bal-Acc: {cat_balanced_acc:.2%} | "
                      f"Survival: {cat_survival:.2%}")
        
        return results
    
    def data_contribution_analysis(self):
        """2. Enhanced Data Contribution Analysis"""
        print(f"\n📊 2. ENHANCED DATA CONTRIBUTION ANALYSIS")
        print("="*70)
        print("Analyzing contribution of event categories and individual tokens...")
        
        contribution_results = {}
        
        # Event category analysis
        print(f"\n🔍 Event Category Contribution Analysis:")
        
        category_order = [
            ('company_country', 'Company Country'),
            ('company_category', 'Company Category'),
            ('company_employee_size', 'Company Employee Size'),
            ('investment_types', 'Investment Types'),
            ('acquisition_types', 'Acquisition Types'),
            ('education_degree_type', 'Education - Degree Types'),
            ('event_types', 'Event Types'),
        ]
        
        for category_key, category_name in category_order:
            if category_key not in self.token_categories or len(self.token_categories[category_key]) == 0:
                continue
                
            print(f"\n📋 Analyzing {category_name}:")
            
            token_dict = self.token_categories[category_key]
            
            # Calculate presence
            category_presence = []
            for sequence in self.sequences:
                clean_sequence = sequence[sequence > 0]
                has_category = any(int(token) in token_dict for token in clean_sequence)
                category_presence.append(has_category)
            
            category_presence = np.array(category_presence)
            
            # Enhanced statistical analysis
            if category_presence.sum() > 0 and (~category_presence).sum() > 0:
                survived_with = self.labels[category_presence].mean()
                survived_without = self.labels[~category_presence].mean()
                contribution_score = survived_with - survived_without
                
                # Statistical significance test
                try:
                    from scipy.stats import chi2_contingency
                    
                    # Create contingency table
                    contingency = np.array([
                        [self.labels[category_presence].sum(), category_presence.sum() - self.labels[category_presence].sum()],
                        [self.labels[~category_presence].sum(), (~category_presence).sum() - self.labels[~category_presence].sum()]
                    ])
                    
                    chi2, p_val, _, _ = chi2_contingency(contingency)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    
                except:
                    chi2, p_val, significance = np.nan, np.nan, "na"
                
                print(f"  Companies with {category_name}: {category_presence.sum():,} ({category_presence.mean()*100:.1f}%)")
                print(f"  Survival rate WITH: {survived_with:.2%}")
                print(f"  Survival rate WITHOUT: {survived_without:.2%}")
                print(f"  Contribution score: {contribution_score:+.3f} {significance}")
                
                contribution_results[category_key] = {
                    'category_name': category_name,
                    'presence_count': int(category_presence.sum()),
                    'presence_rate': float(category_presence.mean()),
                    'survival_with': float(survived_with),
                    'survival_without': float(survived_without),
                    'contribution_score': float(contribution_score),
                    'chi2': float(chi2) if not np.isnan(chi2) else None,
                    'p_value': float(p_val) if not np.isnan(p_val) else None,
                    'significance': significance,
                    'token_count': len(token_dict)
                }
        
        # Individual token analysis
        print(f"\n🔤 Individual Token Analysis:")
        self._analyze_individual_tokens_enhanced()
        
        # Rank by absolute contribution
        ranked_contributions = sorted(contribution_results.items(), 
                                    key=lambda x: abs(x[1]['contribution_score']), 
                                    reverse=True)
        
        print(f"\n🏆 Event Category Importance Ranking:")
        for i, (category_key, scores) in enumerate(ranked_contributions, 1):
            direction = "survival ✅" if scores['contribution_score'] > 0 else "failure ❌"
            sig = scores.get('significance', 'na')
            print(f"  {i}. {scores['category_name']}: {scores['contribution_score']:+.3f} {sig} (→ {direction})")
        
        self.data_contribution_results = {
            'category_contributions': contribution_results,
            'ranking': ranked_contributions
        }
    
    def _analyze_individual_tokens_enhanced(self):
        """Enhanced individual token analysis with statistical testing"""
        key_categories = [
            ('investment_types', 'Investment Types'),
            ('company_country', 'Company Countries'), 
        ]
        
        for category_key, category_name in key_categories:
            if category_key not in self.token_categories:
                continue
                
            print(f"\n🧮 Individual tokens in {category_name}:")
            token_dict = self.token_categories[category_key]
            
            token_contributions = []
            
            for token_id, token_str in token_dict.items():
                # Calculate presence for this token
                token_presence = []
                for sequence in self.sequences:
                    clean_sequence = sequence[sequence > 0]
                    has_token = int(token_id) in clean_sequence
                    token_presence.append(has_token)
                
                token_presence = np.array(token_presence)
                
                if token_presence.sum() >= 5:  # Minimum occurrences
                    survived_with = self.labels[token_presence].mean() if token_presence.sum() > 0 else 0
                    survived_without = self.labels[~token_presence].mean() if (~token_presence).sum() > 0 else 0
                    contribution = survived_with - survived_without
                    
                    token_contributions.append({
                        'token': token_str,
                        'contribution': contribution,
                        'count': token_presence.sum(),
                        'survival_with': survived_with,
                        'survival_without': survived_without
                    })
            
            # Sort and display
            if token_contributions:
                sorted_tokens = sorted(token_contributions, key=lambda x: x['contribution'], reverse=True)
                
                print(f"    Top survival predictors:")
                for token_data in sorted_tokens[:5]:
                    print(f"      {token_data['token']}: {token_data['contribution']:+.4f} "
                          f"({token_data['count']} occurrences)")
                
                print(f"    Top failure predictors:")
                for token_data in sorted_tokens[-3:]:
                    print(f"      {token_data['token']}: {token_data['contribution']:+.4f} "
                          f"({token_data['count']} occurrences)")
    
    def visual_exploration(self):
        """3. Enhanced Visual Exploration"""
        print(f"\n🎨 3. ENHANCED VISUAL EXPLORATION")
        print("="*60)
        print("Creating life2vec-style arch visualization...")
        
        # Install UMAP if needed
        try:
            import umap.umap_ as umap
            umap_available = True
        except ImportError:
            print("⚠️ Installing UMAP for visualization...")
            try:
                os.system("pip install umap-learn")
                import umap.umap_ as umap
                umap_available = True
                print("✅ UMAP installed successfully")
            except:
                print("❌ Could not install UMAP - using t-SNE only")
                umap_available = False
        
        # Sample for visualization
        max_viz = min(3000, len(self.embeddings))
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
        
        print(f"Using {len(viz_embeddings):,} samples for visualization...")
        
        # Create dimensionality reductions
        print("📊 Creating PCA projection...")
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(viz_embeddings)
        
        umap_embeddings = None
        if umap_available:
            print("🗺️ Creating UMAP projection for arch...")
            try:
                umap_reducer = umap.UMAP(
                    n_components=2, 
                    n_neighbors=30,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                umap_embeddings = umap_reducer.fit_transform(viz_embeddings)
                print("✅ UMAP arch created successfully")
            except Exception as e:
                print(f"⚠️ UMAP failed: {e}")
        
        # Store results
        self.visual_exploration_results = {
            'pca_embeddings': pca_embeddings,
            'umap_embeddings': umap_embeddings,
            'viz_probabilities': viz_probs,
            'viz_labels': viz_labels,
            'viz_metadata': viz_metadata,
            'sample_size': len(viz_embeddings)
        }
        
        # Create life2vec-style visualization
        self._create_life2vec_style_visualization()
    
    def _create_life2vec_style_visualization(self):
        """Create life2vec-style arch visualization with clear regions"""
        if not hasattr(self, 'visual_exploration_results'):
            return
        
        viz_data = self.visual_exploration_results
        
        if viz_data['umap_embeddings'] is None:
            print("⚠️ UMAP not available, skipping life2vec-style visualization")
            return
        
        # Create figure similar to life2vec
        fig = plt.figure(figsize=(16, 12))
        
        # Main arch in center
        ax_main = plt.subplot2grid((3, 5), (0, 1), colspan=3, rowspan=3)
        
        # Scatter plot with survival probability
        scatter = ax_main.scatter(
            viz_data['umap_embeddings'][:, 0], 
            viz_data['umap_embeddings'][:, 1],
            c=viz_data['viz_probabilities'], 
            cmap='RdYlBu_r', 
            alpha=0.7, 
            s=12
        )
        
        # Highlight true deceased (failed startups)
        failed_mask = viz_data['viz_labels'] == 0
        if failed_mask.sum() > 0:
            ax_main.scatter(
                viz_data['umap_embeddings'][failed_mask, 0], 
                viz_data['umap_embeddings'][failed_mask, 1],
                c='red', 
                marker='x', 
                s=20, 
                alpha=0.8,
                label='True Failed'
            )
        
        ax_main.set_title('Startup Embedding Space\n(projected with UMAP)', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_main, label='Predicted Probability')
        cbar.set_label('Survival Probability', fontsize=12)
        
        # Add legend for failed startups
        if failed_mask.sum() > 0:
            ax_main.legend(loc='upper right')
        
        # Side panels for different characteristics
        characteristics = [
            ('Country', [m['country'] for m in viz_data['viz_metadata']]),
            ('Industry', [m['industry_category'] for m in viz_data['viz_metadata']]),
            ('Employee Size', [m['employee_size'] for m in viz_data['viz_metadata']]),
            ('Business Model', [m['business_model'] for m in viz_data['viz_metadata']])
        ]
        
        panel_positions = [
            (0, 0), (1, 0),  # Left panels
            (0, 4), (1, 4)   # Right panels
        ]
        
        for i, ((char_name, char_values), (row, col)) in enumerate(zip(characteristics, panel_positions)):
            ax = plt.subplot2grid((3, 5), (row, col))
            
            # Create categorical colors
            unique_values = list(set(char_values))
            if len(unique_values) > 10:  # Too many categories, group others
                value_counts = Counter(char_values)
                top_values = [v for v, c in value_counts.most_common(8)]
                char_values = [v if v in top_values else 'Other' for v in char_values]
                unique_values = list(set(char_values))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
            color_map = {val: colors[i] for i, val in enumerate(unique_values)}
            point_colors = [color_map[val] for val in char_values]
            
            ax.scatter(
                viz_data['umap_embeddings'][:, 0], 
                viz_data['umap_embeddings'][:, 1],
                c=point_colors, 
                alpha=0.6, 
                s=8
            )
            
            ax.set_title(f'{char_name}', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add small legend if not too many categories
            if len(unique_values) <= 6:
                for j, (val, color) in enumerate(color_map.items()):
                    if j < 6:  # Limit legend entries
                        ax.scatter([], [], c=[color], label=val, s=20)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Bottom panel for true labels
        ax_bottom = plt.subplot2grid((3, 5), (2, 1), colspan=3)
        
        # Color by true outcomes
        true_colors = ['red' if label == 0 else 'green' for label in viz_data['viz_labels']]
        ax_bottom.scatter(
            viz_data['umap_embeddings'][:, 0], 
            viz_data['umap_embeddings'][:, 1],
            c=true_colors, 
            alpha=0.6, 
            s=8
        )
        ax_bottom.set_title('True Survival Outcomes', fontweight='bold')
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        
        # Add legend
        ax_bottom.scatter([], [], c='green', label='Survived', s=20)
        ax_bottom.scatter([], [], c='red', label='Failed', s=20)
        ax_bottom.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'startup_life2vec_style.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Life2vec-style visualization created")
    
    def local_explainability(self):
        """4. Enhanced Local Explainability"""
        print(f"\n🔍 4. ENHANCED LOCAL EXPLAINABILITY")
        print("="*70)
        print("Analyzing individual startup trajectories with detailed token analysis...")
        
        # Select diverse examples
        examples = self._select_diverse_examples()
        
        local_results = {}
        
        for example_type, indices in examples.items():
            if len(indices) == 0:
                continue
                
            print(f"\n🎯 Analyzing {example_type.replace('_', ' ').title()} ({len(indices)} examples):")
            
            example_analyses = []
            for i, idx in enumerate(indices[:5]):  # Top 5 examples
                analysis = self._analyze_individual_startup_enhanced(idx)
                example_analyses.append(analysis)
                
                # Enhanced display
                meta = self.metadata[idx]
                print(f"  Example {i+1} (Sample {idx}):")
                print(f"    Survival Probability: {self.probabilities[idx]:.3f}")
                print(f"    True Outcome: {'Survived' if self.labels[idx] == 1 else 'Failed'}")
                print(f"    Company: {meta['country']} | {meta['industry_category']} | {meta['employee_size']} employees")
                print(f"    Business: {meta['business_model']} model | {meta['technology_type']} tech")
                print(f"    Events: {analysis['event_summary']}")
                print(f"    Key Tokens: {', '.join(analysis['top_tokens'][:3])}")
            
            local_results[example_type] = example_analyses
        
        self.local_explainability_results = local_results
    
    def _analyze_individual_startup_enhanced(self, idx):
        """Enhanced individual startup analysis"""
        meta = self.metadata[idx]
        sequence = self.sequences[idx]
        clean_sequence = sequence[sequence > 0]
        
        # Categorize all tokens
        token_categories = {
            'company': [],
            'investment': [],
            'acquisition': [],
            'education': [],
            'people': [],
            'events': [],
            'temporal': []
        }
        
        all_tokens = []
        for token in clean_sequence:
            token_str = self.idx_to_vocab.get(int(token), f"Token_{token}")
            all_tokens.append(token_str)
            
            # Categorize
            if any(token_str.startswith(prefix) for prefix in ['COUNTRY_', 'INDUSTRY_', 'MODEL_', 'TECH_', 'EMPLOYEE_']):
                token_categories['company'].append(token_str)
            elif token_str.startswith('INV_'):
                token_categories['investment'].append(token_str)
            elif token_str.startswith('ACQ_'):
                token_categories['acquisition'].append(token_str)
            elif token_str.startswith('EDU_'):
                token_categories['education'].append(token_str)
            elif token_str.startswith('PPL_'):
                token_categories['people'].append(token_str)
            elif token_str.startswith('EVT_'):
                token_categories['events'].append(token_str)
            elif token_str.startswith('DAYS_'):
                token_categories['temporal'].append(token_str)
        
        # Create event summary
        event_counts = {k: len(v) for k, v in token_categories.items() if v}
        event_summary = ", ".join([f"{k}({v})" for k, v in event_counts.items()])
        
        # Get most frequent tokens
        token_counter = Counter(all_tokens)
        top_tokens = [token for token, count in token_counter.most_common(10)]
        
        return {
            'index': idx,
            'probability': float(self.probabilities[idx]),
            'prediction': int((self.probabilities[idx] > 0.5)),
            'true_label': int(self.labels[idx]),
            'characteristics': {
                'country': meta['country'],
                'industry_category': meta['industry_category'],
                'employee_size': meta['employee_size'],
                'business_model': meta['business_model'],
                'technology_type': meta['technology_type'],
                'company_age': meta['company_age'],
                'sequence_length': meta['sequence_length']
            },
            'event_analysis': {
                'has_investment_events': meta['has_investment_events'],
                'has_acquisition_events': meta['has_acquisition_events'],
                'has_ipo_events': meta['has_ipo_events'],
                'has_education_events': meta['has_education_events'],
                'investment_event_count': meta['investment_event_count'],
                'people_job_count': meta['people_job_count'],
                'education_event_count': meta['education_event_count']
            },
            'token_categories': token_categories,
            'event_summary': event_summary,
            'top_tokens': top_tokens,
            'event_counts': event_counts
        }
    
    def _select_diverse_examples(self):
        """Select diverse startup examples for analysis"""
        examples = {
            'high_confidence_successes': [],
            'high_confidence_failures': [],
            'surprising_successes': [],
            'surprising_failures': [],
            'uncertain_predictions': [],
            'well_funded_startups': [],
            'large_companies': [],
            'tech_companies': [],
            'international_companies': []
        }
        
        for i in range(len(self.probabilities)):
            prob = self.probabilities[i]
            true = self.labels[i]
            meta = self.metadata[i]
            
            # Confidence-based categories
            if prob > 0.9 and true == 1:
                examples['high_confidence_successes'].append(i)
            elif prob < 0.1 and true == 0:
                examples['high_confidence_failures'].append(i)
            elif prob < 0.3 and true == 1:
                examples['surprising_successes'].append(i)
            elif prob > 0.8 and true == 0:
                examples['surprising_failures'].append(i)
            elif 0.4 < prob < 0.6:
                examples['uncertain_predictions'].append(i)
            
            # Characteristic-based categories
            if meta['has_investment_events']:
                examples['well_funded_startups'].append(i)
            if meta['employee_size'] in ['501-1000', '5001-10000']:
                examples['large_companies'].append(i)
            if 'TECH' in meta['industry_category'].upper() or 'SAAS' in meta['business_model'].upper():
                examples['tech_companies'].append(i)
            if meta['country'] not in ['USA', 'Unknown']:
                examples['international_companies'].append(i)
        
        # Limit examples
        for key in examples:
            examples[key] = examples[key][:10]
        
        return examples
    
    def global_explainability(self):
        """5. Enhanced Global Explainability"""
        print(f"\n🌐 5. ENHANCED GLOBAL EXPLAINABILITY")
        print("="*70)
        print("Testing operationalized concepts with statistical validation...")
        
        # Define enhanced operationalized concepts
        concepts = {
            'Well Funded': self._test_well_funded_concept,
            'Technology Focus': self._test_technology_concept,
            'Large Company': self._test_large_company_concept,
            'High Activity': self._test_high_activity_concept,
            'B2B Business Model': self._test_b2b_concept,
            'US-Based': self._test_us_based_concept,
            'Education-Heavy': self._test_education_concept,
            'International': self._test_international_concept,
            'Mature Company': self._test_mature_company_concept,
            'Diverse Events': self._test_diverse_events_concept
        }
        
        concept_scores = {}
        concept_stats = {}
        
        for concept_name, concept_test in concepts.items():
            print(f"\n🧠 Testing concept: {concept_name}")
            
            try:
                concept_result = concept_test()
                if isinstance(concept_result, dict):
                    concept_score = concept_result['score']
                    concept_stats[concept_name] = concept_result
                else:
                    concept_score = concept_result
                    concept_stats[concept_name] = {'score': concept_score}
                
                concept_scores[concept_name] = concept_score
                
                # Enhanced interpretation
                if abs(concept_score) > 0.4:
                    strength = "very strong"
                elif abs(concept_score) > 0.3:
                    strength = "strong"
                elif abs(concept_score) > 0.15:
                    strength = "moderate"
                elif abs(concept_score) > 0.05:
                    strength = "weak"
                else:
                    strength = "negligible"
                
                direction = "survival" if concept_score > 0 else "failure"
                
                print(f"  Score: {concept_score:+.3f} ({strength} association with {direction})")
                
                # Show statistical details if available
                if concept_name in concept_stats and 'p_value' in concept_stats[concept_name]:
                    p_val = concept_stats[concept_name]['p_value']
                    if p_val < 0.001:
                        sig = "***"
                    elif p_val < 0.01:
                        sig = "**"
                    elif p_val < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    print(f"  Statistical significance: {sig} (p={p_val:.4f})")
                
            except Exception as e:
                print(f"  ⚠️ Concept test failed: {e}")
                concept_scores[concept_name] = 0.0
        
        # Rank concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🏆 Global Concept Importance Ranking:")
        for i, (concept, score) in enumerate(sorted_concepts, 1):
            direction = "survival ✅" if score > 0 else "failure ❌"
            if abs(score) > 0.4:
                strength = "🔥🔥"
            elif abs(score) > 0.3:
                strength = "🔥"
            elif abs(score) > 0.15:
                strength = "🔶"
            else:
                strength = "🔸"
            
            # Add significance if available
            sig_marker = ""
            if concept in concept_stats and 'p_value' in concept_stats[concept]:
                p_val = concept_stats[concept]['p_value']
                if p_val < 0.001:
                    sig_marker = " ***"
                elif p_val < 0.01:
                    sig_marker = " **"
                elif p_val < 0.05:
                    sig_marker = " *"
            
            print(f"  {i}. {concept}: {score:+.3f} {strength}{sig_marker} (→ {direction})")
        
        self.global_explainability_results = {
            'concept_scores': concept_scores,
            'concept_stats': concept_stats,
            'ranking': sorted_concepts
        }
    
    # Enhanced concept testing methods
    def _test_well_funded_concept(self):
        """Enhanced well-funded concept test"""
        well_funded = np.array([m['has_investment_events'] for m in self.metadata])
        
        correlation = np.corrcoef(well_funded.astype(float), self.probabilities)[0, 1]
        
        # Statistical test
        from scipy.stats import chi2_contingency
        
        # Create contingency table: [funded/not_funded] x [survived/failed]
        contingency = np.array([
            [(well_funded & (self.labels == 1)).sum(), (well_funded & (self.labels == 0)).sum()],
            [(~well_funded & (self.labels == 1)).sum(), (~well_funded & (self.labels == 0)).sum()]
        ])
        
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
        except:
            chi2, p_val = np.nan, np.nan
        
        return {
            'score': correlation,
            'chi2': chi2,
            'p_value': p_val,
            'contingency': contingency
        }
    
    def _test_technology_concept(self):
        """Test technology concept"""
        tech_focused = np.array([
            'TECH' in m['industry_category'].upper() or 
            'SOFTWARE' in m['technology_type'].upper() or
            'SAAS' in m['business_model'].upper() or
            m['industry_category'] in ['AI_ML', 'SAAS']
            for m in self.metadata
        ])
        return np.corrcoef(tech_focused.astype(float), self.probabilities)[0, 1]
    
    def _test_large_company_concept(self):
        """Test large company concept"""
        large_company = np.array([
            m['employee_size'] in ['501-1000', '5001-10000']
            for m in self.metadata
        ])
        if large_company.sum() < 10:  # Not enough large companies
            return 0.0
        return np.corrcoef(large_company.astype(float), self.probabilities)[0, 1]
    
    def _test_high_activity_concept(self):
        """Test high activity concept"""
        seq_lengths = np.array([m['sequence_length'] for m in self.metadata])
        high_activity = seq_lengths > np.percentile(seq_lengths, 75)
        return np.corrcoef(high_activity.astype(float), self.probabilities)[0, 1]
    
    def _test_b2b_concept(self):
        """Test B2B concept"""
        b2b_focused = np.array([m['business_model'] == 'B2B' for m in self.metadata])
        return np.corrcoef(b2b_focused.astype(float), self.probabilities)[0, 1]
    
    def _test_us_based_concept(self):
        """Test US-based concept"""
        us_based = np.array([m['country'] == 'USA' for m in self.metadata])
        return np.corrcoef(us_based.astype(float), self.probabilities)[0, 1]
    
    def _test_education_concept(self):
        """Test education-heavy concept"""
        education_heavy = np.array([m['has_education_events'] for m in self.metadata])
        return np.corrcoef(education_heavy.astype(float), self.probabilities)[0, 1]
    
    def _test_international_concept(self):
        """Test international concept"""
        international = np.array([m['country'] not in ['USA', 'Unknown'] for m in self.metadata])
        return np.corrcoef(international.astype(float), self.probabilities)[0, 1]
    
    def _test_mature_company_concept(self):
        """Test mature company concept"""
        ages = np.array([m['company_age'] for m in self.metadata])
        mature = ages > np.percentile(ages, 75)
        return np.corrcoef(mature.astype(float), self.probabilities)[0, 1]
    
    def _test_diverse_events_concept(self):
        """Test diverse events concept"""
        diverse = np.array([m['token_diversity'] > np.median([meta['token_diversity'] for meta in self.metadata]) 
                           for m in self.metadata])
        return np.corrcoef(diverse.astype(float), self.probabilities)[0, 1]
    
    def create_comprehensive_visualizations(self):
        """Create enhanced visualizations"""
        print(f"\n🎨 CREATING ENHANCED VISUALIZATIONS")
        print("="*60)
        
        # Create multiple visualization styles
        self._create_life2vec_style_visualization()
        self._create_performance_dashboard()
        self._create_bias_analysis_plots()
        
        print("✅ All enhanced visualizations created")
    
    def _create_performance_dashboard(self):
        """Create performance analysis dashboard"""
        if not hasattr(self, 'performance_metrics'):
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Metrics comparison
        ax = axes[0, 0]
        metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall']
        values = [self.performance_metrics[m] for m in metrics]
        bars = ax.bar(metrics, values, color=['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral'])
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Confusion Matrix
        ax = axes[0, 1]
        cm = self.performance_metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix', fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Failed', 'Survived'])
        ax.set_yticklabels(['Failed', 'Survived'])
        
        # 3. ROC Curve
        ax = axes[0, 2]
        if not np.isnan(self.performance_metrics['auc_roc']):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(self.labels, self.probabilities)
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {self.performance_metrics["auc_roc"]:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve', fontweight='bold')
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROC Curve', fontweight='bold')
        
        # 4. Precision-Recall Curve
        ax = axes[1, 0]
        if not np.isnan(self.performance_metrics['average_precision']):
            precision, recall, _ = precision_recall_curve(self.labels, self.probabilities)
            ax.plot(recall, precision, color='blue', lw=2,
                   label=f'PR curve (AP = {self.performance_metrics["average_precision"]:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve', fontweight='bold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'PR Curve\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Precision-Recall Curve', fontweight='bold')
        
        # 5. Prediction Distribution
        ax = axes[1, 1]
        ax.hist(self.probabilities[self.labels == 0], bins=30, alpha=0.7, 
               label='Failed Startups', color='red', density=True)
        ax.hist(self.probabilities[self.labels == 1], bins=30, alpha=0.7, 
               label='Successful Startups', color='green', density=True)
        ax.set_xlabel('Predicted Survival Probability')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution by True Label', fontweight='bold')
        ax.legend()
        
        # 6. Class Balance
        ax = axes[1, 2]
        labels = ['Failed', 'Survived']
        sizes = [(self.labels == 0).sum(), (self.labels == 1).sum()]
        colors = ['red', 'green']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Class Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_dashboard.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance dashboard created")
    
    def _create_bias_analysis_plots(self):
        """Create bias analysis visualizations"""
        if not hasattr(self, 'algorithmic_audit_results'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance by Country
        ax = axes[0, 0]
        if 'country' in self.algorithmic_audit_results:
            country_results = self.algorithmic_audit_results['country']
            countries = [r['category'] for r in country_results]
            accuracies = [r['balanced_accuracy'] for r in country_results]
            
            bars = ax.bar(range(len(countries)), accuracies, color='skyblue')
            ax.set_title('Balanced Accuracy by Country', fontweight='bold')
            ax.set_ylabel('Balanced Accuracy')
            ax.set_xticks(range(len(countries)))
            ax.set_xticklabels(countries, rotation=45, ha='right')
            
            # Highlight potential bias
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            if max_acc - min_acc > 0.05:
                ax.axhline(y=min_acc, color='red', linestyle='--', alpha=0.7, label='Min Performance')
                ax.axhline(y=max_acc, color='green', linestyle='--', alpha=0.7, label='Max Performance')
                ax.legend()
        
        # 2. Performance by Industry
        ax = axes[0, 1]
        if 'industry' in self.algorithmic_audit_results:
            industry_results = self.algorithmic_audit_results['industry']
            industries = [r['category'] for r in industry_results]
            accuracies = [r['balanced_accuracy'] for r in industry_results]
            
            bars = ax.bar(range(len(industries)), accuracies, color='lightgreen')
            ax.set_title('Balanced Accuracy by Industry', fontweight='bold')
            ax.set_ylabel('Balanced Accuracy')
            ax.set_xticks(range(len(industries)))
            ax.set_xticklabels(industries, rotation=45, ha='right')
        
        # 3. Sample Size Distribution
        ax = axes[1, 0]
        all_counts = []
        all_categories = []
        for category, results in self.algorithmic_audit_results.items():
            if category != 'bias_analysis':
                for result in results:
                    all_counts.append(result['count'])
                    all_categories.append(f"{category}:{result['category']}")
        
        # Show distribution of sample sizes
        ax.hist(all_counts, bins=20, color='orange', alpha=0.7)
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Subgroup Sample Sizes', fontweight='bold')
        ax.axvline(x=50, color='red', linestyle='--', label='Min Reliable Size')
        ax.legend()
        
        # 4. Bias Summary
        ax = axes[1, 1]
        if 'bias_analysis' in self.algorithmic_audit_results:
            bias_issues = self.algorithmic_audit_results['bias_analysis']['bias_issues']
            
            # Count bias types
            bias_types = Counter()
            for issue in bias_issues:
                if 'accuracy' in issue:
                    bias_types['Accuracy Bias'] += 1
                elif 'survival' in issue:
                    bias_types['Survival Rate Bias'] += 1
                elif 'significant' in issue:
                    bias_types['Statistical Bias'] += 1
                else:
                    bias_types['Other Bias'] += 1
            
            if bias_types:
                types = list(bias_types.keys())
                counts = list(bias_types.values())
                ax.bar(types, counts, color='red', alpha=0.7)
                ax.set_title('Detected Bias Types', fontweight='bold')
                ax.set_ylabel('Count')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'No Significant\nBias Detected', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=14, color='green', fontweight='bold')
                ax.set_title('Bias Detection Results', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bias_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Bias analysis plots created")
    
    def run_complete_analysis(self, target_batches=500, balanced_sampling=False):
        """Run enhanced complete interpretability analysis"""
        print("🚀 ENHANCED STARTUP2VEC INTERPRETABILITY ANALYSIS")
        print("=" * 90)
        print("Complete interpretability analysis with enhanced metrics and visualizations")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data
        if not self.extract_data_with_characteristics(target_batches, balanced_sampling):
            return False
        
        # Run complete analysis pipeline
        print("\\n" + "="*70)
        print("RUNNING ENHANCED INTERPRETABILITY ANALYSIS")
        print("="*70)
        
        self.algorithmic_auditing()
        self.data_contribution_analysis()
        self.visual_exploration()
        self.local_explainability()
        self.global_explainability()
        
        # Create enhanced visualizations
        self.create_comprehensive_visualizations()
        
        # Save results
        self._save_complete_results()
        
        print(f"\n🎉 ENHANCED INTERPRETABILITY ANALYSIS FINISHED!")
        print(f"📊 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to '{self.output_dir}' directory")
        print(f"🎯 Check the following key visualizations:")
        print(f"   📊 startup_life2vec_style.png - Main life2vec-style arch")
        print(f"   📈 performance_dashboard.png - Comprehensive metrics")
        print(f"   ⚖️ bias_analysis.png - Bias detection results")
        
        return True
    
    def _save_complete_results(self):
        """Save all enhanced analysis results"""
        complete_results = {
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'labels': self.labels,
            'embeddings': self.embeddings,
            'sequences': self.sequences,
            'metadata': self.metadata,
            'token_categories': self.token_categories,
            'startup_characteristics': self.startup_characteristics,
            'token_frequencies': self.token_frequencies,
            'performance_metrics': getattr(self, 'performance_metrics', None),
            'algorithmic_audit': getattr(self, 'algorithmic_audit_results', None),
            'data_contribution': getattr(self, 'data_contribution_results', None),
            'visual_exploration': getattr(self, 'visual_exploration_results', None),
            'local_explainability': getattr(self, 'local_explainability_results', None),
            'global_explainability': getattr(self, 'global_explainability_results', None),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methodology': 'Enhanced token-based startup survival interpretability analysis with statistical validation'
        }
        
        results_path = os.path.join(self.output_dir, 'enhanced_interpretability_analysis.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print(f"✅ Enhanced interpretability analysis saved to {results_path}")

def main():
    """Enhanced main function"""
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    analyzer = EnhancedStartupInterpretabilityAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir="startup_interpretability"
    )
    
    print("🔧 ENHANCED STARTUP2VEC INTERPRETABILITY ANALYSIS")
    print("="*70)
    print("🎯 ENHANCED ANALYSIS FEATURES:")
    print("✅ Balanced accuracy and MCC metrics (like finetuning)")
    print("✅ Precision, recall, and F1 scores")
    print("✅ Statistical bias detection with p-values")
    print("✅ Token frequency analysis")
    print("✅ Life2vec-style visualizations")
    print("✅ Enhanced performance dashboard")
    print("✅ Comprehensive bias analysis plots")
    print("✅ Optional balanced sampling")
    print()
    print("📊 ANALYSIS COMPONENTS:")
    print("1. Enhanced Algorithmic Auditing - Statistical performance testing")
    print("2. Enhanced Data Contribution - Statistical significance testing")
    print("3. Enhanced Visual Exploration - Life2vec-style arch")
    print("4. Enhanced Local Explainability - Detailed token analysis")
    print("5. Enhanced Global Explainability - Statistical concept validation")
    print()
    
    # Get user preferences
    print("🎛️ ANALYSIS OPTIONS:")
    print("1. Standard analysis (original data distribution)")
    print("2. Balanced analysis (equal success/failure samples)")
    
    choice = input("Choose analysis type (1 or 2): ").strip()
    balanced_sampling = choice == "2"
    
    if balanced_sampling:
        print("🎯 Using balanced sampling (equal success/failure cases)")
    else:
        print("📊 Using original data distribution")
    
    batch_choice = input("Enter number of batches (500+ recommended): ").strip()
    try:
        target_batches = int(batch_choice)
    except ValueError:
        target_batches = 500
    
    print(f"\\n🚀 Starting enhanced interpretability analysis with {target_batches} batches...")
    
    success = analyzer.run_complete_analysis(
        target_batches=target_batches,
        balanced_sampling=balanced_sampling
    )
    
    if success:
        print("\\n🎉 SUCCESS! Enhanced interpretability analysis completed")
        print("\\n📁 KEY OUTPUT FILES:")
        print("  🎨 startup_life2vec_style.png - Life2vec-style arch visualization")
        print("  📈 performance_dashboard.png - Comprehensive metrics dashboard")
        print("  ⚖️ bias_analysis.png - Statistical bias detection plots")
        print("  📋 enhanced_interpretability_analysis.pkl - All results with statistics")
        print("\\n🎯 KEY ENHANCEMENTS:")
        print("  - Balanced accuracy and MCC (matching finetuning metrics)")
        print("  - Statistical significance testing for bias detection")
        print("  - Enhanced token frequency analysis")
        print("  - Life2vec-style embedding space visualization")
        print("  - Comprehensive performance and bias dashboards")
        return 0
    else:
        print("\\n❌ Enhanced analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
