#!/usr/bin/env python3
"""
STARTUP2VEC INTERPRETABILITY ANALYSIS
Complete interpretability analysis using actual token structure
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

class StartupInterpretabilityAnalyzer:
    """Interpretability analyzer using actual token structure"""
    
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
        
        # Token categories (parsed from actual vocabulary)
        self.token_categories = None
        self.startup_characteristics = None
        
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
            
            print(f"\\n📋 Token Categories Found:")
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
            # Company characteristics (from startup.py)
            'company_country': {},          # COUNTRY_[code]
            'company_category': {},         # CATEGORY_[industry]
            'company_employee_size': {},    # EMPLOYEE_[range]
            'company_industry': {},         # INDUSTRY_[type]
            'company_business_model': {},   # MODEL_[type]
            'company_technology': {},       # TECH_[type]
            
            # Event types and categories
            'event_types': {},              # EVT_TYPE_[type]
            'event_categories': {},         # EVT_CAT_[category]
            'event_terms': {},              # EVT_TERM_[term]
            'event_roles': {},              # EVENT_ROLES_[role]
            
            # People/roles
            'people_jobs': {},              # PPL_JOB_[role]
            'people_terms': {},             # PPL_TERM_[term]
            'people_job_titles': {},        # PEOPLE_JOB_TITLE_[title]
            
            # Education events (multiple fields)
            'education_degree_type': {},    # EDU_DEGREE_TYPE_[type]
            'education_institution': {},    # EDU_INSTITUTION_[institution]
            'education_subject': {},        # EDU_SUBJECT_[subject]
            
            # Investment events (comprehensive)
            'investment_types': {},         # INV_INVESTMENT_TYPE_[type]
            'investment_investor_types': {},# INV_INVESTOR_TYPES_[type]
            'investment_amounts': {},       # INV_RAISED_AMOUNT_USD_[bin]
            'investment_fund_sizes': {},    # INV_FUND_SIZE_USD_[bin]
            'investment_counts': {},        # INV_INVESTOR_COUNT_[bin]
            'investment_valuations': {},    # INV_POST_MONEY_VALUATION_USD_[bin]
            'investment_other': {},         # Other INV_ tokens
            
            # Acquisition events
            'acquisition_types': {},        # ACQ_ACQUISITION_TYPE_[type]
            'acquisition_prices': {},       # ACQ_PRICE_USD_[bin]
            'acquisition_other': {},        # Other ACQ_ tokens
            
            # IPO events (comprehensive)
            'ipo_exchanges': {},            # IPO_EXCHANGE_[exchange]
            'ipo_money_raised': {},         # IPO_MONEY_RAISED_USD_[bin]
            'ipo_share_prices': {},         # IPO_SHARE_PRICE_USD_[bin]
            'ipo_valuations': {},           # IPO_VALUATION_USD_[bin]
            'ipo_other': {},                # Other IPO_ tokens
            
            # Temporal
            'days_since_founding': {},      # DAYS_[bin]
        }
        
        for token_str, token_id in self.vocab_to_idx.items():
            token_upper = token_str.upper()
            
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
            
            # Event types
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
            
            # Education (multiple specific fields)
            elif token_str.startswith('EDU_DEGREE_TYPE_'):
                categories['education_degree_type'][token_id] = token_str
            elif token_str.startswith('EDU_INSTITUTION_'):
                categories['education_institution'][token_id] = token_str
            elif token_str.startswith('EDU_SUBJECT_'):
                categories['education_subject'][token_id] = token_str
            
            # Investment (comprehensive)
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
            
            # IPO (comprehensive)
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
    
    def extract_data_with_characteristics(self, target_batches=500):
        """Extract data and parse startup characteristics from sequences"""
        print(f"\\n🎯 EXTRACTING DATA WITH TOKEN ANALYSIS")
        print("="*60)
        
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
                    print(f"  Batch {batch_idx}/{max_batches}", end='\\r')
                
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
                    print(f"\\nError in batch {batch_idx}: {e}")
                    continue
        
        print(f"\\n✅ Data extraction complete: {len(all_predictions):,} samples")
        
        # Store results
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.labels = np.array(all_labels)
        self.embeddings = np.array(all_embeddings)
        self.sequences = all_sequences
        self.metadata = all_metadata
        
        # Parse startup characteristics from tokens
        self._parse_startup_characteristics()
        
        # Performance analysis
        self._detailed_performance_analysis()
        
        return True
    
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
        print("\\n📋 Parsing startup characteristics from tokens...")
        
        # Count characteristics from tokens
        countries = Counter([m['country'] for m in self.metadata])
        industries = Counter([m['industry_category'] for m in self.metadata])
        employee_sizes = Counter([m['employee_size'] for m in self.metadata])
        business_models = Counter([m['business_model'] for m in self.metadata])
        tech_types = Counter([m['technology_type'] for m in self.metadata])
        
        print(f"\\n📊 Startup Characteristics Found:")
        print(f"  Countries: {dict(countries.most_common(10))}")
        print(f"  Industries: {dict(industries.most_common(10))}")
        print(f"  Employee Sizes: {dict(employee_sizes)}")
        print(f"  Business Models: {dict(business_models)}")
        print(f"  Technology Types: {dict(tech_types.most_common(5))}")
        
        # Event presence statistics
        investment_events = sum(1 for m in self.metadata if m['has_investment_events'])
        acquisition_events = sum(1 for m in self.metadata if m['has_acquisition_events'])
        ipo_events = sum(1 for m in self.metadata if m['has_ipo_events'])
        education_events = sum(1 for m in self.metadata if m['has_education_events'])
        
        print(f"\\n📈 Event Presence:")
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
            'event_stats': {
                'investment_events': investment_events,
                'acquisition_events': acquisition_events,
                'ipo_events': ipo_events,
                'education_events': education_events
            }
        }
    
    def _detailed_performance_analysis(self):
        """Detailed performance analysis"""
        print(f"\\n📊 DETAILED PERFORMANCE ANALYSIS")
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
        
        print(f"📈 Overall Performance:")
        print(f"  Total samples: {len(self.predictions):,}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Survival rate: {survival_rate:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        print(f"\\nConfusion Matrix:")
        print(cm)
        
        if not np.isnan(auc):
            if auc > 0.65:
                print(f"✅ Excellent performance (AUC > 0.65)")
            elif auc > 0.6:
                print(f"✅ Good performance (AUC > 0.6)")
            elif auc > 0.55:
                print(f"📊 Moderate performance (AUC > 0.55)")
            else:
                print(f"🤔 Lower performance - likely due to extreme class imbalance")
    
    def algorithmic_auditing(self):
        """1. Algorithmic Auditing using token categories"""
        print(f"\\n🔍 1. ALGORITHMIC AUDITING")
        print("="*60)
        print("Examining model performance across startup subgroups...")
        
        results = {}
        
        # 1. Employee Size (using EMPLOYEE_ tokens)
        print("\\n👥 Performance by Employee Size:")
        employee_sizes = [m['employee_size'] for m in self.metadata]
        employee_results = self._analyze_subgroup_performance(employee_sizes, 'Employee Size')
        results['employee_size'] = employee_results
        
        # 2. Industry (using CATEGORY_/INDUSTRY_ tokens)
        print("\\n🏭 Performance by Industry:")
        industries = [m['industry_category'] for m in self.metadata]
        industry_results = self._analyze_subgroup_performance(industries, 'Industry')
        results['industry'] = industry_results
        
        # 3. Country (using COUNTRY_ tokens)
        print("\\n🌍 Performance by Country:")
        countries = [m['country'] for m in self.metadata]
        # Only analyze countries with sufficient samples
        country_counts = Counter(countries)
        major_countries = [country for country, count in country_counts.items() if count >= 50]
        major_country_categories = [country if country in major_countries else 'Other' for country in countries]
        country_results = self._analyze_subgroup_performance(major_country_categories, 'Country')
        results['country'] = country_results
        
        # 4. Business Model (using MODEL_ tokens)
        print("\\n💼 Performance by Business Model:")
        business_models = [m['business_model'] for m in self.metadata]
        model_results = self._analyze_subgroup_performance(business_models, 'Business Model')
        results['business_model'] = model_results
        
        # 5. Technology Type (using TECH_ tokens)
        print("\\n💻 Performance by Technology Type:")
        tech_types = [m['technology_type'] for m in self.metadata]
        tech_results = self._analyze_subgroup_performance(tech_types, 'Technology Type')
        results['technology_type'] = tech_results
        
        # 6. Event Presence Analysis
        print("\\n🎯 Performance by Investment Event Presence:")
        investment_presence = ['Has Investment Events' if m['has_investment_events'] else 'No Investment Events' for m in self.metadata]
        investment_results = self._analyze_subgroup_performance(investment_presence, 'Investment Events')
        results['investment_events'] = investment_results
        
        # Enhanced bias detection
        self._enhanced_bias_detection(results)
        
        self.algorithmic_audit_results = results
    
    def data_contribution_analysis(self):
        """2. Data Contribution Analysis using token categories and individual tokens"""
        print(f"\\n📊 2. DATA CONTRIBUTION ANALYSIS")
        print("="*60)
        print("Analyzing contribution of event categories and individual tokens...")
        
        contribution_results = {}
        
        # Analyze contribution by event category
        print("\\n🔍 Event Category Contribution Analysis:")
        
        category_order = [
            ('company_country', 'Company Country'),
            ('company_category', 'Company Category'),
            ('company_employee_size', 'Company Employee Size'),
            ('investment_types', 'Investment Types'),
            ('investment_amounts', 'Investment Amounts'),
            ('acquisition_types', 'Acquisition Types'),
            ('ipo_exchanges', 'IPO Events'),
            ('education_degree_type', 'Education - Degree Types'),
            ('education_subject', 'Education - Subjects'),
            ('people_job_titles', 'People Job Titles'),
            ('event_types', 'Event Types'),
            ('event_categories', 'Event Categories')
        ]
        
        for category_key, category_name in category_order:
            if category_key not in self.token_categories or len(self.token_categories[category_key]) == 0:
                continue
                
            print(f"\\n📋 Analyzing {category_name}:")
            
            token_dict = self.token_categories[category_key]
            
            # Calculate presence of this event category in each startup
            category_presence = []
            for sequence in self.sequences:
                clean_sequence = sequence[sequence > 0]
                has_category = any(int(token) in token_dict for token in clean_sequence)
                category_presence.append(has_category)
            
            category_presence = np.array(category_presence)
            
            # Calculate survival rates
            survived_with_category = self.labels[category_presence].mean() if category_presence.sum() > 0 else 0
            survived_without_category = self.labels[~category_presence].mean() if (~category_presence).sum() > 0 else 0
            
            contribution_score = survived_with_category - survived_without_category
            
            print(f"  Companies with {category_name}: {category_presence.sum():,} ({category_presence.mean()*100:.1f}%)")
            print(f"  Survival rate WITH {category_name}: {survived_with_category:.2%}")
            print(f"  Survival rate WITHOUT {category_name}: {survived_without_category:.2%}")
            print(f"  Contribution score: {contribution_score:+.3f}")
            
            contribution_results[category_key] = {
                'category_name': category_name,
                'presence_count': int(category_presence.sum()),
                'presence_rate': float(category_presence.mean()),
                'survival_with': float(survived_with_category),
                'survival_without': float(survived_without_category),
                'contribution_score': float(contribution_score),
                'token_count': len(token_dict)
            }
        
        # Individual token analysis for top categories
        print(f"\\n🔤 Individual Token Analysis:")
        self._analyze_individual_tokens()
        
        # Rank by contribution
        ranked_contributions = sorted(contribution_results.items(), 
                                    key=lambda x: abs(x[1]['contribution_score']), 
                                    reverse=True)
        
        print(f"\\n🏆 Event Category Importance Ranking:")
        for i, (category_key, scores) in enumerate(ranked_contributions, 1):
            direction = "survival ✅" if scores['contribution_score'] > 0 else "failure ❌"
            print(f"  {i}. {scores['category_name']}: {scores['contribution_score']:+.3f} (→ {direction})")
        
        self.data_contribution_results = {
            'category_contributions': contribution_results,
            'ranking': ranked_contributions
        }
    
    def _analyze_individual_tokens(self):
        """Analyze individual token contributions within key categories"""
        # Focus on key categories for individual token analysis
        key_categories = [
            ('investment_types', 'Investment Types'),
            ('company_country', 'Company Countries'), 
            ('people_job_titles', 'People Job Titles'),
            ('education_subject', 'Education Subjects'),
            ('event_categories', 'Event Categories')
        ]
        
        for category_key, category_name in key_categories:
            if category_key not in self.token_categories:
                continue
                
            print(f"\\n�� Individual tokens in {category_name}:")
            token_dict = self.token_categories[category_key]
            
            token_contributions = {}
            
            for token_id, token_str in token_dict.items():
                # Count occurrences in survived vs died companies
                survived_count = 0
                died_count = 0
                total_survived = 0
                total_died = 0
                
                for i, sequence in enumerate(self.sequences):
                    clean_sequence = sequence[sequence > 0]
                    has_token = int(token_id) in clean_sequence
                    
                    if self.labels[i] == 1:  # Survived
                        total_survived += 1
                        if has_token:
                            survived_count += 1
                    else:  # Died
                        total_died += 1
                        if has_token:
                            died_count += 1
                
                if total_survived > 0 and total_died > 0:
                    survived_rate = survived_count / total_survived
                    died_rate = died_count / total_died
                    contribution = survived_rate - died_rate
                    
                    if abs(contribution) > 0.005:  # Only meaningful differences
                        token_contributions[token_str] = contribution
            
            # Show top tokens for this category
            if token_contributions:
                sorted_tokens = sorted(token_contributions.items(), key=lambda x: x[1], reverse=True)
                
                print(f"    Top survival predictors:")
                for token, score in sorted_tokens[:5]:
                    print(f"      {token}: {score:+.4f}")
                
                print(f"    Top failure predictors:")
                for token, score in sorted_tokens[-3:]:
                    print(f"      {token}: {score:+.4f}")
    
    def visual_exploration(self):
        """3. Visual Exploration with arch visualization"""
        print(f"\\n🎨 3. VISUAL EXPLORATION - Startup Arch")
        print("="*60)
        print("Creating arch visualization of startup embedding space...")
        
        # Try to install and use UMAP
        try:
            import umap.umap_ as umap
            umap_available = True
        except ImportError:
            print("⚠️ Installing UMAP for arch visualization...")
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
        
        print("🔄 Creating t-SNE projection...")
        try:
            tsne = TSNE(n_components=2, perplexity=min(50, len(viz_embeddings)//4), random_state=42)
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
            'viz_metadata': viz_metadata,
            'sample_size': len(viz_embeddings)
        }
        
        # Create comprehensive arch visualization
        self._create_startup_arch_visualization()
    
    def local_explainability(self):
        """4. Local Explainability with token analysis"""
        print(f"\\n🔍 4. LOCAL EXPLAINABILITY")
        print("="*60)
        print("Analyzing individual startup trajectories using tokens...")
        
        # Select diverse examples
        examples = self._select_diverse_examples()
        
        local_results = {}
        
        for example_type, indices in examples.items():
            if len(indices) == 0:
                continue
                
            print(f"\\n🎯 Analyzing {example_type.replace('_', ' ').title()} ({len(indices)} examples):")
            
            example_analyses = []
            for i, idx in enumerate(indices[:5]):  # Top 5 examples
                analysis = self._analyze_individual_startup(idx)
                example_analyses.append(analysis)
                
                # Show summary using characteristics
                meta = self.metadata[idx]
                print(f"  Example {i+1} (Sample {idx}):")
                print(f"    Survival Probability: {self.probabilities[idx]:.3f}")
                print(f"    True Outcome: {'Survived' if self.labels[idx] == 1 else 'Failed'}")
                print(f"    Country: {meta['country']}")
                print(f"    Industry: {meta['industry_category']}")
                print(f"    Employee Size: {meta['employee_size']}")
                print(f"    Business Model: {meta['business_model']}")
                print(f"    Has Investment Events: {meta['has_investment_events']}")
                print(f"    Key Events: {analysis['key_events']}")
            
            local_results[example_type] = example_analyses
        
        self.local_explainability_results = local_results
    
    def _select_diverse_examples(self):
        """Select diverse startup examples for analysis"""
        examples = {
            'high_confidence_successes': [],
            'high_confidence_failures': [],
            'surprising_successes': [],
            'surprising_failures': [],
            'uncertain_predictions': [],
            'well_funded_startups': [],
            'large_companies': []
        }
        
        for i in range(len(self.probabilities)):
            prob = self.probabilities[i]
            true = self.labels[i]
            meta = self.metadata[i]
            
            # Basic confidence categories
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
            
            # Token-based categories
            if meta['has_investment_events']:
                examples['well_funded_startups'].append(i)
            if meta['employee_size'] == '500+':  # Only 500+ is large
                examples['large_companies'].append(i)
        
        # Limit examples
        for key in examples:
            examples[key] = examples[key][:10]
        
        return examples
    
    def _analyze_individual_startup(self, idx):
        """Analyze individual startup using tokens"""
        meta = self.metadata[idx]
        sequence = self.sequences[idx]
        clean_sequence = sequence[sequence > 0]
        
        # Identify key events in sequence
        key_events = []
        event_categories = []
        
        for token in clean_sequence:
            token_str = self.idx_to_vocab.get(int(token), f"Token_{token}")
            
            # Categorize using token structure
            if token_str.startswith('INV_'):
                key_events.append(token_str)
                event_categories.append('Investment')
            elif token_str.startswith('ACQ_'):
                key_events.append(token_str)
                event_categories.append('Acquisition')
            elif token_str.startswith('IPO_'):
                key_events.append(token_str)
                event_categories.append('IPO')
            elif token_str.startswith('PPL_JOB_') or token_str.startswith('PEOPLE_JOB_TITLE_'):
                key_events.append(token_str)
                event_categories.append('People')
            elif token_str.startswith('EDU_'):
                key_events.append(token_str)
                event_categories.append('Education')
            elif token_str.startswith('EVT_'):
                key_events.append(token_str)
                event_categories.append('Event')
        
        # Get most important events
        event_counter = Counter(key_events)
        top_events = [event for event, count in event_counter.most_common(5)]
        
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
            'key_events': top_events,
            'event_categories': list(set(event_categories))
        }
    
    def global_explainability(self):
        """5. Global Explainability with operationalized concepts"""
        print(f"\\n🌐 5. GLOBAL EXPLAINABILITY")
        print("="*60)
        print("Testing operationalized concepts using token presence...")
        
        # Define operationalized concepts using tokens
        concepts = {
            'Well Funded': self._test_well_funded_concept,
            'Technology Focus': self._test_technology_concept,
            'Large Company': self._test_large_company_concept,  # Only 500+ employees
            'High Activity': self._test_high_activity_concept,
            'B2B Business Model': self._test_b2b_concept,
            'US-Based': self._test_us_based_concept,
            'Education-Heavy': self._test_education_concept
        }
        
        concept_scores = {}
        
        for concept_name, concept_test in concepts.items():
            print(f"\\n🧠 Testing concept: {concept_name}")
            
            try:
                concept_score = concept_test()
                concept_scores[concept_name] = concept_score
                
                # Interpret results
                direction = "survival" if concept_score > 0 else "failure"
                if abs(concept_score) > 0.3:
                    strength = "strong"
                elif abs(concept_score) > 0.15:
                    strength = "moderate"
                elif abs(concept_score) > 0.05:
                    strength = "weak"
                else:
                    strength = "negligible"
                
                print(f"  Score: {concept_score:+.3f} ({strength} association with {direction})")
                
            except Exception as e:
                print(f"  ⚠️ Concept test failed: {e}")
                concept_scores[concept_name] = 0.0
        
        # Rank concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\\n🏆 Global Concept Importance Ranking:")
        for i, (concept, score) in enumerate(sorted_concepts, 1):
            direction = "survival ✅" if score > 0 else "failure ❌"
            strength = "🔥" if abs(score) > 0.3 else "🔶" if abs(score) > 0.15 else "🔸"
            print(f"  {i}. {concept}: {score:+.3f} {strength} (→ {direction})")
        
        self.global_explainability_results = {
            'concept_scores': concept_scores,
            'ranking': sorted_concepts
        }
    
    # Operationalized concept testing methods using tokens
    def _test_well_funded_concept(self):
        """Test well-funded concept using investment tokens"""
        well_funded = np.array([m['has_investment_events'] for m in self.metadata])
        return np.corrcoef(well_funded.astype(float), self.probabilities)[0, 1]
    
    def _test_technology_concept(self):
        """Test technology concept using industry/tech tokens"""
        tech_focused = np.array([
            'TECH' in m['industry_category'].upper() or 
            'SOFTWARE' in m['technology_type'].upper() or
            'SAAS' in m['business_model'].upper()
            for m in self.metadata
        ])
        return np.corrcoef(tech_focused.astype(float), self.probabilities)[0, 1]
    
    def _test_large_company_concept(self):
        """Test large company concept using employee size tokens (only 500+)"""
        large_company = np.array([
            m['employee_size'] == '500+'  # Only 500+ is considered large
            for m in self.metadata
        ])
        return np.corrcoef(large_company.astype(float), self.probabilities)[0, 1]
    
    def _test_high_activity_concept(self):
        """Test high activity using sequence length"""
        seq_lengths = np.array([m['sequence_length'] for m in self.metadata])
        high_activity = seq_lengths > np.percentile(seq_lengths, 75)
        return np.corrcoef(high_activity.astype(float), self.probabilities)[0, 1]
    
    def _test_b2b_concept(self):
        """Test B2B concept using business model tokens"""
        b2b_focused = np.array([m['business_model'] == 'B2B' for m in self.metadata])
        return np.corrcoef(b2b_focused.astype(float), self.probabilities)[0, 1]
    
    def _test_us_based_concept(self):
        """Test US-based concept using country tokens"""
        us_based = np.array([m['country'] == 'USA' for m in self.metadata])
        return np.corrcoef(us_based.astype(float), self.probabilities)[0, 1]
    
    def _test_education_concept(self):
        """Test education-heavy concept using education tokens"""
        education_heavy = np.array([m['has_education_events'] for m in self.metadata])
        return np.corrcoef(education_heavy.astype(float), self.probabilities)[0, 1]
    
    def create_comprehensive_visualizations(self):
        """Create all visualizations"""
        print(f"\\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        self._create_startup_arch_visualization()
        
        print("✅ All visualizations created")
    
    def _create_startup_arch_visualization(self):
        """Create startup arch visualization"""
        if not hasattr(self, 'visual_exploration_results'):
            return
        
        viz_data = self.visual_exploration_results
        
        fig = plt.figure(figsize=(20, 12))
        
        # Main UMAP arch
        if viz_data['umap_embeddings'] is not None:
            plt.subplot(2, 3, 1)
            scatter = plt.scatter(viz_data['umap_embeddings'][:, 0], 
                                 viz_data['umap_embeddings'][:, 1],
                                 c=viz_data['viz_probabilities'], 
                                 cmap='RdYlBu_r', alpha=0.7, s=8)
            plt.colorbar(scatter, label='Survival Probability')
            plt.title('Startup Arch (UMAP)\\nColored by Survival Probability', fontweight='bold')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        
        # Other views
        plot_idx = 2
        
        # PCA view
        if viz_data['pca_embeddings'] is not None:
            plt.subplot(2, 3, plot_idx)
            scatter = plt.scatter(viz_data['pca_embeddings'][:, 0], 
                                 viz_data['pca_embeddings'][:, 1],
                                 c=viz_data['viz_probabilities'], 
                                 cmap='RdYlBu_r', alpha=0.6, s=4)
            plt.title('PCA View')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 1
        
        # Country coloring
        if viz_data['umap_embeddings'] is not None:
            plt.subplot(2, 3, plot_idx)
            countries = [m['country'] for m in viz_data['viz_metadata']]
            country_codes = pd.Categorical(countries).codes
            scatter = plt.scatter(viz_data['umap_embeddings'][:, 0], 
                                 viz_data['umap_embeddings'][:, 1],
                                 c=country_codes, cmap='tab10', alpha=0.6, s=4)
            plt.title('Colored by Country')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 1
        
        # Employee size coloring
        if viz_data['umap_embeddings'] is not None:
            plt.subplot(2, 3, plot_idx)
            employee_sizes = [m['employee_size'] for m in viz_data['viz_metadata']]
            size_codes = pd.Categorical(employee_sizes).codes
            scatter = plt.scatter(viz_data['umap_embeddings'][:, 0], 
                                 viz_data['umap_embeddings'][:, 1],
                                 c=size_codes, cmap='viridis', alpha=0.6, s=4)
            plt.title('Colored by Employee Size')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 1
        
        # Business model coloring
        if viz_data['umap_embeddings'] is not None:
            plt.subplot(2, 3, plot_idx)
            business_models = [m['business_model'] for m in viz_data['viz_metadata']]
            model_codes = pd.Categorical(business_models).codes
            scatter = plt.scatter(viz_data['umap_embeddings'][:, 0], 
                                 viz_data['umap_embeddings'][:, 1],
                                 c=model_codes, cmap='plasma', alpha=0.6, s=4)
            plt.title('Colored by Business Model')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 1
        
        # True labels
        if viz_data['umap_embeddings'] is not None:
            plt.subplot(2, 3, plot_idx)
            scatter = plt.scatter(viz_data['umap_embeddings'][:, 0], 
                                 viz_data['umap_embeddings'][:, 1],
                                 c=viz_data['viz_labels'], 
                                 cmap='RdYlBu_r', alpha=0.6, s=4)
            plt.title('True Survival Outcomes')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        
        plt.suptitle('Startup2Vec: Arch Visualization', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'startup_arch.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Startup Arch visualization created")
    
    def _analyze_subgroup_performance(self, categories, category_name):
        """Analyze performance across subgroups"""
        unique_cats = list(set(categories))
        results = []
        
        for cat in unique_cats:
            mask = np.array([c == cat for c in categories])
            if mask.sum() >= 20:  # Minimum sample size
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
                
                if acc_range > 0.03:  # 3% threshold
                    bias_issues.append(f"{category}: {acc_range:.1%} accuracy gap")
                
                if survival_range > 0.05:  # 5% threshold
                    bias_issues.append(f"{category}: {survival_range:.1%} survival rate gap")
        
        if bias_issues:
            print("  ⚠️ Potential bias detected:")
            for issue in bias_issues:
                print(f"    - {issue}")
        else:
            print("  ✅ No significant bias detected")
    
    def run_complete_analysis(self, target_batches=500):
        """Run complete interpretability analysis"""
        print("🚀 STARTUP2VEC INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        print("Complete interpretability analysis using actual token structure")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data with token analysis
        if not self.extract_data_with_characteristics(target_batches):
            return False
        
        # Run complete analysis pipeline
        print("\\n" + "="*60)
        print("RUNNING COMPLETE INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        self.algorithmic_auditing()
        self.data_contribution_analysis()
        self.visual_exploration()
        self.local_explainability()
        self.global_explainability()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Save results
        self._save_complete_results()
        
        print(f"\\n🎉 COMPLETE INTERPRETABILITY ANALYSIS FINISHED!")
        print(f"📊 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to '{self.output_dir}' directory")
        print(f"🎯 Check startup_arch.png for the main visualization")
        
        return True
    
    def _save_complete_results(self):
        """Save all analysis results"""
        complete_results = {
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'labels': self.labels,
            'embeddings': self.embeddings,
            'sequences': self.sequences,
            'metadata': self.metadata,
            'token_categories': self.token_categories,
            'startup_characteristics': self.startup_characteristics,
            'algorithmic_audit': getattr(self, 'algorithmic_audit_results', None),
            'data_contribution': getattr(self, 'data_contribution_results', None),
            'visual_exploration': getattr(self, 'visual_exploration_results', None),
            'local_explainability': getattr(self, 'local_explainability_results', None),
            'global_explainability': getattr(self, 'global_explainability_results', None),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methodology': 'Token-based startup survival interpretability analysis'
        }
        
        results_path = os.path.join(self.output_dir, 'complete_interpretability_analysis.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print(f"✅ Complete interpretability analysis saved to {results_path}")

def main():
    """Main function"""
    checkpoint_path = "./survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    analyzer = StartupInterpretabilityAnalyzer(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        output_dir="startup_interpretability"
    )
    
    print("🔧 STARTUP2VEC INTERPRETABILITY ANALYSIS")
    print("="*60)
    print("🎯 COMPREHENSIVE ANALYSIS FEATURES:")
    print("✅ Company characteristics from COUNTRY_, EMPLOYEE_, CATEGORY_ tokens")
    print("✅ Event analysis from INV_, ACQ_, IPO_, EDU_, EVT_, PPL_ tokens")
    print("✅ Business models from MODEL_ tokens")
    print("✅ Technology types from TECH_ tokens")
    print("✅ Individual token contribution analysis")
    print("✅ Event category and individual token importance")
    print("✅ Operationalized concept testing")
    print("✅ Comprehensive bias analysis")
    print()
    print("📊 ANALYSIS COMPONENTS:")
    print("1. Algorithmic Auditing - Performance across subgroups")
    print("2. Data Contribution - Event types and individual tokens")
    print("3. Visual Exploration - Startup arch visualization")
    print("4. Local Explainability - Individual startup analysis")
    print("5. Global Explainability - Concept-level insights")
    print()
    
    choice = input("Enter number of batches (500+ recommended): ").strip()
    try:
        target_batches = int(choice)
    except ValueError:
        target_batches = 500
    
    print(f"\\n🚀 Starting interpretability analysis with {target_batches} batches...")
    
    success = analyzer.run_complete_analysis(target_batches=target_batches)
    
    if success:
        print("\\n🎉 SUCCESS! Interpretability analysis completed")
        print("\\n📁 KEY OUTPUT FILES:")
        print("  📊 startup_arch.png - Main arch visualization")
        print("  📋 complete_interpretability_analysis.pkl - All results")
        print("\\n🎯 KEY INSIGHTS FROM ANALYSIS:")
        print("  - Performance analysis across employee sizes, countries, industries")
        print("  - Event category importance (investment, acquisition, IPO, education)")
        print("  - Individual token contributions within categories")
        print("  - Startup arch visualization colored by characteristics")
        print("  - Individual startup explanations")
        print("  - Global concept analysis (well-funded, tech-focus, etc.)")
        return 0
    else:
        print("\\n❌ Analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
