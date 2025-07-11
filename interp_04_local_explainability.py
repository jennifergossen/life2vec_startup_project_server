# interp_04_local_explainability.py
#!/usr/bin/env python3
"""
STARTUP2VEC LOCAL EXPLAINABILITY - Script 4/5
Local explainability analysis with attention weights and individual explanations
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
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Remove LIME/SHAP imports and related variables
# try:
#     import shap
#     from lime.lime_tabular import LimeTabularExplainer
#     LIME_SHAP_AVAILABLE = True
# except ImportError:
#     LIME_SHAP_AVAILABLE = False
#     print("[WARN] LIME/SHAP not installed. Install with: pip install shap lime")

# Replace model import and loading logic to match interp_02_data_contribution_analysis.py
from models.survival_model import FixedStartupSurvivalModel
from dataloaders.survival_datamodule import SurvivalDataModule

class StartupLocalExplainer:
    """Local explainability analysis for startup survival predictions"""
    
    def __init__(self, checkpoint_path, output_dir="local_explainability_results"):
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
        self.attention_weights = None  # For attention analysis
        
        # Vocabulary
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "explanations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "attention_viz"), exist_ok=True)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_and_data(self):
        """Load model and data with GPU memory management (matching interp_02_data_contribution_analysis.py)"""
        print("🔍 Loading model, data, and parsing vocabulary...")
        try:
            # Model loading
            print("Loading model from:", self.checkpoint_path)
            model = FixedStartupSurvivalModel.load_from_checkpoint(self.checkpoint_path, map_location='cpu')
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            self.model = model
            print(f"✅ Model loaded successfully to {device}")
            # DataModule loading (batch_size=128 for A100, num_workers=4)
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
        """Extract vocabulary for local explanations"""
        try:
            if hasattr(self.datamodule, 'vocabulary'):
                self.vocab_to_idx = self.datamodule.vocabulary.token2index
                self.idx_to_vocab = self.datamodule.vocabulary.index2token
                print(f"✅ Vocabulary extracted: {len(self.vocab_to_idx):,} tokens")
            else:
                print("⚠️ Could not extract vocabulary")
        except Exception as e:
            print(f"⚠️ Vocabulary parsing failed: {e}")
    
    def extract_data_with_attention(self, target_batches=200, balanced_sampling=False):
        """Extract data with attention weights for local explainability"""
        print(f"\n🎯 EXTRACTING DATA WITH ATTENTION WEIGHTS")
        print("="*60)
        
        if balanced_sampling:
            return self._extract_balanced_data_with_attention(target_batches)
        else:
            return self._extract_standard_data_with_attention(target_batches)
    
    def _extract_standard_data_with_attention(self, target_batches):
        """Standard data extraction with attention weights"""
        val_loader = self.datamodule.val_dataloader()
        max_batches = min(target_batches, len(val_loader)) if target_batches > 0 else len(val_loader)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        all_sequences = []
        all_metadata = []
        all_attention_weights = []
        
        # GPU handling
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"�� Using device: {device}")
        
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
        
        print(f"Processing {max_batches:,} batches with attention extraction...")
        successful_batches = 0
        
        # Attention hook function
        def register_attention_hooks():
            attention_weights = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    # Try to extract attention weights if available
                    if hasattr(output, 'attentions') and output.attentions is not None:
                        attention_weights[name] = output.attentions.detach().cpu()
                    elif isinstance(output, tuple) and len(output) > 1:
                        # Sometimes attention is in second element of tuple
                        if hasattr(output[1], 'shape') and len(output[1].shape) >= 3:
                            attention_weights[name] = output[1].detach().cpu()
                return hook
            
            hooks = []
            # Register hooks for transformer layers that might have attention
            for name, module in self.model.named_modules():
                if ('attention' in name.lower() or 'attn' in name.lower() or 
                    'multihead' in name.lower() or 'self_attn' in name.lower()):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
            
            return attention_weights, hooks
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if target_batches > 0 and batch_idx >= max_batches:
                    break
                
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}/{max_batches} (successful: {successful_batches})", end='\r')
                    
                    if device == 'cuda' and batch_idx % 50 == 0:
                        self.clear_cuda_cache()
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    # Register attention hooks
                    attention_weights, hooks = register_attention_hooks()
                    
                    # Forward pass
                    outputs = self.model.forward(input_ids=input_ids, padding_mask=padding_mask)
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
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
                    
                    # Process attention weights
                    batch_attention = self._process_attention_weights(attention_weights, input_ids.size(0))
                    all_attention_weights.extend(batch_attention)
                    
                    # Extract metadata
                    for i in range(input_ids.size(0)):
                        metadata = self._extract_local_metadata(batch, i, input_ids[i, 0, :])
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
        self.attention_weights = all_attention_weights
        
        return True
    
    def _extract_balanced_data_with_attention(self, target_batches):
        """Extract balanced data with attention weights"""
        val_loader = self.datamodule.val_dataloader()
        
        survival_data = {'predictions': [], 'probabilities': [], 'labels': [], 'embeddings': [], 
                        'sequences': [], 'metadata': [], 'attention_weights': []}
        failure_data = {'predictions': [], 'probabilities': [], 'labels': [], 'embeddings': [], 
                       'sequences': [], 'metadata': [], 'attention_weights': []}
        
        target_per_class = target_batches * 4  # Smaller due to batch size 8
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            if torch.cuda.is_available():
                self.clear_cuda_cache()
            self.model = self.model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                device = 'cpu'
                self.model = self.model.to(device)
        
        print(f"Collecting balanced samples with attention (target: {target_per_class} per class)...")
        
        # Attention hook function
        def register_attention_hooks():
            attention_weights = {}
            def hook_fn(name):
                def hook(module, input, output):
                    if hasattr(output, 'attentions') and output.attentions is not None:
                        attention_weights[name] = output.attentions.detach().cpu()
                    elif isinstance(output, tuple) and len(output) > 1:
                        if hasattr(output[1], 'shape') and len(output[1].shape) >= 3:
                            attention_weights[name] = output[1].detach().cpu()
                return hook
            
            hooks = []
            for name, module in self.model.named_modules():
                if ('attention' in name.lower() or 'attn' in name.lower() or 
                    'multihead' in name.lower() or 'self_attn' in name.lower()):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
            return attention_weights, hooks
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if (len(survival_data['labels']) >= target_per_class and 
                    len(failure_data['labels']) >= target_per_class):
                    break
                
                if batch_idx % 20 == 0:
                    survived = len(survival_data['labels'])
                    failed = len(failure_data['labels'])
                    print(f"  Batch {batch_idx}: {survived} survived, {failed} failed", end='\r')
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    survival_labels = batch['survival_label'].to(device)
                    
                    # Register attention hooks
                    attention_weights, hooks = register_attention_hooks()
                    
                    outputs = self.model.forward(input_ids=input_ids, padding_mask=padding_mask)
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
                    survival_logits = outputs['survival_logits']
                    survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                    survival_preds = torch.argmax(survival_logits, dim=1)
                    
                    transformer_output = outputs['transformer_output']
                    company_embeddings = transformer_output[:, 0, :]
                    
                    # Process attention for this batch
                    batch_attention = self._process_attention_weights(attention_weights, input_ids.size(0))
                    
                    for i in range(input_ids.size(0)):
                        true_label = survival_labels[i].squeeze().item()
                        
                        sample_data = {
                            'prediction': survival_preds[i].cpu().numpy(),
                            'probability': survival_probs[i].cpu().numpy(),
                            'label': true_label,
                            'embedding': company_embeddings[i].cpu().numpy(),
                            'sequence': input_ids[i, 0, :].cpu().numpy(),
                            'metadata': self._extract_local_metadata(batch, i, input_ids[i, 0, :]),
                            'attention_weight': batch_attention[i] if i < len(batch_attention) else None
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
        print(f"\n✅ Balanced sampling with attention complete: {min_samples} per class")
        
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
        
        self.attention_weights = (survival_data['attention_weights'][:min_samples] + 
                                 failure_data['attention_weights'][:min_samples])
        
        return True
    
    def _process_attention_weights(self, attention_weights, batch_size):
        """Process attention weights from hooks"""
        processed_attention = []
        
        try:
            if not attention_weights:
                # Create placeholder attention if none available
                for i in range(batch_size):
                    processed_attention.append(None)
                return processed_attention
            
            # Use the first available attention weights
            attention_tensor = list(attention_weights.values())[0]
            
            if attention_tensor is not None and len(attention_tensor.shape) >= 3:
                # Average over attention heads if multiple heads
                if len(attention_tensor.shape) == 4:  # [batch, heads, seq, seq]
                    attention_tensor = attention_tensor.mean(dim=1)  # Average over heads
                
                # Extract attention for each sample in batch
                for i in range(min(batch_size, attention_tensor.size(0))):
                    sample_attention = attention_tensor[i].numpy()
                    processed_attention.append(sample_attention)
                
                # Fill remaining with None if needed
                while len(processed_attention) < batch_size:
                    processed_attention.append(None)
            else:
                # No valid attention found
                for i in range(batch_size):
                    processed_attention.append(None)
                    
        except Exception as e:
            print(f"Warning: Could not process attention weights: {e}")
            for i in range(batch_size):
                processed_attention.append(None)
        
        return processed_attention
    
    def _extract_local_metadata(self, batch, sample_idx, sequence):
        """Extract metadata for local explainability"""
        try:
            base_metadata = {
                'sample_idx': sample_idx,
                'sequence_length': (sequence > 0).sum().item(),
                'company_age': batch['company_age_at_prediction'][sample_idx].item() if 'company_age_at_prediction' in batch else 2,
                'prediction_window': batch['prediction_window'][sample_idx].item() if 'prediction_window' in batch else 1,
            }
            
            # Parse characteristics for explanation
            explanation_characteristics = self._parse_explanation_characteristics(sequence)
            base_metadata.update(explanation_characteristics)
            
            return base_metadata
        except Exception as e:
            return {
                'sample_idx': sample_idx, 'sequence_length': 0, 'company_age': 2, 'prediction_window': 1,
                'country': 'Unknown', 'industry': 'Unknown', 'key_events': [], 'funding_events': []
            }
    
    def _parse_explanation_characteristics(self, sequence):
        """Parse characteristics for generating explanations"""
        characteristics = {
            'country': 'Unknown', 'industry': 'Unknown', 'key_events': [], 'funding_events': []
        }
        
        try:
            clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
            
            for token_id in clean_sequence:
                token_str = self.idx_to_vocab.get(int(token_id), "")
                
                # Country
                if token_str.startswith('COUNTRY_'):
                    characteristics['country'] = token_str.replace('COUNTRY_', '')
                
                # Industry
                elif token_str.startswith('INDUSTRY_'):
                    characteristics['industry'] = token_str.replace('INDUSTRY_', '')
                elif token_str.startswith('CATEGORY_'):
                    characteristics['industry'] = token_str.replace('CATEGORY_', '')
                
                # Key events for explanation
                elif any(keyword in token_str.lower() for keyword in ['funding', 'investment', 'acquisition', 'ipo']):
                    characteristics['key_events'].append(token_str)
                
                # Funding events specifically
                elif token_str.startswith('INV_'):
                    characteristics['funding_events'].append(token_str)
                    
        except Exception as e:
            pass
        
        return characteristics
    
    def run_local_explainability_analysis(self):
        """Run comprehensive local explainability analysis"""
        print("\n" + "="*70)
        print("🔍 LOCAL EXPLAINABILITY ANALYSIS")
        print("="*70)
        
        explainability_results = {}
        
        # 1. Attention Analysis
        print("\n👁️ Attention Weight Analysis:")
        attention_analysis = self._analyze_attention_patterns()
        explainability_results['attention_analysis'] = attention_analysis
        
        # 2. Token Importance Analysis
        print("\n🎯 Token Importance Analysis:")
        token_importance = self._analyze_token_importance()
        explainability_results['token_importance'] = token_importance
        
        # 3. Individual Explanations
        print("\n📖 Individual Startup Explanations:")
        individual_explanations = self._generate_individual_explanations()
        explainability_results['individual_explanations'] = individual_explanations
        
        # 4. Confidence-Based Analysis
        print("\n📊 Confidence-Based Analysis:")
        confidence_analysis = self._analyze_by_confidence_level()
        explainability_results['confidence_analysis'] = confidence_analysis
        
        # 5. Visualization of Explanations
        print("\n🎨 Creating Explanation Visualizations:")
        explanation_viz = self._create_explanation_visualizations()
        explainability_results['explanation_visualizations'] = explanation_viz
        
        # Save results
        self._save_explainability_results(explainability_results)
        
        return explainability_results
    
    def _analyze_attention_patterns(self):
        """Analyze attention weight patterns"""
        attention_results = {}
        
        try:
            if not self.attention_weights or all(att is None for att in self.attention_weights):
                print("    ⚠️ No attention weights available")
                return {'status': 'no_attention_available'}
            
            # Filter out None attention weights
            valid_attention = [att for att in self.attention_weights if att is not None]
            valid_indices = [i for i, att in enumerate(self.attention_weights) if att is not None]
            
            if len(valid_attention) == 0:
                print("    ⚠️ No valid attention weights found")
                return {'status': 'no_valid_attention'}
            
            print(f"    📊 Analyzing attention patterns for {len(valid_attention)} samples")
            
            # Analyze attention patterns
            token_attention_scores = {}
            position_attention_avg = []
            
            for idx, attention_matrix in enumerate(valid_attention[:50]):  # Limit to first 50 for efficiency
                if attention_matrix is not None and len(attention_matrix.shape) >= 2:
                    original_idx = valid_indices[idx]
                    sequence = self.sequences[original_idx]
                    
                    # Get clean sequence
                    clean_sequence = sequence[sequence > 0]
                    
                    # Average attention (use diagonal or first row depending on structure)
                    if attention_matrix.shape[0] == attention_matrix.shape[1]:
                        # Square attention matrix - use diagonal or first row
                        avg_attention = np.mean(attention_matrix, axis=0)
                    else:
                        # Non-square - use first dimension
                        avg_attention = attention_matrix[0] if len(attention_matrix.shape) > 1 else attention_matrix
                    
                    position_attention_avg.append(avg_attention[:len(clean_sequence)])
                    
                    # Map attention to tokens
                    min_len = min(len(avg_attention), len(clean_sequence))
                    for i in range(min_len):
                        token_id = int(clean_sequence[i])
                        token_name = self.idx_to_vocab.get(token_id, f"Token_{token_id}")
                        attention_score = avg_attention[i] if i < len(avg_attention) else 0
                        
                        if token_name not in token_attention_scores:
                            token_attention_scores[token_name] = []
                        token_attention_scores[token_name].append(float(attention_score))
            
            # Calculate average attention scores
            avg_token_attention = {}
            for token, scores in token_attention_scores.items():
                avg_token_attention[token] = np.mean(scores)
            
            # Sort by attention score
            sorted_tokens = sorted(avg_token_attention.items(), key=lambda x: x[1], reverse=True)
            
            print(f"    🏆 Top 10 tokens by attention:")
            for token, score in sorted_tokens[:10]:
                print(f"      {token}: {score:.4f}")
            
            attention_results = {
                'status': 'success',
                'num_valid_samples': len(valid_attention),
                'token_attention_scores': avg_token_attention,
                'top_attended_tokens': sorted_tokens[:20],
                'position_attention_patterns': position_attention_avg
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not analyze attention patterns: {e}")
            attention_results = {'status': 'error', 'error': str(e)}
        
        return attention_results
    
    def _analyze_token_importance(self):
        """Analyze token importance for predictions"""
        token_importance = {}
        
        try:
            print("    📊 Calculating token importance scores...")
            
            # Calculate token importance by survival correlation
            token_survival_scores = {}
            token_failure_scores = {}
            
            for sequence, label in zip(self.sequences, self.labels):
                clean_sequence = sequence[sequence > 0]
                for token_id in clean_sequence:
                    token_id = int(token_id)
                    
                    if token_id not in token_survival_scores:
                        token_survival_scores[token_id] = 0
                        token_failure_scores[token_id] = 0
                    
                    if label == 1:
                        token_survival_scores[token_id] += 1
                    else:
                        token_failure_scores[token_id] += 1
            
            # Calculate importance scores
            token_importance_scores = {}
            overall_survival_rate = self.labels.mean()
            
            for token_id in token_survival_scores.keys():
                total_occurrences = token_survival_scores[token_id] + token_failure_scores[token_id]
                if total_occurrences >= 5:  # Only consider tokens with sufficient occurrences
                    survival_rate = token_survival_scores[token_id] / total_occurrences
                    importance = abs(survival_rate - overall_survival_rate)
                    
                    token_name = self.idx_to_vocab.get(token_id, f"Token_{token_id}")
                    token_importance_scores[token_name] = {
                        'importance_score': importance,
                        'survival_rate': survival_rate,
                        'total_occurrences': total_occurrences,
                        'survival_occurrences': token_survival_scores[token_id],
                        'failure_occurrences': token_failure_scores[token_id]
                    }
            
            # Sort by importance
            sorted_importance = sorted(token_importance_scores.items(), 
                                     key=lambda x: x[1]['importance_score'], 
                                     reverse=True)
            
            print(f"    🏆 Top 10 most important tokens:")
            for token, scores in sorted_importance[:10]:
                print(f"      {token}: {scores['importance_score']:.4f} "
                      f"(SR: {scores['survival_rate']:.3f}, Count: {scores['total_occurrences']})")
            
            token_importance = {
                'token_scores': token_importance_scores,
                'sorted_importance': sorted_importance,
                'overall_survival_rate': overall_survival_rate
            }
            
        except Exception as e:
            print(f"    ⚠️ Could not analyze token importance: {e}")
            token_importance = {'error': str(e)}
        
        return token_importance
    
    def _generate_individual_explanations(self):
        """Generate explanations for individual startup predictions"""
        explanations = {}
        
        try:
            print("    📖 Generating individual explanations...")
            
            # Select diverse examples for explanation
            example_indices = []
            
            # High confidence correct predictions
            high_conf_correct = np.where((np.abs(self.probabilities - 0.5) > 0.3) & 
                                       (self.predictions == self.labels))[0]
            if len(high_conf_correct) > 0:
                example_indices.append(('high_confidence_correct', np.random.choice(high_conf_correct)))
            
            # High confidence incorrect predictions
            high_conf_wrong = np.where((np.abs(self.probabilities - 0.5) > 0.3) & 
                                     (self.predictions != self.labels))[0]
            if len(high_conf_wrong) > 0:
                example_indices.append(('high_confidence_wrong', np.random.choice(high_conf_wrong)))
            
            # Low confidence predictions
            low_conf = np.where(np.abs(self.probabilities - 0.5) < 0.1)[0]
            if len(low_conf) > 0:
                example_indices.append(('low_confidence', np.random.choice(low_conf)))
            
            # High survival probability
            high_surv = np.where(self.probabilities > 0.8)[0]
            if len(high_surv) > 0:
                example_indices.append(('high_survival_prob', np.random.choice(high_surv)))
            
            # Low survival probability
            low_surv = np.where(self.probabilities < 0.2)[0]
            if len(low_surv) > 0:
                example_indices.append(('low_survival_prob', np.random.choice(low_surv)))
            
            # Generate explanations for selected examples
            for category, idx in example_indices:
                explanation = self._generate_single_explanation(idx, category)
                explanations[category] = explanation
                
                print(f"    📋 {category}:")
                print(f"      Prediction: {explanation['prediction']}, Actual: {explanation['true_label']}")
                print(f"      Probability: {explanation['probability']:.3f}")
                print(f"      Industry: {explanation['industry']}, Country: {explanation['country']}")
                print(f"      Key factors: {', '.join(explanation['key_factors'][:3])}")
            
        except Exception as e:
            print(f"    ⚠️ Could not generate individual explanations: {e}")
            explanations = {'error': str(e)}
        
        return explanations
    
    def _generate_single_explanation(self, idx, category):
        """Generate explanation for a single startup"""
        try:
            metadata = self.metadata[idx]
            sequence = self.sequences[idx]
            clean_sequence = sequence[sequence > 0]
            
            explanation = {
                'category': category,
                'index': idx,
                'prediction': self.predictions[idx],
                'probability': self.probabilities[idx],
                'true_label': self.labels[idx],
                'industry': metadata.get('industry', 'Unknown'),
                'country': metadata.get('country', 'Unknown'),
                'sequence_length': len(clean_sequence),
                'key_factors': [],
                'attention_summary': None,
                'explanation_text': ""
            }
            
            # Extract key factors from sequence
            key_factors = []
            funding_indicators = []
            team_indicators = []
            
            for token_id in clean_sequence:
                token_str = self.idx_to_vocab.get(int(token_id), "")
                
                # Identify key factors
                if any(keyword in token_str.lower() for keyword in ['investment', 'funding', 'series']):
                    funding_indicators.append(token_str)
                elif any(keyword in token_str.lower() for keyword in ['employee', 'hire', 'team']):
                    team_indicators.append(token_str)
                elif any(keyword in token_str.lower() for keyword in ['acquisition', 'ipo', 'exit']):
                    key_factors.append(token_str)
            
            explanation['key_factors'] = list(set(funding_indicators + team_indicators + key_factors))[:10]
            
            # Add attention summary if available
            if (self.attention_weights and idx < len(self.attention_weights) and 
                self.attention_weights[idx] is not None):
                explanation['attention_summary'] = "Attention weights available"
            else:
                explanation['attention_summary'] = "No attention weights"
            
            # Generate text explanation
            prob = explanation['probability']
            if prob > 0.7:
                risk_level = "low risk"
                outcome = "likely to survive"
            elif prob > 0.4:
                risk_level = "moderate risk"  
                outcome = "uncertain survival"
            else:
                risk_level = "high risk"
                outcome = "likely to fail"
            
            explanation['explanation_text'] = (
                f"This {explanation['industry']} startup from {explanation['country']} "
                f"has a {prob:.1%} predicted survival probability ({outcome}, {risk_level}). "
                f"The prediction {'matches' if explanation['prediction'] == explanation['true_label'] else 'differs from'} "
                f"the actual outcome. Key factors include: {', '.join(explanation['key_factors'][:3])}."
            )
            
            return explanation
            
        except Exception as e:
            return {
                'category': category, 'index': idx, 'error': str(e),
                'prediction': self.predictions[idx] if idx < len(self.predictions) else None,
                'probability': self.probabilities[idx] if idx < len(self.probabilities) else None,
                'true_label': self.labels[idx] if idx < len(self.labels) else None
            }
    
    def _analyze_by_confidence_level(self):
        """Analyze explainability by confidence level"""
        confidence_analysis = {}
        
        try:
            print("    📊 Analyzing by confidence levels...")
            
            # Calculate confidence (distance from 0.5)
            confidences = np.abs(self.probabilities - 0.5)
            
            # Define confidence levels
            high_conf_mask = confidences > 0.3
            medium_conf_mask = (confidences > 0.1) & (confidences <= 0.3)
            low_conf_mask = confidences <= 0.1
            
            confidence_levels = {
                'high_confidence': high_conf_mask,
                'medium_confidence': medium_conf_mask,
                'low_confidence': low_conf_mask
            }
            
            for level_name, mask in confidence_levels.items():
                if mask.sum() > 0:
                    level_accuracy = (self.predictions[mask] == self.labels[mask]).mean()
                    level_avg_prob = self.probabilities[mask].mean()
                    level_avg_conf = confidences[mask].mean()
                    
                    confidence_analysis[level_name] = {
                        'count': mask.sum(),
                        'accuracy': level_accuracy,
                        'avg_probability': level_avg_prob,
                        'avg_confidence': level_avg_conf
                    }
                    
                    print(f"    📊 {level_name}: {mask.sum()} samples")
                    print(f"      Accuracy: {level_accuracy:.3f}")
                    print(f"      Avg probability: {level_avg_prob:.3f}")
                    print(f"      Avg confidence: {level_avg_conf:.3f}")
            
        except Exception as e:
            print(f"    ⚠️ Could not analyze by confidence: {e}")
            confidence_analysis = {'error': str(e)}
        
        return confidence_analysis
    
    def _create_explanation_visualizations(self):
        """Create visualizations for explanations"""
        viz_results = {}
        
        try:
            print("    🎨 Creating explanation visualizations...")
            
            # 1. Token importance visualization
            if hasattr(self, 'token_importance') or 'token_importance' in dir(self):
                self._create_token_importance_plot()
                viz_results['token_importance_plot'] = os.path.join(self.output_dir, "explanations", "token_importance.png")
            
            # 2. Attention heatmap (if available)
            if (self.attention_weights and any(att is not None for att in self.attention_weights)):
                self._create_attention_heatmaps()
                viz_results['attention_heatmaps'] = os.path.join(self.output_dir, "attention_viz")
            
            # 3. Confidence vs accuracy plot
            self._create_confidence_analysis_plot()
            viz_results['confidence_plot'] = os.path.join(self.output_dir, "explanations", "confidence_analysis.png")
            
            # 4. Prediction explanation examples
            self._create_explanation_examples_plot()
            viz_results['explanation_examples'] = os.path.join(self.output_dir, "explanations", "explanation_examples.png")
            
        except Exception as e:
            print(f"    ⚠️ Could not create explanation visualizations: {e}")
            viz_results = {'error': str(e)}
        
        return viz_results
    
    def _create_token_importance_plot(self):
        """Create token importance visualization"""
        try:
            # Get token importance from analysis
            token_importance_scores = {}
            
            # Calculate token importance (simplified version)
            for sequence, label in zip(self.sequences[:1000], self.labels[:1000]):  # Limit for efficiency
                clean_sequence = sequence[sequence > 0]
                for token_id in clean_sequence:
                    token_name = self.idx_to_vocab.get(int(token_id), f"Token_{token_id}")
                    if token_name not in token_importance_scores:
                        token_importance_scores[token_name] = {'survival': 0, 'failure': 0}
                    
                    if label == 1:
                        token_importance_scores[token_name]['survival'] += 1
                    else:
                        token_importance_scores[token_name]['failure'] += 1
            
            # Calculate importance scores
            importance_data = []
            overall_survival_rate = self.labels.mean()
            
            for token, counts in token_importance_scores.items():
                total = counts['survival'] + counts['failure']
                if total >= 10:  # Minimum occurrences
                    survival_rate = counts['survival'] / total
                    importance = abs(survival_rate - overall_survival_rate)
                    importance_data.append({
                        'token': token[:30],  # Truncate long token names
                        'importance': importance,
                        'survival_rate': survival_rate,
                        'total_count': total
                    })
            
            # Sort and take top 20
            importance_data = sorted(importance_data, key=lambda x: x['importance'], reverse=True)[:20]
            
            if importance_data:
                # Create plot
                plt.figure(figsize=(12, 8))
                
                tokens = [item['token'] for item in importance_data]
                importances = [item['importance'] for item in importance_data]
                survival_rates = [item['survival_rate'] for item in importance_data]
                
                # Create horizontal bar plot
                bars = plt.barh(range(len(tokens)), importances, color='skyblue', alpha=0.7)
                
                # Color bars by survival rate
                for i, (bar, sr) in enumerate(zip(bars, survival_rates)):
                    if sr > overall_survival_rate:
                        bar.set_color('green')
                        bar.set_alpha(0.7)
                    else:
                        bar.set_color('red')
                        bar.set_alpha(0.7)
                
                plt.yticks(range(len(tokens)), tokens)
                plt.xlabel('Importance Score (deviation from overall survival rate)')
                plt.title('Token Importance for Startup Survival Prediction\n(Green: positive for survival, Red: negative for survival)')
                plt.gca().invert_yaxis()  # Top tokens at top
                
                # Add survival rate annotations
                for i, (importance, sr) in enumerate(zip(importances, survival_rates)):
                    plt.text(importance + 0.01, i, f'{sr:.2f}', va='center', fontsize=8)
                
                plt.tight_layout()
                
                # Save plot
                importance_plot_path = os.path.join(self.output_dir, "explanations", "token_importance.png")
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"      ✅ Token importance plot saved to: {importance_plot_path}")
            
        except Exception as e:
            print(f"      ⚠️ Could not create token importance plot: {e}")
    
    def _create_attention_heatmaps(self):
        """Create attention weight heatmaps for selected examples"""
        try:
            # Find examples with valid attention weights
            valid_attention_indices = [i for i, att in enumerate(self.attention_weights) 
                                     if att is not None and hasattr(att, 'shape')]
            
            if len(valid_attention_indices) == 0:
                print("      ⚠️ No valid attention weights for heatmaps")
                return
            
            # Select a few examples for visualization
            n_examples = min(3, len(valid_attention_indices))
            selected_indices = np.random.choice(valid_attention_indices, n_examples, replace=False)
            
            for i, idx in enumerate(selected_indices):
                try:
                    attention_matrix = self.attention_weights[idx]
                    sequence = self.sequences[idx]
                    clean_sequence = sequence[sequence > 0]
                    
                    # Get token names
                    token_names = [self.idx_to_vocab.get(int(token_id), f"T{token_id}") 
                                 for token_id in clean_sequence]
                    
                    # Truncate long token names for display
                    token_names = [name[:15] + '...' if len(name) > 15 else name for name in token_names]
                    
                    # Limit attention matrix size for visualization
                    max_tokens = min(20, len(token_names))
                    attention_viz = attention_matrix[:max_tokens, :max_tokens]
                    token_names_viz = token_names[:max_tokens]
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(attention_viz, 
                              xticklabels=token_names_viz,
                              yticklabels=token_names_viz,
                              cmap='Blues', 
                              annot=False,
                              fmt='.2f',
                              cbar=True)
                    
                    prob = self.probabilities[idx]
                    pred = self.predictions[idx]
                    actual = self.labels[idx]
                    
                    plt.title(f'Attention Heatmap - Example {i+1}\n'
                             f'Prediction: {pred}, Actual: {actual}, Probability: {prob:.3f}')
                    plt.xlabel('Attended Tokens')
                    plt.ylabel('Query Tokens') 
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    
                    # Save heatmap
                    heatmap_path = os.path.join(self.output_dir, "attention_viz", f"attention_heatmap_{i+1}.png")
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"      ✅ Attention heatmap {i+1} saved to: {heatmap_path}")
                    
                except Exception as e:
                    print(f"      ⚠️ Could not create heatmap {i+1}: {e}")
                    continue
            
        except Exception as e:
            print(f"      ⚠️ Could not create attention heatmaps: {e}")
    
    def _create_confidence_analysis_plot(self):
        """Create confidence vs accuracy analysis plot"""
        try:
            # Calculate confidence levels
            confidences = np.abs(self.probabilities - 0.5)
            
            # Create confidence bins
            confidence_bins = np.linspace(0, 0.5, 11)  # 10 bins
            bin_accuracies = []
            bin_centers = []
            bin_counts = []
            
            for i in range(len(confidence_bins) - 1):
                bin_mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
                if bin_mask.sum() > 0:
                    bin_accuracy = (self.predictions[bin_mask] == self.labels[bin_mask]).mean()
                    bin_center = (confidence_bins[i] + confidence_bins[i + 1]) / 2
                    
                    bin_accuracies.append(bin_accuracy)
                    bin_centers.append(bin_center)
                    bin_counts.append(bin_mask.sum())
            
            if bin_accuracies:
                # Create plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Confidence vs Accuracy
                ax1.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8, color='blue')
                ax1.set_xlabel('Prediction Confidence')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Accuracy vs Prediction Confidence')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1)
                
                # Add count annotations
                for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
                    ax1.annotate(f'n={count}', (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8)
                
                # Plot 2: Confidence distribution
                ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel('Prediction Confidence')
                ax2.set_ylabel('Number of Samples')
                ax2.set_title('Distribution of Prediction Confidence')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                confidence_plot_path = os.path.join(self.output_dir, "explanations", "confidence_analysis.png")
                plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"      ✅ Confidence analysis plot saved to: {confidence_plot_path}")
        
        except Exception as e:
            print(f"      ⚠️ Could not create confidence analysis plot: {e}")
    
    def _create_explanation_examples_plot(self):
        """Create visualization of explanation examples"""
        try:
            # Select diverse examples
            examples = []
            
            # High confidence correct
            high_correct = np.where((np.abs(self.probabilities - 0.5) > 0.3) & 
                                  (self.predictions == self.labels))[0]
            if len(high_correct) > 0:
                idx = high_correct[0]
                examples.append({
                    'category': 'High Confidence Correct',
                    'index': idx,
                    'probability': self.probabilities[idx],
                    'prediction': self.predictions[idx],
                    'actual': self.labels[idx],
                    'industry': self.metadata[idx].get('industry', 'Unknown')
                })
            
            # High confidence wrong
            high_wrong = np.where((np.abs(self.probabilities - 0.5) > 0.3) & 
                                (self.predictions != self.labels))[0]
            if len(high_wrong) > 0:
                idx = high_wrong[0]
                examples.append({
                    'category': 'High Confidence Wrong',
                    'index': idx,
                    'probability': self.probabilities[idx],
                    'prediction': self.predictions[idx],
                    'actual': self.labels[idx],
                    'industry': self.metadata[idx].get('industry', 'Unknown')
                })
            
            # Low confidence
            low_conf = np.where(np.abs(self.probabilities - 0.5) < 0.1)[0]
            if len(low_conf) > 0:
                idx = low_conf[0]
                examples.append({
                    'category': 'Low Confidence',
                    'index': idx,
                    'probability': self.probabilities[idx],
                    'prediction': self.predictions[idx],
                    'actual': self.labels[idx],
                    'industry': self.metadata[idx].get('industry', 'Unknown')
                })
            
            if examples:
                # Create plot
                fig, axes = plt.subplots(len(examples), 1, figsize=(12, 4*len(examples)))
                if len(examples) == 1:
                    axes = [axes]
                
                for i, example in enumerate(examples):
                    # Create bar chart for this example
                    categories = ['Prediction', 'Actual', 'Probability']
                    values = [example['prediction'], example['actual'], example['probability']]
                    colors = ['lightblue', 'lightgreen', 'orange']
                    
                    bars = axes[i].bar(categories, values, color=colors, alpha=0.7)
                    
                    # Customize
                    axes[i].set_title(f"{example['category']} - {example['industry']} Startup")
                    axes[i].set_ylim(0, 1)
                    axes[i].set_ylabel('Value')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                examples_plot_path = os.path.join(self.output_dir, "explanations", "explanation_examples.png")
                plt.savefig(examples_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"      ✅ Explanation examples plot saved to: {examples_plot_path}")
        
        except Exception as e:
            print(f"      ⚠️ Could not create explanation examples plot: {e}")
    
    def _save_explainability_results(self, explainability_results):
        """Save local explainability results"""
        # Save as pickle
        results_path = os.path.join(self.output_dir, "local_explainability_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({
                'explainability_results': explainability_results,
                'predictions': self.predictions,
                'probabilities': self.probabilities,
                'labels': self.labels,
                'sequences': self.sequences,
                'metadata': self.metadata,
                'attention_weights': self.attention_weights
            }, f)
        
        # Save as text report
        report_path = os.path.join(self.output_dir, "local_explainability_report.txt")
        with open(report_path, 'w') as f:
            f.write("STARTUP2VEC LOCAL EXPLAINABILITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples analyzed: {len(self.predictions):,}\n")
            f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Attention analysis
            if 'attention_analysis' in explainability_results:
                attention_data = explainability_results['attention_analysis']
                f.write("ATTENTION ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Status: {attention_data.get('status', 'unknown')}\n")
                if 'num_valid_samples' in attention_data:
                    f.write(f"Valid attention samples: {attention_data['num_valid_samples']}\n")
                
                if 'top_attended_tokens' in attention_data:
                    f.write("\nTop Attended Tokens:\n")
                    for token, score in attention_data['top_attended_tokens'][:10]:
                        f.write(f"  {token}: {score:.4f}\n")
                f.write("\n")
            
            # Token importance
            if 'token_importance' in explainability_results:
                token_data = explainability_results['token_importance']
                f.write("TOKEN IMPORTANCE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                if 'sorted_importance' in token_data:
                    f.write("Most Important Tokens:\n")
                    for token, scores in token_data['sorted_importance'][:10]:
                        f.write(f"  {token}: {scores['importance_score']:.4f} "
                               f"(SR: {scores['survival_rate']:.3f})\n")
                f.write("\n")
            
            # Individual explanations
            if 'individual_explanations' in explainability_results:
                explanations_data = explainability_results['individual_explanations']
                f.write("INDIVIDUAL EXPLANATIONS:\n")
                f.write("-" * 30 + "\n")
                for category, explanation in explanations_data.items():
                    if isinstance(explanation, dict) and 'explanation_text' in explanation:
                        f.write(f"{category}:\n")
                        f.write(f"  {explanation['explanation_text']}\n\n")
            
            # Confidence analysis
            if 'confidence_analysis' in explainability_results:
                conf_data = explainability_results['confidence_analysis']
                f.write("CONFIDENCE ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                for level, stats in conf_data.items():
                    if isinstance(stats, dict) and 'count' in stats:
                        f.write(f"{level}: {stats['count']} samples, "
                               f"accuracy: {stats['accuracy']:.3f}\n")
        
        print(f"\n✅ Local explainability results saved to:")
        print(f"  📊 Data: {results_path}")
        print(f"  📋 Report: {report_path}")
        print(f"  🎨 Explanations: {self.output_dir}/explanations/")
        print(f"  👁️ Attention viz: {self.output_dir}/attention_viz/")
    
    def run_complete_explainability(self, target_batches=200, balanced_sampling=False):
        """Run complete local explainability analysis"""
        print("🚀 STARTUP2VEC LOCAL EXPLAINABILITY")
        print("=" * 80)
        print("Local explainability analysis with attention weights and individual explanations")
        print()
        
        # Load model and data
        if not self.load_model_and_data():
            return False
        
        # Extract data with attention weights
        if not self.extract_data_with_attention(target_batches, balanced_sampling):
            return False
        
        # Run local explainability analysis
        explainability_results = self.run_local_explainability_analysis()
        
        print(f"\n🎉 LOCAL EXPLAINABILITY ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.predictions):,} startup samples")
        print(f"📁 Results saved to: {self.output_dir}/")
        
        return explainability_results

# Remove LIME/SHAP is not used for this model type; attention-based explainability only
# def explain_with_lime_shap(model, dataloader, num_samples=10):
#     if not LIME_SHAP_AVAILABLE:
#         print("LIME/SHAP not available.")
#         return
#     print("\n[INFO] Running LIME/SHAP explanations on a few samples...")
#     batch = next(iter(dataloader))
#     input_ids = batch['input_ids']
#     padding_mask = batch['padding_mask']
#     sample_input = input_ids[0].cpu().numpy().flatten()
#     import numpy as np  # Ensure numpy is imported
#     def predict_fn(x):
#         x_tensor = torch.tensor(x, dtype=torch.long).view(1, *input_ids.shape[1:])
#         with torch.no_grad():
#             logits = model(input_ids=x_tensor, padding_mask=padding_mask[0:1])['survival_logits']
#             probs = torch.softmax(logits, dim=1).cpu().numpy()
#         return probs
#     explainer = LimeTabularExplainer(np.array([sample_input]), mode='classification')
#     exp = explainer.explain_instance(sample_input, predict_fn, num_features=10)
#     print("LIME explanation:")
#     print(exp.as_list())
#     explainer_shap = shap.Explainer(predict_fn, [sample_input])
#     shap_values = explainer_shap([sample_input])
#     print("SHAP values:")
#     print(shap_values.values)

def main():
    """Main function for local explainability"""
    CHECKPOINT_PATH = "survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"
    print("🔧 STARTUP2VEC LOCAL EXPLAINABILITY")
    print("="*70)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 CUDA Available: {gpu_count} GPU(s)")
    else:
        print("❌ CUDA not available - will use CPU")
    print()
    print(" LOCAL EXPLAINABILITY FEATURES:")
    print("✅ Attention weight extraction and analysis")
    print("✅ Token importance calculation")
    print("✅ Individual startup explanations")
    print("✅ Confidence-based analysis")
    print("✅ Attention heatmap visualizations")
    print("✅ Explanation examples and plots")
    print("✅ GPU memory management with CPU fallback")
    print("✅ Balanced sampling option")
    print()
    explainer = StartupLocalExplainer(CHECKPOINT_PATH)
    explainer.load_model_and_data()
    # LIME/SHAP is not used for this model type; attention-based explainability only
    explainer.run_complete_explainability(target_batches=200, balanced_sampling=False)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
