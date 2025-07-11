#!/usr/bin/env python3
"""
ENHANCED Startup2Vec Concept Space Visualizer
- Better spacing and reduced overlap
- Enhanced color scheme like Life2vec
- Cluster labels and zoom regions
- Token debugging functionality
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pacmap import PaCMAP
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict, Counter
import warnings
import os
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, TextArea
import re
import colorsys
import argparse
import random
from adjustText import adjust_text
warnings.filterwarnings('ignore')

class EnhancedConceptVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model_state = None
        self.concept_embeddings = None
        self.concept_tokens = None
        self.token_categories = None
        self.checkpoint_paths = None
        
    def debug_available_tokens(self):
        """Debug function to see what tokens are actually available"""
        if not self.concept_tokens:
            print("❌ No tokens loaded yet")
            return
            
        print(f"\n🔍 TOKEN DEBUGGING - Total tokens: {len(self.concept_tokens)}")
        print("=" * 60)
        
        # Find all unique prefixes
        prefixes = defaultdict(list)
        for token in self.concept_tokens:
            token_str = str(token)
            # Look for patterns with underscores
            if '_' in token_str and not token_str.startswith('[') and not token_str.startswith('<'):
                parts = token_str.split('_')
                if len(parts) >= 2:
                    prefix = '_'.join(parts[:2]) + '_'  # First two parts + underscore
                    prefixes[prefix].append(token_str)
        
        # Sort by frequency
        sorted_prefixes = sorted(prefixes.items(), key=lambda x: len(x[1]), reverse=True)
        
        print("🎯 TOP TOKEN PREFIXES (showing first 25):")
        for i, (prefix, tokens) in enumerate(sorted_prefixes[:25]):
            print(f"  {prefix:30s}: {len(tokens):4d} tokens")
            if i < 5:  # Show examples for top 5
                examples = tokens[:3]
                print(f"    Examples: {', '.join(examples)}")
        
        # Look specifically for the missing patterns
        missing_patterns = [
            'DAYS_SINCE_FOUNDING_BINNED_',
            'INV_INVESTOR_TYPES_',
            'days_since_founding_binned_',
            'inv_investor_types_',
            'DAYS_SINCE_FOUNDING',
            'INV_INVESTOR'
        ]
        
        print(f"\n🔍 SEARCHING FOR MISSING PATTERNS:")
        for pattern in missing_patterns:
            matches = [t for t in self.concept_tokens if pattern.lower() in str(t).lower()]
            if matches:
                print(f"  ✅ Found {len(matches)} tokens matching '{pattern}':")
                for match in matches[:5]:  # Show first 5
                    print(f"     {match}")
            else:
                print(f"  ❌ No tokens found for '{pattern}'")
        
        return sorted_prefixes
    
    def load_and_verify_checkpoint(self):
        """Load checkpoint and THOROUGHLY verify embeddings"""
        possible_paths = getattr(self, 'checkpoint_paths', [
            "startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt"
        ])
        
        print("🔍 LOADING & VERIFYING CHECKPOINT (PRETRAINED BALANCED)")
        print("=" * 50)
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📂 Trying: {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    # Try model_state_dict (pretrained)
                    if 'model_state_dict' in checkpoint:
                        self.model_state = checkpoint['model_state_dict']
                        embedding_key = "_orig_mod.transformer.embedding.token.parametrizations.weight.original"
                        print(f"✅ Loaded model_state_dict from checkpoint (pretrained format)")
                    # Try state_dict (finetuned)
                    elif 'state_dict' in checkpoint:
                        self.model_state = checkpoint['state_dict']
                        embedding_key = "transformer.embedding.token.parametrizations.weight.original"
                        print(f"✅ Loaded state_dict from checkpoint (finetuned format)")
                    else:
                        print(f"❌ No model_state_dict or state_dict found in checkpoint")
                        print(f"Available keys: {list(checkpoint.keys())}")
                        continue
                    # Check embedding key
                        if embedding_key in self.model_state:
                            embeddings = self.model_state[embedding_key]
                            print(f"✅ Found embeddings: {embeddings.shape}")
                            # VERIFICATION STEP 2: Deep quality check
                            if self.verify_embedding_quality(embeddings, path):
                                return True
                            else:
                                print(f"❌ Embeddings failed quality check")
                                continue
                        else:
                        print(f"❌ No embedding key found: {embedding_key}")
                        print(f"Available keys: {[k for k in self.model_state.keys() if 'embedding' in k]}")
                            continue
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
        return False
    
    def verify_embedding_quality(self, embeddings, checkpoint_path):
        """THOROUGHLY verify embedding quality"""
        print(f"\n🔬 DEEP EMBEDDING VERIFICATION")
        print("-" * 40)
        
        embeddings_np = embeddings.cpu().numpy()
        
        # Basic stats
        print(f"📊 Shape: {embeddings_np.shape}")
        print(f"📊 Data type: {embeddings_np.dtype}")
        print(f"📊 Mean: {np.mean(embeddings_np):.4f}")
        print(f"📊 Std: {np.std(embeddings_np):.4f}")
        print(f"📊 Min: {np.min(embeddings_np):.4f}")
        print(f"📊 Max: {np.max(embeddings_np):.4f}")
        
        # Check for problems
        nan_count = np.isnan(embeddings_np).sum()
        inf_count = np.isinf(embeddings_np).sum()
        zero_count = np.count_nonzero(embeddings_np == 0)
        
        print(f"🚨 NaN values: {nan_count} ({100*nan_count/embeddings_np.size:.3f}%)")
        print(f"🚨 Inf values: {inf_count} ({100*inf_count/embeddings_np.size:.3f}%)")
        print(f"🚨 Zero values: {zero_count} ({100*zero_count/embeddings_np.size:.3f}%)")
        
        # Check if embeddings are actually trained (not random)
        dim_stds = np.std(embeddings_np, axis=0)
        avg_dim_std = np.mean(dim_stds)
        min_dim_std = np.min(dim_stds)
        max_dim_std = np.max(dim_stds)
        
        print(f"📈 Dimension variance - avg: {avg_dim_std:.4f}, min: {min_dim_std:.4f}, max: {max_dim_std:.4f}")
        
        # Check for suspicious patterns
        if avg_dim_std < 0.001:
            print(f"⚠️ Very low variance - embeddings might not be trained")
            return False
        
        if nan_count > embeddings_np.size * 0.01:  # > 1% NaN
            print(f"⚠️ Too many NaN values")
            return False
        
        if avg_dim_std > 10:
            print(f"⚠️ Extremely high variance - might be corrupted")
            return False
        
        print(f"✅ Embeddings pass quality verification!")
        return True
    
    def load_vocabulary(self):
        """Load vocabulary from trained model (has processed tokens)"""
        # Try multiple possible vocab paths
        possible_vocab_paths = [
            "data/processed/vocab/startup_vocab/result.tsv",
            "data/processed/vocab/result.tsv", 
            "vocab/result.tsv",
            "checkpoints/vocab.txt",  # Sometimes saved with checkpoint
        ]
        
        for vocab_path in possible_vocab_paths:
            if os.path.exists(vocab_path):
                print(f"📂 Found vocabulary at: {vocab_path}")
                try:
                    if vocab_path.endswith('.tsv'):
                        vocab_df = pd.read_csv(vocab_path, sep="\t", index_col=0)
                        if 'TOKEN' in vocab_df.columns:
                            self.concept_tokens = vocab_df['TOKEN'].tolist()
                            print(f"✅ Loaded {len(self.concept_tokens)} tokens from TSV")
                            return True
                    elif vocab_path.endswith('.txt'):
                        with open(vocab_path, 'r') as f:
                            self.concept_tokens = [line.strip() for line in f.readlines()]
                            print(f"✅ Loaded {len(self.concept_tokens)} tokens from TXT")
                            return True
                except Exception as e:
                    print(f"❌ Error loading {vocab_path}: {e}")
                    continue
        
        print("❌ No vocabulary file found - using raw data tokens")
        return False
    
    def extract_verified_embeddings(self):
        """Extract embeddings with final verification"""
        # Use the correct embedding key depending on which model_state was loaded
        if '_orig_mod.transformer.embedding.token.parametrizations.weight.original' in self.model_state:
            embedding_key = '_orig_mod.transformer.embedding.token.parametrizations.weight.original'
        else:
            embedding_key = 'transformer.embedding.token.parametrizations.weight.original'
        embeddings = self.model_state[embedding_key].cpu().numpy()
        
        # Handle any remaining NaN values
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            print(f"🔧 Fixing {nan_count} NaN values...")
            nan_mask = np.isnan(embeddings)
            embeddings[nan_mask] = np.random.normal(0, 0.01, size=nan_count)
        
        # Center embeddings (important for visualization)
        global_mean = np.mean(embeddings, axis=0)
        self.concept_embeddings = embeddings - global_mean
        
        print(f"✅ Final embeddings: {self.concept_embeddings.shape}")
        print(f"📊 Centered - new mean: {np.mean(self.concept_embeddings):.6f}")
        
        return True
    
    def match_vocab_and_embeddings(self):
        """Match vocabulary size to embedding size"""
        if not self.concept_tokens or self.concept_embeddings is None:
            return False
            
        vocab_size = len(self.concept_tokens)
        embed_size = self.concept_embeddings.shape[0]
        
        if vocab_size != embed_size:
            min_size = min(vocab_size, embed_size)
            self.concept_tokens = self.concept_tokens[:min_size]
            self.concept_embeddings = self.concept_embeddings[:min_size]
        
        return True
    
    def filter_for_enhanced_space(self):
        """Filter tokens with ENHANCED pattern matching for PROCESSED tokens"""
        print("\n🎯 CREATING ENHANCED CONCEPT SPACE FROM PROCESSED TOKENS")
        print("=" * 50)
        
        # ENHANCED patterns - EXCLUDING Education, Acquisition Targets, Event Types, Technology Types, Countries, Business Models
        include_patterns = {
            # Business fundamentals  
            'CATEGORY': 'Industries',
            'INDUSTRY': 'Industries',
            
            # People & Jobs
            'PEOPLE_JOB': 'Job Types',
            'PPL_JOB': 'Job Types',
            'JOB_TYPE': 'Job Types',
            'JOB_TITLE': 'Job Titles',
            
            # Investment (keeping this)
            'INV_INVESTMENT': 'Investment Types',
            'INVESTMENT': 'Investment Types',
            
            'INV_INVESTOR_TYPES': 'Investor Types',  # Your processed investor types
            'INVESTOR_TYPE': 'Investor Types',
            
            # Company age - your processed binned data
            'DAYS_SINCE_FOUNDING_BINNED': 'Company Age',
            'DAYS_': 'Company Age',  # The prefix you use
        }
        excluded_categories = ['Education', 'Acquisition Targets', 'Event Types', 'Technology Types', 'Countries', 'Business Models']
        print(f"❗ Excluding categories from visualization: {excluded_categories}")
        
        def should_include_token(token_str):
            # Special tokens
            if token_str.startswith('[') or token_str.startswith('<'):
                return True, 'Special Tokens'
            
            # Check all patterns with flexible matching
            for pattern, category in include_patterns.items():
                if pattern.upper() in token_str.upper():  # Case insensitive
                    return True, category
            
            return False, None
        
        # Filter all tokens
        filtered_indices = []
        filtered_tokens = []
        filtered_categories = {}
        
        category_counts = Counter()
        
        print(f"🔍 Filtering {len(self.concept_tokens)} PROCESSED tokens...")
        matched_patterns = set()
        
        for i, token in enumerate(self.concept_tokens):
            include, category = should_include_token(str(token))
            
            if include and category:
                filtered_indices.append(i)
                filtered_tokens.append(token)
                filtered_categories[token] = category
                category_counts[category] += 1
                
                # Track which pattern matched
                for pattern in include_patterns.keys():
                    if pattern.upper() in str(token).upper():
                        matched_patterns.add(pattern)
        
        print(f"✅ Matched patterns from processed data: {sorted(matched_patterns)}")
        
        if len(filtered_indices) < 10:
            print(f"❌ Too few tokens ({len(filtered_indices)}) for enhanced space")
            return False
        
        # Balance the data - sample down dominant categories
        max_tokens_per_category = 200  # Limit to prevent one category dominating
        
        balanced_indices = []
        balanced_tokens = []
        balanced_categories = {}
        
        for category, count in category_counts.items():
            category_indices = [i for i, token in enumerate(filtered_tokens) 
                              if filtered_categories[token] == category]
            
            if count > max_tokens_per_category:
                # Sample down large categories
                random.seed(42)  # Reproducible
                sampled_indices = random.sample(category_indices, max_tokens_per_category)
                print(f"⚖️ Sampled {category}: {count} → {max_tokens_per_category} tokens")
            else:
                sampled_indices = category_indices
            
            for idx in sampled_indices:
                balanced_indices.append(filtered_indices[idx])
                balanced_tokens.append(filtered_tokens[idx])
                balanced_categories[filtered_tokens[idx]] = category
        
        # Update data structures with balanced data
        self.concept_embeddings = self.concept_embeddings[balanced_indices]
        self.concept_tokens = balanced_tokens
        self.token_categories = balanced_categories
        
        # Recount after balancing
        final_counts = Counter(balanced_categories.values())
        
        print(f"✅ Enhanced balanced space: {len(balanced_tokens)} tokens across {len(final_counts)} categories")
        print(f"📊 Balanced distribution:")
        for category, count in final_counts.most_common():
            print(f"   {category:20s}: {count:3d} tokens")
        
        return True
    
    def create_enhanced_visualization(self):
        """Create enhanced visualization like Life2vec paper, with Life2vec-style zoomed insets"""
        if self.concept_embeddings is None or not self.token_categories:
            return False
        
        print(f"\n🎨 CREATING ENHANCED VISUALIZATION")
        print("=" * 55)
        print(f"📊 Visualizing {len(self.concept_tokens)} tokens")
        
        # ENHANCED PaCMAP parameters for better separation
        print("🔄 Computing enhanced PaCMAP projection...")
        pacmap_reducer = PaCMAP(
            n_neighbors=40,      # Increased for better global structure
            MN_ratio=0.5,        # More neighbors for cluster separation
            FP_ratio=2.0,        # More repulsion for cluster separation
            random_state=42,
            apply_pca=True,
            n_components=2
        )
        
        embedding_2d = pacmap_reducer.fit_transform(self.concept_embeddings)
        embedding_2d = embedding_2d * 2.0  # Spread out the points more
        print("✅ Enhanced PaCMAP completed!")
        
        # DISTINCT COLOR SCHEME FOR ALL CATEGORIES (maximally distinct)
        distinct_colors = [
            "#e6194b",  # red
            "#3cb44b",  # green
            "#ffe119",  # yellow
            "#4363d8",  # blue
            "#f58231",  # orange
            "#911eb4",  # purple
            "#46f0f0",  # cyan
            "#f032e6",  # magenta
            "#bcf60c",  # lime
            "#fabebe",  # pink
        ]
        unique_categories = [cat for cat in set(self.token_categories.values()) if cat not in ['Countries', 'Business Models']]
        category_colors = {cat: distinct_colors[i % len(distinct_colors)] for i, cat in enumerate(unique_categories)}
        print(f"🎨 Assigned maximally distinct colors to categories:")
        for category, color in category_colors.items():
            print(f"   {category}: {color}")
        
        # Create large, detailed plot like Life2vec
        fig, ax = plt.subplots(figsize=(30, 16))
        
        # Calculate cluster centers for labels
        cluster_centers = {}
        
        # Plot each category with Life2vec styling
        for category in unique_categories:
            indices = [i for i, token in enumerate(self.concept_tokens) 
                      if self.token_categories[token] == category]
            
            if indices:
                x_coords = embedding_2d[indices, 0]
                y_coords = embedding_2d[indices, 1]
                
                # Calculate center for labeling
                cluster_centers[category] = (np.mean(x_coords), np.mean(y_coords))
                
                color = category_colors[category]
                
                # Enhanced styling - larger size, more alpha
                size = 80  # Larger size
                alpha = 0.85
                edgewidth = 0.7
                
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=color, 
                                   label=f'{category} ({len(indices)})',
                                   alpha=alpha, s=size, 
                                   marker='o',
                                   edgecolors='white', 
                                   linewidth=edgewidth, 
                                   zorder=5)
        
        # Add cluster labels like Life2vec (custom offset for 'Company Age' to avoid overlap)
        for category, (x, y) in cluster_centers.items():
            if category != 'Special Tokens':
                if category == 'Company Age':
                    offset = (40, 0)  # 40 points to the right
                else:
                    offset = (0, 20)  # 20 points above
                ax.annotate(category, 
                           xy=(x, y), 
                           xytext=offset,
                           textcoords='offset points',
                           fontsize=13, 
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   alpha=0.8,
                                   edgecolor='gray'),
                           zorder=10)
        # For each category, randomly select 3 tokens and annotate them in grey font, using adjustText to avoid overlap
        random.seed(42)
        texts = []
        for category in unique_categories:
            indices = [i for i, token in enumerate(self.concept_tokens) if self.token_categories[token] == category]
            if len(indices) == 0:
                continue
            sample_indices = random.sample(indices, min(3, len(indices)))
            for idx in sample_indices:
                token = self.concept_tokens[idx]
                x, y = embedding_2d[idx]
                texts.append(ax.text(x, y, str(token), fontsize=14, ha='left', va='bottom', color='#888888', fontweight='normal', zorder=30,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5), max_iter=100)
        # Remove axes and the frame from the plot
        ax.set_axis_off()
        # After plotting, expand the axis limits by 15% margin
        all_x = embedding_2d[:, 0]
        all_y = embedding_2d[:, 1]
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_margin = 0.15 * (x_max - x_min)
        y_margin = 0.15 * (y_max - y_min)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        # Build custom legend without numbers in brackets, one line
        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                                 markerfacecolor=category_colors[cat], markersize=10, linewidth=0)
                          for cat in unique_categories]
        ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=len(unique_categories), fontsize=18, frameon=False, fancybox=True, shadow=False,
                  title='Concept Categories', title_fontsize=20)
        # Save high-quality versions
        save_path = 'concept_space.png'
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Saved enhanced visualization: {save_path}")
        
        pdf_path = 'concept_space.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        print(f"✅ PDF version: {pdf_path}")
        
        return True
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis"""
        print("🚀 ENHANCED STARTUP2VEC ANALYZER")
        print("=" * 60)
        print("Creating Life2vec-style concept space visualization")
        print("=" * 60)
        
        steps = [
            ("Loading & verifying checkpoint", self.load_and_verify_checkpoint),
            ("Loading vocabulary", self.load_vocabulary),
            ("Debugging available tokens", self.debug_available_tokens),
            ("Extracting verified embeddings", self.extract_verified_embeddings),
            ("Matching vocab and embeddings", self.match_vocab_and_embeddings),
            ("Filtering for enhanced space", self.filter_for_enhanced_space),
            ("Creating enhanced visualization", self.create_enhanced_visualization),
        ]
        
        for step_name, step_func in steps:
            print(f"\n⏳ {step_name}...")
            result = step_func()
            if step_name == "Debugging available tokens":
                continue  # This step always returns something, not boolean
            if not result:
                print(f"❌ Failed at: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ ENHANCED ANALYSIS COMPLETE!")
        print("📊 Generated files:")
        print("   • concept_space.png")
        print("   • concept_space.pdf")
        print("🎨 Life2vec-style visualization with better spacing and colors!")
        print("=" * 60)
        return True

def main():
    parser = argparse.ArgumentParser(description="Startup2Vec Concept Space Visualizer")
    parser.add_argument('--finetuned', action='store_true', help='Use finetuned checkpoint instead of pretrained')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualizer = EnhancedConceptVisualizer(device=device)

    if args.finetuned:
        print("\n🚩 Using FINETUNED checkpoint!")
        visualizer.checkpoint_paths = ["survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"]
    else:
        print("\n🚩 Using PRETRAINED checkpoint!")
        visualizer.checkpoint_paths = ["startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt"]

    visualizer.run_enhanced_analysis()

if __name__ == "__main__":
    main()
