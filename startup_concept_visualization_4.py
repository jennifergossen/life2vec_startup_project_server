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
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import warnings
import os
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, TextArea
import re
warnings.filterwarnings('ignore')

class EnhancedConceptVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model_state = None
        self.concept_embeddings = None
        self.concept_tokens = None
        self.token_categories = None
        
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
        
        possible_paths = [
            "checkpoints/startup2vec-full-1gpu-512d-epoch=14-step=091380.ckpt",
            "checkpoints/startup2vec-full-1gpu-512d-epoch=13-step=085288.ckpt", 
            "checkpoints/startup2vec-full-1gpu-512d-epoch=12-step=079196.ckpt",
            "checkpoints/last.ckpt",
        ]
        
        print("🔍 LOADING & VERIFYING CHECKPOINT")
        print("=" * 50)
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📂 Trying: {path}")
                
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    
                    if 'state_dict' in checkpoint:
                        self.model_state = checkpoint['state_dict']
                        print(f"✅ Loaded Lightning checkpoint")
                        
                        # VERIFICATION STEP 1: Check embedding key exists
                        embedding_key = "transformer.embedding.token.parametrizations.weight.original"
                        
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
                            print(f"❌ No embedding key found")
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
        embedding_key = "transformer.embedding.token.parametrizations.weight.original"
        
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
        
        # ENHANCED patterns - matching your ACTUAL processed token structure
        include_patterns = {
            # Geographic
            'COUNTRY': 'Countries',
            'EVT_COUNTRY': 'Countries',
            
            # Business fundamentals  
            'CATEGORY': 'Industries',
            'INDUSTRY': 'Industries',
            
            # Events
            'EVENT_TYPE': 'Event Types',
            'EVT_TYPE': 'Event Types',
            'EVT_CAT': 'Event Types',
            
            # Education 
            'EDU_DEGREE': 'Education',
            'EDU_INSTITUTION': 'Education',
            'EDUCATION': 'Education',
            
            # People & Jobs
            'PEOPLE_JOB': 'Job Types',
            'PPL_JOB': 'Job Types',
            'JOB_TYPE': 'Job Types',
            'JOB_TITLE': 'Job Titles',
            
            # Investment & Acquisition
            'ACQ_ACQUISITION': 'Acquisition Types',
            'ACQUISITION': 'Acquisition Types',
            'ACQ_TARGET': 'Acquisition Targets',
            
            'INV_INVESTMENT': 'Investment Types',
            'INVESTMENT': 'Investment Types',
            
            'INV_INVESTOR_TYPES': 'Investor Types',  # Your processed investor types
            'INVESTOR_TYPE': 'Investor Types',
            
            # Company age - your processed binned data
            'DAYS_SINCE_FOUNDING_BINNED': 'Company Age',
            'DAYS_': 'Company Age',  # The prefix you use
            
            # Additional business concepts from your data
            'MODEL_': 'Business Models',  # From your description categorization
            'TECH_': 'Technology Types',  # From your description categorization
            'INDUSTRY_': 'Industry Categories',  # From your description categorization
        }
        
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
                import random
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
        """Create enhanced visualization like Life2vec paper"""
        if self.concept_embeddings is None or not self.token_categories:
            return False
        
        print(f"\n🎨 CREATING ENHANCED VISUALIZATION")
        print("=" * 55)
        print(f"📊 Visualizing {len(self.concept_tokens)} tokens")
        
        # ENHANCED PaCMAP parameters for better separation
        print("🔄 Computing enhanced PaCMAP projection...")
        pacmap_reducer = PaCMAP(
            n_neighbors=20,      # Increased for better global structure
            MN_ratio=0.3,        # Reduced for tighter clusters
            FP_ratio=1.5,        # Reduced for less repulsion
            random_state=42,
            apply_pca=True,
            n_components=2
        )
        
        embedding_2d = pacmap_reducer.fit_transform(self.concept_embeddings)
        print("✅ Enhanced PaCMAP completed!")
        
        # LIFE2VEC-STYLE COLOR SCHEME
        unique_categories = list(set(self.token_categories.values()))
        n_categories = len(unique_categories)
        
        # Enhanced color palette inspired by Life2vec
        life2vec_colors = {
            'Special Tokens': '#FF6B6B',      # Red
            'Countries': '#4ECDC4',           # Teal  
            'Industries': '#45B7D1',          # Blue
            'Event Types': '#96CEB4',         # Green
            'Education': '#FECA57',           # Yellow
            'Job Types': '#FF9FF3',           # Pink
            'Job Titles': '#54A0FF',          # Light blue
            'Acquisition Types': '#5F27CD',    # Purple
            'Investment Types': '#00D2D3',     # Cyan
            'Investor Types': '#FF9F43',       # Orange
            'Company Age': '#1DD1A1',          # Mint
        }
        
        # Fallback colors if we have more categories
        fallback_colors = plt.cm.Set3(np.linspace(0, 1, max(0, n_categories - len(life2vec_colors))))
        
        category_colors = {}
        for i, category in enumerate(unique_categories):
            if category in life2vec_colors:
                category_colors[category] = life2vec_colors[category]
            else:
                idx = i - len(life2vec_colors)
                if idx >= 0 and idx < len(fallback_colors):
                    category_colors[category] = fallback_colors[idx]
                else:
                    category_colors[category] = '#888888'  # Gray fallback
        
        # Create large, detailed plot like Life2vec
        fig, ax = plt.subplots(figsize=(16, 12))
        
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
                
                # Enhanced styling - uniform size like Life2vec
                size = 60  # Consistent size
                alpha = 0.8
                edgewidth = 0.5
                
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=color, 
                                   label=f'{category} ({len(indices)})',
                                   alpha=alpha, s=size, 
                                   marker='o',
                                   edgecolors='white', 
                                   linewidth=edgewidth, 
                                   zorder=5)
        
        # Add cluster labels like Life2vec
        for category, (x, y) in cluster_centers.items():
            if category != 'Special Tokens':  # Skip special tokens label
                ax.annotate(category, 
                           xy=(x, y), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=10, 
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   alpha=0.8,
                                   edgecolor='gray'),
                           zorder=10)
        
        # Life2vec-style formatting
        ax.set_xlabel('PaCMAP Dimension 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('PaCMAP Dimension 2', fontsize=14, fontweight='bold')
        ax.set_title('Startup2Vec: Enhanced Concept Space\\n' + 
                    f'Business Concept Clustering ({len(self.concept_tokens)} concepts)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Clean styling like Life2vec
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Enhanced legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                 frameon=True, fancybox=True, shadow=True, 
                 title='Concept Categories', title_fontsize=12)
        
        plt.tight_layout()
        
        # Save high-quality versions
        save_path = 'startup2vec_ENHANCED_CLUSTERS.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Saved enhanced visualization: {save_path}")
        
        pdf_path = 'startup2vec_ENHANCED_CLUSTERS.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        print(f"✅ PDF version: {pdf_path}")
        
        plt.show()
        
        # Create zoom regions like Life2vec (optional)
        self.create_zoom_regions(embedding_2d, category_colors)
        
        return True
    
    def create_zoom_regions(self, embedding_2d, category_colors):
        """Create ACTUAL zoom-in regions like Life2vec paper"""
        print("\n🔍 Creating focused zoom regions...")
        
        # Calculate cluster centers and identify interesting regions
        cluster_centers = {}
        cluster_data = {}
        
        for category in set(self.token_categories.values()):
            indices = [i for i, token in enumerate(self.concept_tokens) 
                      if self.token_categories[token] == category]
            
            if indices:
                x_coords = embedding_2d[indices, 0]
                y_coords = embedding_2d[indices, 1]
                cluster_centers[category] = (np.mean(x_coords), np.mean(y_coords))
                cluster_data[category] = {
                    'x': x_coords, 'y': y_coords, 'indices': indices,
                    'tokens': [self.concept_tokens[i] for i in indices]
                }
        
        # Define 4 interesting zoom regions based on actual clusters
        zoom_regions = []
        
        # Get the 4 largest clusters for zooming
        sorted_clusters = sorted(cluster_data.items(), key=lambda x: len(x[1]['indices']), reverse=True)
        
        for category, data in sorted_clusters[:4]:
            x_center, y_center = cluster_centers[category]
            # Calculate zoom bounds around cluster
            x_std = np.std(data['x'])
            y_std = np.std(data['y'])
            
            zoom_regions.append({
                'name': f'{category} Cluster',
                'xlim': (x_center - 2*x_std, x_center + 2*x_std),
                'ylim': (y_center - 2*y_std, y_center + 2*y_std),
                'category': category,
                'sample_tokens': data['tokens'][:5]  # Show 5 example tokens
            })
        
        # Create the zoom visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, region in enumerate(zoom_regions):
            ax = axes[i]
            
            # Plot all points with reduced size and transparency
            for category in set(self.token_categories.values()):
                indices = [j for j, token in enumerate(self.concept_tokens) 
                          if self.token_categories[token] == category]
                
                if indices:
                    x_coords = embedding_2d[indices, 0]
                    y_coords = embedding_2d[indices, 1]
                    color = category_colors[category]
                    
                    # Highlight focus category, dim others
                    if category == region['category']:
                        ax.scatter(x_coords, y_coords, 
                                 c=color, alpha=0.8, s=40, 
                                 marker='o', edgecolors='white', linewidth=1,
                                 label=f'{category} ({len(indices)})')
                    else:
                        ax.scatter(x_coords, y_coords, 
                                 c=color, alpha=0.3, s=15, 
                                 marker='o', edgecolors='none')
            
            # Apply zoom
            ax.set_xlim(region['xlim'])
            ax.set_ylim(region['ylim'])
            
            # Add title and sample tokens
            ax.set_title(f"Region {i+1}: {region['name']}", fontsize=14, fontweight='bold')
            
            # Add sample tokens as text
            sample_text = "Sample tokens:\\n" + "\\n".join(region['sample_tokens'])
            ax.text(0.02, 0.98, sample_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        plt.suptitle('Startup2Vec Concept Space: Detailed Cluster Views', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('startup2vec_ZOOM_REGIONS.png', dpi=300, bbox_inches='tight')
        print("✅ Created focused zoom regions with actual cluster details")
        plt.show()
    
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
        print("   • startup2vec_ENHANCED_CLUSTERS.png")
        print("   • startup2vec_ENHANCED_CLUSTERS.pdf")
        print("   • startup2vec_ZOOM_REGIONS.png")
        print("🎨 Life2vec-style visualization with better spacing and colors!")
        print("=" * 60)
        return True

def main():
    visualizer = EnhancedConceptVisualizer(device='cpu')
    visualizer.run_enhanced_analysis()

if __name__ == "__main__":
    main()
