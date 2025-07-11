#!/usr/bin/env python3
"""
FOCUSED Startup2Vec Concept Space Visualizer
- Creates a highly focused concept space with 10 key concept types
- Should produce much clearer clusters and better interpretability
- Thoroughly verifies embeddings quality
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
warnings.filterwarnings('ignore')

class FocusedConceptVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model_state = None
        self.concept_embeddings = None
        self.concept_tokens = None
        self.token_categories = None
        
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
        """Load vocabulary from TSV file"""
        vocab_path = "data/processed/vocab/startup_vocab/result.tsv"
        
        if not os.path.exists(vocab_path):
            return False
        
        try:
            vocab_df = pd.read_csv(vocab_path, sep="\t", index_col=0)
            if 'TOKEN' in vocab_df.columns:
                self.concept_tokens = vocab_df['TOKEN'].tolist()
                return True
        except:
            pass
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
    
    def filter_for_focused_space(self):
        """Filter tokens for FOCUSED concept space with 10 key concept types"""
        print("\n🎯 CREATING FOCUSED CONCEPT SPACE")
        print("=" * 50)
        print("Target concept types:")
        print("  • COUNTRY")
        print("  • CATEGORY") 
        print("  • EVENT_TYPE")
        print("  • EDU_degree_type")
        print("  • PEOPLE_job_type")
        print("  • ACQ_acquisition_type")
        print("  • INV_investment_type")
        print("  • INV_investor_types")
        print("  • PEOPLE_job_title")
        print("  • DAYS_SINCE_FOUNDING_BINNED")
        print("=" * 50)
        
        # FOCUSED patterns - exactly the 10 requested concept types
        include_patterns = {
            # Geographic
            'COUNTRY_': 'Countries',
            
            # Business fundamentals  
            'CATEGORY_': 'Industries',
            
            # Events
            'EVENT_TYPE_': 'Event Types',
            'EVT_TYPE_': 'Event Types',  # Alternative pattern
            
            # Education
            'EDU_DEGREE_TYPE_': 'Education',
            'EDU_degree_type_': 'Education',  # Alternative pattern
            
            # People & Jobs
            'PEOPLE_JOB_TYPE_': 'Job Types',
            'PPL_JOB_TYPE_': 'Job Types',  # Alternative pattern
            'PEOPLE_job_type_': 'Job Types',  # Alternative pattern
            
            'PEOPLE_JOB_TITLE_': 'Job Titles', 
            'PPL_JOB_TITLE_': 'Job Titles',  # Alternative pattern
            'PEOPLE_job_title_': 'Job Titles',  # Alternative pattern
            
            # Investment & Acquisition
            'ACQ_ACQUISITION_TYPE_': 'Acquisition Types',
            'ACQ_acquisition_type_': 'Acquisition Types',  # Alternative pattern
            
            'INV_INVESTMENT_TYPE_': 'Investment Types',
            'INV_investment_type_': 'Investment Types',  # Alternative pattern
            
            'INV_INVESTOR_TYPES_': 'Investor Types',
            'INV_investor_types_': 'Investor Types',  # Alternative pattern
            
            # Company age
            'DAYS_SINCE_FOUNDING_BINNED_': 'Company Age',
        }
        
        def should_include_token(token_str):
            # Special tokens
            if token_str.startswith('[') or token_str.startswith('<'):
                return True, 'Special Tokens'
            
            # Check all patterns
            for pattern, category in include_patterns.items():
                if token_str.startswith(pattern):
                    return True, category
            
            return False, None
        
        # Filter all tokens
        filtered_indices = []
        filtered_tokens = []
        filtered_categories = {}
        
        category_counts = Counter()
        
        for i, token in enumerate(self.concept_tokens):
            include, category = should_include_token(str(token))
            
            if include and category:
                filtered_indices.append(i)
                filtered_tokens.append(token)
                filtered_categories[token] = category
                category_counts[category] += 1
        
        if len(filtered_indices) < 10:
            print(f"❌ Too few tokens ({len(filtered_indices)}) for focused space")
            return False
        
        # Update data structures
        self.concept_embeddings = self.concept_embeddings[filtered_indices]
        self.concept_tokens = filtered_tokens
        self.token_categories = filtered_categories
        
        print(f"✅ Focused space: {len(filtered_tokens)} tokens across {len(category_counts)} categories")
        print(f"📊 Distribution of the 10 focused concept types:")
        for category, count in category_counts.most_common():
            print(f"   {category:20s}: {count:3d} tokens")
        
        # Verify we have good representation
        if len(category_counts) >= 8:  # At least 8 of the 10 target categories
            print(f"✅ Excellent coverage: {len(category_counts)}/10 target categories found")
        elif len(category_counts) >= 5:
            print(f"⚠️ Decent coverage: {len(category_counts)}/10 target categories found")
        else:
            print(f"⚠️ Limited coverage: {len(category_counts)}/10 target categories found")
        
        return True
    
    def create_focused_visualization(self):
        """Create focused concept space visualization with clear clusters"""
        if self.concept_embeddings is None or not self.token_categories:
            return False
        
        print(f"\n🎨 CREATING FOCUSED CONCEPT SPACE VISUALIZATION")
        print("=" * 55)
        print(f"📊 Visualizing {len(self.concept_tokens)} tokens")
        print(f"🎯 Expecting clearer clusters with focused concept types")
        
        # Optimized PaCMAP for focused space
        print("🔄 Computing PaCMAP projection...")
        pacmap_reducer = PaCMAP(
            n_neighbors=10,      # Slightly fewer neighbors for clearer clusters
            MN_ratio=0.5,        # Balance local/global structure
            FP_ratio=2.0,        # Standard far pairs
            random_state=42,
            apply_pca=True,
            n_components=2
        )
        
        embedding_2d = pacmap_reducer.fit_transform(self.concept_embeddings)
        print("✅ PaCMAP completed!")
        
        # Create focused visualization
        unique_categories = list(set(self.token_categories.values()))
        n_categories = len(unique_categories)
        
        print(f"🎨 Creating visualization with {n_categories} focused categories...")
        
        # High-contrast colors for clear distinction
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_categories, 10)))
        if n_categories > 10:
            colors = np.vstack([colors, plt.cm.Set3(np.linspace(0, 1, n_categories - 10))])
        
        category_colors = dict(zip(unique_categories, colors))
        
        # Create large, detailed plot optimized for cluster visibility
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Plot each category with optimized styling for cluster clarity
        for category in unique_categories:
            indices = [i for i, token in enumerate(self.concept_tokens) 
                      if self.token_categories[token] == category]
            
            if indices:
                x_coords = embedding_2d[indices, 0]
                y_coords = embedding_2d[indices, 1]
                
                color = category_colors[category]
                
                # Category-specific styling - ALL DOTS with different sizes
                if category == 'Special Tokens':
                    size, alpha, edgewidth = 250, 1.0, 3.0
                elif category == 'Countries':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Industries':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Event Types':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Education':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Job Types':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Job Titles':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Acquisition Types':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Investment Types':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Investor Types':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                elif category == 'Company Age':
                    size, alpha, edgewidth = 120, 0.9, 2.0
                else:
                    size, alpha, edgewidth = 100, 0.8, 1.5
                
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=[color], 
                                   label=f'{category} ({len(indices)})',
                                   alpha=alpha, s=size, 
                                   marker='o',  # All dots
                                   edgecolors='white', linewidth=edgewidth, 
                                   zorder=5)
        
        # Enhanced styling for cluster clarity
        ax.set_xlabel('PaCMAP Dimension 1', fontsize=18, fontweight='bold')
        ax.set_ylabel('PaCMAP Dimension 2', fontsize=18, fontweight='bold')
        ax.set_title('Startup2Vec: Focused Concept Space\\n' + 
                    f'10 Key Business Concept Types - Optimized for Clear Clusters ({len(self.concept_tokens)} concepts)', 
                    fontsize=20, fontweight='bold', pad=30)
        # No grid
        
        # Optimized legend for focused categories
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12,
                 frameon=True, fancybox=True, shadow=True, 
                 title='Focused Concept Types', title_fontsize=14)
        
        # Clean background for better cluster visibility
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # Save high-quality version
        save_path = 'startup2vec_FOCUSED_CLUSTERS.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Saved focused clusters visualization: {save_path}")
        
        # Also save PDF
        pdf_path = 'startup2vec_FOCUSED_CLUSTERS.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        print(f"✅ PDF version: {pdf_path}")
        
        plt.show()
        
        return True
    
    def analyze_focused_clusters(self):
        """Analyze the focused concept space for cluster quality"""
        print(f"\n🔍 FOCUSED CLUSTER ANALYSIS")
        print("=" * 40)
        
        if self.concept_embeddings is None:
            return False
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(self.concept_embeddings)
        
        # Analyze within-category vs cross-category similarities
        print("📊 Cluster quality analysis:")
        
        categories = list(set(self.token_categories.values()))
        within_category_sims = []
        cross_category_sims = []
        
        for i in range(len(self.concept_tokens)):
            for j in range(i+1, len(self.concept_tokens)):
                token1 = self.concept_tokens[i]
                token2 = self.concept_tokens[j]
                cat1 = self.token_categories[token1]
                cat2 = self.token_categories[token2]
                similarity = similarity_matrix[i, j]
                
                if cat1 == cat2:
                    within_category_sims.append(similarity)
                else:
                    cross_category_sims.append(similarity)
        
        if within_category_sims and cross_category_sims:
            avg_within = np.mean(within_category_sims)
            avg_cross = np.mean(cross_category_sims)
            cluster_separation = avg_within - avg_cross
            
            print(f"   Average within-category similarity: {avg_within:.3f}")
            print(f"   Average cross-category similarity:  {avg_cross:.3f}")
            print(f"   Cluster separation score:           {cluster_separation:.3f}")
            
            if cluster_separation > 0.1:
                print(f"   ✅ Excellent cluster separation!")
            elif cluster_separation > 0.05:
                print(f"   ✅ Good cluster separation")
            else:
                print(f"   ⚠️ Moderate cluster separation")
        
        # Find most similar cross-category pairs (interesting connections)
        print(f"\n🔗 Most interesting cross-category connections:")
        
        found_pairs = []
        for i in range(len(self.concept_tokens)):
            for j in range(i+1, len(self.concept_tokens)):
                token1 = self.concept_tokens[i]
                token2 = self.concept_tokens[j]
                cat1 = self.token_categories[token1]
                cat2 = self.token_categories[token2]
                
                if cat1 != cat2:
                    similarity = similarity_matrix[i, j]
                    found_pairs.append((similarity, token1, token2, cat1, cat2))
        
        found_pairs.sort(reverse=True)
        for i, (sim, token1, token2, cat1, cat2) in enumerate(found_pairs[:8]):
            print(f"   {sim:.3f}: {str(token1)[:25]} ({cat1}) ↔ {str(token2)[:25]} ({cat2})")
        
        return True
    
    def run_focused_analysis(self):
        """Run the complete focused analysis"""
        print("🎯 FOCUSED STARTUP2VEC CLUSTER ANALYZER")
        print("=" * 60)
        print("Creating highly focused concept space with 10 key concept types")
        print("Optimized for maximum cluster clarity and interpretability")
        print("=" * 60)
        
        steps = [
            ("Loading & verifying checkpoint", self.load_and_verify_checkpoint),
            ("Loading vocabulary", self.load_vocabulary),
            ("Extracting verified embeddings", self.extract_verified_embeddings),
            ("Matching vocab and embeddings", self.match_vocab_and_embeddings),
            ("Filtering for focused space", self.filter_for_focused_space),
            ("Creating focused visualization", self.create_focused_visualization),
            ("Analyzing cluster quality", self.analyze_focused_clusters),
        ]
        
        for step_name, step_func in steps:
            print(f"\n⏳ {step_name}...")
            if not step_func():
                print(f"❌ Failed at: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ FOCUSED CLUSTER ANALYSIS COMPLETE!")
        print("📊 Generated files:")
        print("   • startup2vec_FOCUSED_CLUSTERS.png (focused concept clusters)")
        print("   • startup2vec_FOCUSED_CLUSTERS.pdf (publication-ready)")
        print("🎯 Focus: 10 key concept types for maximum cluster clarity")
        print("🔬 Should show much clearer semantic clusters!")
        print("=" * 60)
        return True

def main():
    visualizer = FocusedConceptVisualizer(device='cpu')
    visualizer.run_focused_analysis()

if __name__ == "__main__":
    main()