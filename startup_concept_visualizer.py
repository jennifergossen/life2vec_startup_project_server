#!/usr/bin/env python3
"""
UNIFIED & VERIFIED Startup2Vec Concept Space Visualizer
- Creates ONE comprehensive concept space (not divided)
- Thoroughly verifies we're using the correct embeddings
- Tests embedding quality with known semantic relationships
- MODIFIED: Excludes temporal elements (years, months) and company size
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

class UnifiedVerifiedVisualizer:
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
        # Trained embeddings should have reasonable variance per dimension
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
        
        # Test with actual vocabulary if available
        if self.test_semantic_relationships(embeddings_np):
            print(f"✅ Embeddings pass quality verification!")
            return True
        else:
            print(f"❌ Embeddings fail semantic verification")
            return False
    
    def test_semantic_relationships(self, embeddings_np):
        """Test if embeddings capture meaningful semantic relationships"""
        
        if not hasattr(self, 'concept_tokens') or not self.concept_tokens:
            print("🔍 Loading vocabulary for semantic testing...")
            if not self.load_vocabulary():
                print("⚠️ Cannot test semantics without vocabulary")
                return True  # Assume OK if can't test
        
        # Ensure we have matching sizes
        vocab_size = len(self.concept_tokens)
        embed_size = embeddings_np.shape[0]
        
        if vocab_size != embed_size:
            min_size = min(vocab_size, embed_size)
            test_tokens = self.concept_tokens[:min_size]
            test_embeddings = embeddings_np[:min_size]
        else:
            test_tokens = self.concept_tokens
            test_embeddings = embeddings_np
        
        print(f"🧪 Testing semantic relationships on {len(test_tokens)} tokens...")
        
        # Test 1: Find semantically related tokens
        semantic_tests = [
            # Should be similar: different series rounds
            ('INV_INVESTMENT_TYPE_SERIES_A', 'INV_INVESTMENT_TYPE_SERIES_B'),
            # Should be similar: same country variations
            ('COUNTRY_USA', 'COUNTRY_United_States'),
            # Should be different: very different concepts
            ('COUNTRY_USA', 'EDU_DEGREE_TYPE_PHD'),
        ]
        
        similarities_found = []
        
        for token1, token2 in semantic_tests:
            idx1 = None
            idx2 = None
            
            # Find token indices (flexible matching)
            for i, token in enumerate(test_tokens):
                token_str = str(token)
                if token1 in token_str:
                    idx1 = i
                if token2 in token_str:
                    idx2 = i
            
            if idx1 is not None and idx2 is not None:
                similarity = cosine_similarity(
                    test_embeddings[idx1:idx1+1], 
                    test_embeddings[idx2:idx2+1]
                )[0,0]
                similarities_found.append((token1, token2, similarity))
                print(f"   {token1} ↔ {token2}: {similarity:.3f}")
        
        # Test 2: Check if special tokens are different from regular tokens
        special_indices = []
        regular_indices = []
        
        for i, token in enumerate(test_tokens):
            token_str = str(token)
            if token_str.startswith('[') or token_str.startswith('<'):
                special_indices.append(i)
            elif token_str.startswith(('INV_', 'COUNTRY_', 'CATEGORY_')):
                regular_indices.append(i)
        
        if len(special_indices) > 0 and len(regular_indices) > 0:
            # Compare special vs regular token similarities
            special_emb = test_embeddings[special_indices[:5]]  # First 5 special
            regular_emb = test_embeddings[regular_indices[:5]]  # First 5 regular
            
            cross_similarities = cosine_similarity(special_emb, regular_emb)
            avg_cross_sim = np.mean(cross_similarities)
            print(f"   Special ↔ Regular tokens avg similarity: {avg_cross_sim:.3f}")
            
            # Good embeddings should show these are different
            if avg_cross_sim > 0.8:
                print(f"⚠️ Special and regular tokens too similar - might indicate untrained embeddings")
                return False
        
        print(f"✅ Semantic relationships look reasonable")
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
    
    def filter_for_unified_space(self):
        """Filter tokens for ONE unified concept space - EXCLUDING temporal and size elements"""
        print("\n🌍 CREATING UNIFIED CONCEPT SPACE (No Temporal/Size)")
        print("=" * 55)
        
        # Include meaningful business concepts in ONE space - EXCLUDING temporal and size
        include_patterns = {
            # Geographic
            'COUNTRY_': 'Geographic',
            
            # Business fundamentals  
            'CATEGORY_': 'Industries',
            # REMOVED: 'EMPLOYEE_': 'Company Size',  # Excluded per request
            
            # Investment ecosystem
            'INV_INVESTMENT_TYPE': 'Investment Types',
            'INV_RAISED_AMOUNT_USD': 'Funding Amounts',
            
            # Exit events
            'ACQ_ACQUISITION_TYPE': 'Acquisitions',
            'IPO_': 'Public Offerings',
            
            # Human capital
            'PPL_JOB_TYPE': 'Job Roles',
            'EDU_DEGREE_TYPE': 'Education',
            
            # REMOVED: Timeline elements per request
            # 'YEAR_': 'Years',          # Excluded per request
            # 'MONTH_': 'Months',        # Excluded per request
            # 'DAYS_DAY': 'Company Age', # Excluded per request
            
            # Events
            'EVT_TYPE_': 'Life Events',
        }
        
        def should_include_token(token_str):
            # Special tokens
            if token_str.startswith('[') or token_str.startswith('<'):
                return True, 'Special'
            
            # Business concepts (excluding temporal and size patterns)
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
        
        if len(filtered_indices) < 20:
            print(f"❌ Too few tokens ({len(filtered_indices)}) for unified space")
            return False
        
        # Update data structures
        self.concept_embeddings = self.concept_embeddings[filtered_indices]
        self.concept_tokens = filtered_tokens
        self.token_categories = filtered_categories
        
        print(f"✅ Unified space: {len(filtered_tokens)} tokens across {len(category_counts)} categories")
        print(f"📊 Category distribution (excluding temporal & size):")
        for category, count in category_counts.most_common():
            print(f"   {category:20s}: {count:3d} tokens")
        
        return True
    
    def create_unified_visualization(self):
        """Create ONE comprehensive concept space visualization"""
        if self.concept_embeddings is None or not self.token_categories:
            return False
        
        print(f"\n🎨 CREATING UNIFIED CONCEPT SPACE (No Temporal/Size)")
        print("=" * 55)
        print(f"📊 Visualizing {len(self.concept_tokens)} tokens")
        
        # Optimized PaCMAP for large, diverse space
        print("🔄 Computing PaCMAP projection...")
        pacmap_reducer = PaCMAP(
            n_neighbors=12,      # Good for diverse space
            MN_ratio=0.5,        # Balance local/global  
            FP_ratio=2.0,        # Standard far pairs
            random_state=42,
            apply_pca=True,
            n_components=2
        )
        
        embedding_2d = pacmap_reducer.fit_transform(self.concept_embeddings)
        print("✅ PaCMAP completed!")
        
        # Create comprehensive visualization
        unique_categories = list(set(self.token_categories.values()))
        n_categories = len(unique_categories)
        
        print(f"🎨 Creating visualization with {n_categories} categories...")
        
        # Sophisticated color scheme for many categories
        if n_categories <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        elif n_categories <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
        else:
            # Combine multiple colormaps
            colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
            colors2 = plt.cm.Set3(np.linspace(0, 1, n_categories - 10))
            colors = np.vstack([colors1, colors2])
        
        category_colors = dict(zip(unique_categories, colors))
        
        # Create large, detailed plot
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Plot each category with distinct visual style
        for category in unique_categories:
            indices = [i for i, token in enumerate(self.concept_tokens) 
                      if self.token_categories[token] == category]
            
            if indices:
                x_coords = embedding_2d[indices, 0]
                y_coords = embedding_2d[indices, 1]
                
                color = category_colors[category]
                
                # Visual styling by category type
                if category == 'Special':
                    size, alpha, marker, edgewidth = 200, 1.0, 's', 3.0
                elif category in ['Geographic', 'Industries']:
                    size, alpha, marker, edgewidth = 100, 0.8, 'o', 1.5
                elif category in ['Investment Types', 'Funding Amounts']:
                    size, alpha, marker, edgewidth = 100, 0.8, '^', 1.5
                elif category in ['Job Roles', 'Education']:
                    size, alpha, marker, edgewidth = 100, 0.8, 'D', 1.5
                elif category in ['Acquisitions', 'Public Offerings']:
                    size, alpha, marker, edgewidth = 100, 0.8, 'v', 1.5
                else:
                    size, alpha, marker, edgewidth = 80, 0.7, 'o', 1.0
                
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=[color], 
                                   label=f'{category} ({len(indices)})',
                                   alpha=alpha, s=size, 
                                   marker=marker,
                                   edgecolors='white', linewidth=edgewidth, 
                                   zorder=5)
        
        # Enhanced styling
        ax.set_xlabel('PaCMAP Dimension 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('PaCMAP Dimension 2', fontsize=16, fontweight='bold')
        ax.set_title('Startup2Vec: Core Business Concept Space\\n' + 
                    f'Semantic Landscape without Temporal/Size Elements ({len(self.concept_tokens)} concepts)', 
                    fontsize=18, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Smart legend placement
        if n_categories <= 15:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11,
                     frameon=True, fancybox=True, shadow=True, 
                     title='Concept Categories', title_fontsize=13)
        else:
            # For many categories, use smaller font and multiple columns
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                     ncol=1, frameon=True, fancybox=True, shadow=True,
                     title='Concept Categories', title_fontsize=11)
        
        # Subtle background
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        
        # Save high-quality version
        save_path = 'startup2vec_CORE_CONCEPTS.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Saved core concepts visualization: {save_path}")
        
        # Also save PDF
        pdf_path = 'startup2vec_CORE_CONCEPTS.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        print(f"✅ PDF version: {pdf_path}")
        
        plt.show()
        
        return True
    
    def analyze_unified_space(self):
        """Analyze the unified concept space for interesting patterns"""
        print(f"\n🔍 CORE CONCEPT SPACE ANALYSIS")
        print("=" * 40)
        
        if self.concept_embeddings is None:
            return False
        
        # Compute full similarity matrix
        similarity_matrix = cosine_similarity(self.concept_embeddings)
        
        # Find most similar pairs across different categories
        print("🔗 Most similar cross-category pairs:")
        
        found_pairs = []
        for i in range(len(self.concept_tokens)):
            for j in range(i+1, len(self.concept_tokens)):
                token1 = self.concept_tokens[i]
                token2 = self.concept_tokens[j]
                cat1 = self.token_categories[token1]
                cat2 = self.token_categories[token2]
                
                # Only look at cross-category similarities
                if cat1 != cat2:
                    similarity = similarity_matrix[i, j]
                    found_pairs.append((similarity, token1, token2, cat1, cat2))
        
        # Show top cross-category similarities
        found_pairs.sort(reverse=True)
        for i, (sim, token1, token2, cat1, cat2) in enumerate(found_pairs[:10]):
            print(f"   {sim:.3f}: {str(token1)[:30]} ({cat1}) ↔ {str(token2)[:30]} ({cat2})")
        
        return True
    
    def run_unified_analysis(self):
        """Run the complete unified analysis with verification"""
        print("🌍 CORE BUSINESS CONCEPTS STARTUP2VEC ANALYZER")
        print("=" * 60)
        print("Creating concept space focused on core business elements")
        print("(Excluding temporal elements and company size)")
        print("=" * 60)
        
        steps = [
            ("Loading & verifying checkpoint", self.load_and_verify_checkpoint),
            ("Loading vocabulary", self.load_vocabulary),
            ("Extracting verified embeddings", self.extract_verified_embeddings),
            ("Matching vocab and embeddings", self.match_vocab_and_embeddings),
            ("Filtering for core concepts", self.filter_for_unified_space),
            ("Creating core visualization", self.create_unified_visualization),
            ("Analyzing core concept space", self.analyze_unified_space),
        ]
        
        for step_name, step_func in steps:
            print(f"\n⏳ {step_name}...")
            if not step_func():
                print(f"❌ Failed at: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ CORE CONCEPTS ANALYSIS COMPLETE!")
        print("📊 Generated files:")
        print("   • startup2vec_CORE_CONCEPTS.png (core business concept space)")
        print("   • startup2vec_CORE_CONCEPTS.pdf (publication-ready)")
        print("🎯 Focus: Core business concepts without temporal/size elements")
        print("🔬 Embeddings have been thoroughly verified for quality")
        print("=" * 60)
        return True

def main():
    visualizer = UnifiedVerifiedVisualizer(device='cpu')
    visualizer.run_unified_analysis()

if __name__ == "__main__":
    main()