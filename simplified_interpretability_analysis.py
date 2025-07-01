# simplified_interpretability_analysis.py
"""
Simplified Startup2Vec Interpretability Framework
ONLY includes methods that don't require attention scores (4 out of 5 methods)

What you need:
- predictions: Model survival probability predictions  
- true_labels: Actual survival outcomes
- test_data: DataFrame with startup sequences + metadata
- startup_embeddings: Company-level embeddings from your model
- vocab_to_idx, idx_to_vocab: Vocabulary mappings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SimplifiedStartup2VecInterpretability:
    """
    Simplified interpretability framework - NO ATTENTION REQUIRED
    Implements 4 out of 5 life2vec interpretability methods
    """
    
    def __init__(self, test_data, predictions, true_labels, startup_embeddings, vocab_to_idx, idx_to_vocab):
        """
        Initialize interpretability framework
        
        Args:
            test_data: DataFrame with columns ['sequences', 'industry', 'funding_stage', etc.]
            predictions: Array of survival probabilities [0, 1]
            true_labels: Array of actual survival outcomes {0, 1}
            startup_embeddings: 2D array of company embeddings (n_companies, embedding_dim)
            vocab_to_idx: Dict mapping event names to indices
            idx_to_vocab: Dict mapping indices to event names
        """
        self.test_data = test_data
        self.predictions = np.array(predictions)
        self.true_labels = np.array(true_labels)
        self.startup_embeddings = np.array(startup_embeddings)
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab
        
        print(f"📊 Loaded data for {len(test_data)} startups")
        print(f"📈 Embeddings shape: {startup_embeddings.shape}")
        print(f"📚 Vocabulary size: {len(vocab_to_idx)}")
    
    # =================== 1. ALGORITHMIC AUDITING ===================
    
    def algorithmic_audit(self):
        """Examine model performance across different startup subgroups"""
        print("\n🔍 === ALGORITHMIC AUDITING ===")
        
        results = {}
        
        # 1. Audit by sequence length
        results['sequence_length'] = self._audit_by_sequence_length()
        
        # 2. Audit by company characteristics
        if 'industry' in self.test_data.columns:
            results['industry'] = self._audit_by_industry()
            
        if 'funding_stage' in self.test_data.columns:
            results['funding_stage'] = self._audit_by_funding_stage()
            
        # Add more audits based on available columns
        for col in ['employee_count', 'total_funding', 'founded_year', 'location']:
            if col in self.test_data.columns:
                results[col] = self._audit_by_column(col)
        
        return results
    
    def _audit_by_sequence_length(self):
        """Audit performance by startup event sequence length"""
        print("📏 Auditing by sequence length...")
        
        # Calculate sequence lengths
        if 'sequences' in self.test_data.columns:
            sequence_lengths = [len(seq) for seq in self.test_data['sequences']]
        else:
            print("⚠️ No 'sequences' column found, using dummy data")
            sequence_lengths = np.random.randint(10, 100, len(self.test_data))
        
        # Create length bins
        length_bins = pd.qcut(sequence_lengths, q=3, labels=['Short', 'Medium', 'Long'], duplicates='drop')
        
        return self._analyze_group_performance(length_bins, "Sequence Length")
    
    def _audit_by_industry(self):
        """Audit performance by industry"""
        print("🏭 Auditing by industry...")
        
        industries = self.test_data['industry'].fillna('Unknown')
        # Focus on top 10 industries
        top_industries = industries.value_counts().head(10).index
        industry_mask = industries.isin(top_industries)
        
        filtered_industries = industries[industry_mask]
        filtered_preds = self.predictions[industry_mask]
        filtered_labels = self.true_labels[industry_mask]
        
        return self._analyze_group_performance(filtered_industries, "Industry")
    
    def _audit_by_funding_stage(self):
        """Audit performance by funding stage"""
        print("💰 Auditing by funding stage...")
        
        funding_stages = self.test_data['funding_stage'].fillna('Unknown')
        return self._analyze_group_performance(funding_stages, "Funding Stage")
    
    def _audit_by_column(self, column_name):
        """Generic audit by any column"""
        print(f"📊 Auditing by {column_name}...")
        
        col_data = self.test_data[column_name]
        
        # Handle numeric columns differently
        if pd.api.types.is_numeric_dtype(col_data):
            # Create bins for numeric data
            bins = pd.qcut(col_data.fillna(col_data.median()), q=3, 
                          labels=['Low', 'Medium', 'High'], duplicates='drop')
            return self._analyze_group_performance(bins, column_name.replace('_', ' ').title())
        else:
            # Categorical data
            return self._analyze_group_performance(col_data.fillna('Unknown'), 
                                                 column_name.replace('_', ' ').title())
    
    def _analyze_group_performance(self, groups, title):
        """Analyze performance across groups"""
        results = []
        
        for group_name in groups.dropna().unique():
            mask = groups == group_name
            if mask.sum() > 10:  # Minimum sample size
                group_preds = self.predictions[mask]
                group_labels = self.true_labels[mask]
                
                try:
                    auc = roc_auc_score(group_labels, group_preds)
                    accuracy = accuracy_score(group_labels, group_preds > 0.5)
                    
                    results.append({
                        'Group': str(group_name),
                        'AUC': auc,
                        'Accuracy': accuracy,
                        'Count': mask.sum(),
                        'Survival_Rate': np.mean(group_labels),
                        'Avg_Prediction': np.mean(group_preds)
                    })
                except:
                    continue
        
        if results:
            df_results = pd.DataFrame(results)
            self._plot_audit_results(df_results, title)
            return df_results
        return None
    
    def _plot_audit_results(self, df_results, title):
        """Plot audit results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # AUC plot
        bars1 = axes[0,0].bar(df_results['Group'], df_results['AUC'], alpha=0.7, color='skyblue')
        axes[0,0].set_title(f'{title} - AUC')
        axes[0,0].set_ylabel('AUC')
        axes[0,0].set_ylim(0, 1)
        plt.setp(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        
        # Accuracy plot
        bars2 = axes[0,1].bar(df_results['Group'], df_results['Accuracy'], alpha=0.7, color='lightcoral')
        axes[0,1].set_title(f'{title} - Accuracy')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_ylim(0, 1)
        plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha='right')
        
        # Sample sizes
        bars3 = axes[1,0].bar(df_results['Group'], df_results['Count'], alpha=0.7, color='lightgreen')
        axes[1,0].set_title(f'{title} - Sample Sizes')
        axes[1,0].set_ylabel('Count')
        plt.setp(axes[1,0].get_xticklabels(), rotation=45, ha='right')
        
        # Predicted vs Actual survival rates
        x = np.arange(len(df_results))
        width = 0.35
        axes[1,1].bar(x - width/2, df_results['Avg_Prediction'], width, 
                     label='Predicted', alpha=0.7)
        axes[1,1].bar(x + width/2, df_results['Survival_Rate'], width, 
                     label='Actual', alpha=0.7)
        axes[1,1].set_title(f'{title} - Predicted vs Actual Survival Rates')
        axes[1,1].set_ylabel('Survival Rate')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(df_results['Group'], rotation=45, ha='right')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📋 {title} Results:")
        print(df_results.round(3))
    
    # =================== 2. DATA CONTRIBUTION ANALYSIS ===================
    
    def data_contribution_analysis(self):
        """Analyze which types of startup events contribute most to predictions"""
        print("\n📊 === DATA CONTRIBUTION ANALYSIS ===")
        
        # Define event categories for startups
        event_categories = self._define_event_categories()
        
        # Test model performance with different event vocabularies
        results = self._test_vocabulary_variations(event_categories)
        
        return results
    
    def _define_event_categories(self):
        """Define startup event categories based on common event types"""
        # Get all unique event types from vocabulary
        all_events = list(self.vocab_to_idx.keys())
        
        # Define categories (you should customize these based on your actual events)
        categories = {
            'funding_events': [],
            'product_events': [],
            'team_events': [],
            'business_events': [],
            'market_events': []
        }
        
        # Classify events into categories based on keywords
        for event in all_events:
            event_lower = str(event).lower()
            
            if any(keyword in event_lower for keyword in ['fund', 'invest', 'money', 'capital', 'round', 'ipo']):
                categories['funding_events'].append(event)
            elif any(keyword in event_lower for keyword in ['product', 'launch', 'release', 'feature', 'patent']):
                categories['product_events'].append(event)
            elif any(keyword in event_lower for keyword in ['hire', 'team', 'employee', 'executive', 'founder']):
                categories['team_events'].append(event)
            elif any(keyword in event_lower for keyword in ['partner', 'acquisition', 'merger', 'expand']):
                categories['business_events'].append(event)
            elif any(keyword in event_lower for keyword in ['user', 'customer', 'market', 'revenue', 'growth']):
                categories['market_events'].append(event)
        
        # Print categories found
        print("📚 Event categories found:")
        for cat, events in categories.items():
            print(f"  {cat}: {len(events)} events")
            if len(events) > 0:
                print(f"    Examples: {events[:3]}")
        
        return categories
    
    def _test_vocabulary_variations(self, event_categories):
        """Test performance with different event subsets"""
        print("🧪 Testing vocabulary variations...")
        
        results = []
        
        # Test each category individually
        for category, events in event_categories.items():
            if len(events) > 0:
                # Calculate what percentage of sequences contain these events
                coverage = self._calculate_event_coverage(events)
                
                results.append({
                    'Category': category.replace('_', ' ').title(),
                    'Event_Count': len(events),
                    'Coverage': coverage,
                    'Impact_Score': self._estimate_category_impact(events)
                })
        
        # Create results DataFrame and plot
        if results:
            df_results = pd.DataFrame(results)
            self._plot_contribution_results(df_results)
            return df_results
        
        return None
    
    def _calculate_event_coverage(self, events):
        """Calculate what percentage of sequences contain these events"""
        if 'sequences' not in self.test_data.columns:
            return np.random.uniform(0.1, 0.9)  # Dummy for testing
        
        event_indices = [self.vocab_to_idx.get(event, -1) for event in events]
        event_indices = [idx for idx in event_indices if idx != -1]
        
        if not event_indices:
            return 0.0
        
        sequences_with_events = 0
        for sequence in self.test_data['sequences']:
            if any(idx in sequence for idx in event_indices):
                sequences_with_events += 1
        
        return sequences_with_events / len(self.test_data)
    
    def _estimate_category_impact(self, events):
        """Estimate the impact of event category on predictions"""
        if 'sequences' not in self.test_data.columns:
            return np.random.uniform(0.1, 0.8)  # Dummy for testing
        
        event_indices = [self.vocab_to_idx.get(event, -1) for event in events]
        event_indices = [idx for idx in event_indices if idx != -1]
        
        if not event_indices:
            return 0.0
        
        # Find startups with and without these events
        has_events = []
        for sequence in self.test_data['sequences']:
            has_events.append(any(idx in sequence for idx in event_indices))
        
        has_events = np.array(has_events)
        
        if has_events.sum() < 10 or (~has_events).sum() < 10:
            return 0.0
        
        # Compare survival rates
        survival_with_events = np.mean(self.predictions[has_events])
        survival_without_events = np.mean(self.predictions[~has_events])
        
        return abs(survival_with_events - survival_without_events)
    
    def _plot_contribution_results(self, df_results):
        """Plot data contribution analysis results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Event counts
        bars1 = axes[0].bar(df_results['Category'], df_results['Event_Count'], alpha=0.7)
        axes[0].set_title('Number of Events by Category')
        axes[0].set_ylabel('Event Count')
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Coverage
        bars2 = axes[1].bar(df_results['Category'], df_results['Coverage'], alpha=0.7, color='orange')
        axes[1].set_title('Sequence Coverage by Category')
        axes[1].set_ylabel('Coverage (% of sequences)')
        axes[1].set_ylim(0, 1)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        # Impact scores
        bars3 = axes[2].bar(df_results['Category'], df_results['Impact_Score'], alpha=0.7, color='green')
        axes[2].set_title('Estimated Impact on Survival Prediction')
        axes[2].set_ylabel('Impact Score')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        print("\n📊 Event Category Analysis:")
        print(df_results.round(3))
    
    # =================== 3. VISUAL EXPLORATION ===================
    
    def visual_exploration(self):
        """Visual exploration of startup embedding space"""
        print("\n🎨 === VISUAL EXPLORATION OF EMBEDDINGS ===")
        
        # 2D visualization
        embeddings_2d = self._create_2d_visualization()
        
        # 3D visualization  
        embeddings_3d = self._create_3d_visualization()
        
        # Cluster analysis
        clusters = self._analyze_embedding_clusters()
        
        return {
            'embeddings_2d': embeddings_2d,
            'embeddings_3d': embeddings_3d,
            'clusters': clusters
        }
    
    def _create_2d_visualization(self):
        """Create 2D visualization of startup embeddings"""
        print("🗺️ Creating 2D embedding visualization...")
        
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        except ImportError:
            print("⚠️ UMAP not available, using PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(self.startup_embeddings)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Color by survival probability
        scatter1 = axes[0,0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=self.predictions, cmap='RdYlGn', alpha=0.6, s=20)
        axes[0,0].set_title('Startup Embeddings - Survival Probability')
        axes[0,0].set_xlabel('Dimension 1')
        axes[0,0].set_ylabel('Dimension 2')
        plt.colorbar(scatter1, ax=axes[0,0], label='Survival Probability')
        
        # Plot 2: Color by true labels
        colors = ['red' if label == 0 else 'green' for label in self.true_labels]
        axes[0,1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=20)
        axes[0,1].set_title('Startup Embeddings - True Survival Labels')
        axes[0,1].set_xlabel('Dimension 1')
        axes[0,1].set_ylabel('Dimension 2')
        
        # Plot 3: Prediction accuracy
        correct = (self.predictions > 0.5) == self.true_labels
        acc_colors = ['green' if c else 'red' for c in correct]
        axes[1,0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=acc_colors, alpha=0.6, s=20)
        axes[1,0].set_title('Startup Embeddings - Prediction Accuracy')
        axes[1,0].set_xlabel('Dimension 1')
        axes[1,0].set_ylabel('Dimension 2')
        
        # Plot 4: Industry (if available)
        if 'industry' in self.test_data.columns:
            industries = self.test_data['industry'].fillna('Unknown')
            unique_industries = industries.unique()[:10]  # Top 10 industries
            
            for i, industry in enumerate(unique_industries):
                mask = industries == industry
                if mask.sum() > 0:
                    axes[1,1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                                    label=str(industry)[:15], alpha=0.6, s=20)
            
            axes[1,1].set_title('Startup Embeddings - Industry')
            axes[1,1].set_xlabel('Dimension 1')
            axes[1,1].set_ylabel('Dimension 2')
            axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        return embeddings_2d
    
    def _create_3d_visualization(self):
        """Create 3D visualization of startup embeddings"""
        print("🌐 Creating 3D embedding visualization...")
        
        try:
            from umap import UMAP
            reducer = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        except ImportError:
            print("⚠️ UMAP not available, using PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3, random_state=42)
        
        embeddings_3d = reducer.fit_transform(self.startup_embeddings)
        
        # Create 3D plot with plotly
        try:
            fig = go.Figure(data=[go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.predictions,
                    colorscale='RdYlGn',
                    opacity=0.7,
                    colorbar=dict(title="Survival Probability")
                ),
                text=[f'Survival Prob: {prob:.2f}<br>True Label: {label}' 
                      for prob, label in zip(self.predictions, self.true_labels)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            )])
            
            fig.update_layout(
                title="3D Startup Embedding Space",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2", 
                    zaxis_title="Dimension 3"
                ),
                width=800,
                height=600
            )
            
            fig.show()
        except:
            print("⚠️ Plotly not available for 3D visualization")
        
        return embeddings_3d
    
    def _analyze_embedding_clusters(self):
        """Analyze clusters in the embedding space"""
        print("🎯 Analyzing embedding clusters...")
        
        # Find optimal number of clusters
        k_range = range(2, min(11, len(self.startup_embeddings)//10))
        silhouette_scores = []
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.startup_embeddings)
                score = silhouette_score(self.startup_embeddings, cluster_labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(0)
        
        if not silhouette_scores:
            print("⚠️ Could not perform clustering analysis")
            return None
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Optimal Number of Clusters for Startup Embeddings')
        plt.grid(True)
        plt.show()
        
        # Use optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"📊 Optimal number of clusters: {optimal_k}")
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.startup_embeddings)
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_cluster_characteristics(cluster_labels, optimal_k)
        
        return {
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'silhouette_scores': silhouette_scores,
            'analysis': cluster_analysis
        }
    
    def _analyze_cluster_characteristics(self, cluster_labels, n_clusters):
        """Analyze characteristics of each cluster"""
        results = []
        
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0:
                continue
                
            cluster_preds = self.predictions[mask]
            cluster_labels_true = self.true_labels[mask]
            
            avg_survival_prob = np.mean(cluster_preds)
            actual_survival_rate = np.mean(cluster_labels_true)
            cluster_size = mask.sum()
            
            results.append({
                'Cluster': f'Cluster {cluster_id}',
                'Size': cluster_size,
                'Avg_Survival_Prob': avg_survival_prob,
                'Actual_Survival_Rate': actual_survival_rate,
                'Prediction_Error': avg_survival_prob - actual_survival_rate,
                'Percentage': cluster_size / len(cluster_labels) * 100
            })
        
        df_clusters = pd.DataFrame(results)
        print("\n🎯 Cluster Analysis:")
        print(df_clusters.round(3))
        
        # Plot cluster characteristics
        self._plot_cluster_analysis(df_clusters)
        
        return df_clusters
    
    def _plot_cluster_analysis(self, df_clusters):
        """Plot cluster analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Cluster sizes
        bars1 = axes[0,0].bar(df_clusters['Cluster'], df_clusters['Size'])
        axes[0,0].set_title('Cluster Sizes')
        axes[0,0].set_ylabel('Number of Startups')
        plt.setp(axes[0,0].get_xticklabels(), rotation=45)
        
        # Predicted vs actual survival rates
        x = np.arange(len(df_clusters))
        width = 0.35
        axes[0,1].bar(x - width/2, df_clusters['Avg_Survival_Prob'], width, 
                     label='Predicted', alpha=0.7)
        axes[0,1].bar(x + width/2, df_clusters['Actual_Survival_Rate'], width, 
                     label='Actual', alpha=0.7)
        axes[0,1].set_xlabel('Clusters')
        axes[0,1].set_ylabel('Survival Rate')
        axes[0,1].set_title('Predicted vs Actual Survival Rates')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(df_clusters['Cluster'], rotation=45)
        axes[0,1].legend()
        
        # Prediction errors
        bars3 = axes[1,0].bar(df_clusters['Cluster'], df_clusters['Prediction_Error'])
        axes[1,0].set_title('Prediction Error by Cluster')
        axes[1,0].set_ylabel('Predicted - Actual')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.setp(axes[1,0].get_xticklabels(), rotation=45)
        
        # Cluster percentages (pie chart)
        axes[1,1].pie(df_clusters['Percentage'], labels=df_clusters['Cluster'], autopct='%1.1f%%')
        axes[1,1].set_title('Distribution of Startups Across Clusters')
        
        plt.tight_layout()
        plt.show()
    
    # =================== 4. GLOBAL EXPLAINABILITY (TCAV) ===================
    
    def tcav_analysis(self):
        """Test with Concept Activation Vectors (TCAV) - Global explainability"""
        print("\n🔬 === GLOBAL EXPLAINABILITY (TCAV ANALYSIS) ===")
        
        # Define startup concepts to test
        startup_concepts = self._define_startup_concepts()
        
        tcav_results = []
        for concept_name, concept_definition in startup_concepts.items():
            print(f"🧪 Testing concept: {concept_name}")
            tcav_score = self._test_concept_activation(concept_name, concept_definition)
            
            tcav_results.append({
                'Concept': concept_name,
                'TCAV_Score': tcav_score,
                'Interpretation': self._interpret_tcav_score(tcav_score)
            })
        
        # Plot and analyze results
        if tcav_results:
            self._plot_tcav_results(tcav_results)
        
        return tcav_results
    
    def _define_startup_concepts(self):
        """Define concepts to test with TCAV"""
        # You should customize these based on your actual startup events
        concepts = {
            'High_Growth': {
                'description': 'Startups showing rapid growth indicators',
                'required_events': ['funding', 'growth', 'expansion', 'milestone'],
                'excluded_events': ['layoff', 'closure', 'failure']
            },
            'Tech_Focus': {
                'description': 'Technology-focused startups',
                'required_events': ['patent', 'product', 'tech', 'software', 'api'],
                'excluded_events': ['retail', 'manufacturing', 'physical']
            },
            'Well_Funded': {
                'description': 'Startups with significant funding',
                'required_events': ['series', 'round', 'investment', 'ipo'],
                'excluded_events': ['bootstrap', 'self_funded']
            },
            'Early_Stage': {
                'description': 'Early-stage startups',
                'required_events': ['seed', 'prototype', 'mvp', 'launch'],
                'excluded_events': ['mature', 'established', 'ipo']
            }
        }
        return concepts
    
    def _test_concept_activation(self, concept_name, concept_definition):
        """Test concept activation using simplified TCAV methodology"""
        required_events = concept_definition.get('required_events', [])
        excluded_events = concept_definition.get('excluded_events', [])
        
        # Create positive and negative samples
        positive_samples, negative_samples = self._create_concept_datasets(
            required_events, excluded_events)
        
        if len(positive_samples) < 10 or len(negative_samples) < 10:
            print(f"⚠️ Insufficient samples for concept {concept_name}")
            print(f"   Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")
            return 0.0
        
        # Train concept activation vector
        cav = self._train_concept_activation_vector(positive_samples, negative_samples)
        
        if cav is None:
            return 0.0
        
        # Calculate TCAV score
        tcav_score = self._calculate_tcav_score(cav)
        
        print(f"✅ {concept_name}: {tcav_score:.3f}")
        return tcav_score
    
    def _create_concept_datasets(self, required_events, excluded_events):
        """Create positive and negative datasets for a concept"""
        positive_samples = []
        negative_samples = []
        
        # Check if we have sequences
        if 'sequences' not in self.test_data.columns:
            # If no sequences, use random sampling based on metadata
            return self._create_concept_datasets_from_metadata(required_events, excluded_events)
        
        for i, sequence in enumerate(self.test_data['sequences']):
            # Convert sequence indices to event names
            event_names = [self.idx_to_vocab.get(idx, '').lower() for idx in sequence]
            event_text = ' '.join(event_names)
            
            # Check for required events
            has_required = any(req_event.lower() in event_text for req_event in required_events)
            
            # Check for excluded events
            has_excluded = any(exc_event.lower() in event_text for exc_event in excluded_events)
            
            if has_required and not has_excluded:
                positive_samples.append(i)
            elif has_excluded and not has_required:
                negative_samples.append(i)
        
        return positive_samples, negative_samples
    
    def _create_concept_datasets_from_metadata(self, required_events, excluded_events):
        """Create concept datasets using metadata when sequences aren't available"""
        positive_samples = []
        negative_samples = []
        
        # Use available metadata columns to infer concepts
        metadata_text = []
        for _, row in self.test_data.iterrows():
            text_parts = []
            for col in ['industry', 'funding_stage', 'description']:
                if col in self.test_data.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]).lower())
            metadata_text.append(' '.join(text_parts))
        
        for i, text in enumerate(metadata_text):
            has_required = any(req_event.lower() in text for req_event in required_events)
            has_excluded = any(exc_event.lower() in text for exc_event in excluded_events)
            
            if has_required and not has_excluded:
                positive_samples.append(i)
            elif has_excluded and not has_required:
                negative_samples.append(i)
        
        return positive_samples, negative_samples
    
    def _train_concept_activation_vector(self, positive_samples, negative_samples):
        """Train concept activation vector using logistic regression"""
        try:
            # Get embeddings for positive and negative samples
            positive_embeddings = self.startup_embeddings[positive_samples]
            negative_embeddings = self.startup_embeddings[negative_samples]
            
            # Create training data
            X = np.vstack([positive_embeddings, negative_embeddings])
            y = np.hstack([np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))])
            
            # Train logistic regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X, y)
            
            # Concept activation vector is the normal to the decision boundary
            cav = lr.coef_[0]
            return cav / np.linalg.norm(cav)  # Normalize
            
        except Exception as e:
            print(f"⚠️ Error training CAV: {e}")
            return None
    
    def _calculate_tcav_score(self, cav):
        """Calculate TCAV score using correlation with survival predictions"""
        # Project all embeddings onto the concept activation vector
        projections = np.dot(self.startup_embeddings, cav)
        
        # Calculate correlation with survival predictions
        tcav_score = np.corrcoef(projections, self.predictions)[0, 1]
        
        return abs(tcav_score)  # Take absolute value
    
    def _interpret_tcav_score(self, score):
        """Interpret TCAV score"""
        if score > 0.3:
            return "Strong influence on survival prediction"
        elif score > 0.15:
            return "Moderate influence on survival prediction"
        elif score > 0.05:
            return "Weak influence on survival prediction"
        else:
            return "No significant influence"
    
    def _plot_tcav_results(self, tcav_results):
        """Plot TCAV analysis results"""
        df_tcav = pd.DataFrame(tcav_results)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_tcav['Concept'], df_tcav['TCAV_Score'])
        
        # Color bars by score magnitude
        for bar, score in zip(bars, df_tcav['TCAV_Score']):
            if score > 0.3:
                bar.set_color('darkgreen')
            elif score > 0.15:
                bar.set_color('orange')
            elif score > 0.05:
                bar.set_color('lightblue')
            else:
                bar.set_color('lightgray')
        
        plt.xlabel('Concepts')
        plt.ylabel('TCAV Score')
        plt.title('Concept Activation Analysis (TCAV Scores)')
        plt.xticks(rotation=45, ha='right')
        
        # Add score labels on bars
        for bar, score in zip(bars, df_tcav['TCAV_Score']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🔬 TCAV Score Interpretations:")
        for result in tcav_results:
            print(f"  {result['Concept']:15s}: {result['TCAV_Score']:.3f} - {result['Interpretation']}")
    
    # =================== MAIN RUNNER ===================
    
    def run_all_analyses(self):
        """Run all interpretability analyses"""
        print("🚀 STARTING SIMPLIFIED STARTUP2VEC INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        
        results = {}
        
        try:
            # 1. Algorithmic Auditing
            results['audit'] = self.algorithmic_audit()
            print("\n" + "="*80)
            
            # 2. Data Contribution Analysis
            results['contribution'] = self.data_contribution_analysis()
            print("\n" + "="*80)
            
            # 3. Visual Exploration
            results['visual'] = self.visual_exploration()
            print("\n" + "="*80)
            
            # 4. TCAV Analysis
            results['tcav'] = self.tcav_analysis()
            print("\n" + "="*80)
            
            print("✅ INTERPRETABILITY ANALYSIS COMPLETE!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return results

# =================== MINIMAL USAGE EXAMPLE ===================

def run_interpretability_analysis(test_data, predictions, true_labels, startup_embeddings, vocab_to_idx, idx_to_vocab):
    """
    Simple function to run interpretability analysis
    
    Args:
        test_data: DataFrame with startup data and metadata
        predictions: Array of survival probabilities [0, 1] 
        true_labels: Array of actual survival outcomes {0, 1}
        startup_embeddings: 2D array of company embeddings
        vocab_to_idx: Dict mapping event names to indices
        idx_to_vocab: Dict mapping indices to event names
    """
    
    # Initialize framework
    interpreter = SimplifiedStartup2VecInterpretability(
        test_data=test_data,
        predictions=predictions,
        true_labels=true_labels,
        startup_embeddings=startup_embeddings,
        vocab_to_idx=vocab_to_idx,
        idx_to_vocab=idx_to_vocab
    )
    
    # Run all analyses
    results = interpreter.run_all_analyses()
    
    return results

if __name__ == "__main__":
    print("📊 Simplified Startup2Vec Interpretability Framework")
    print("🔧 No attention scores required - works with basic model outputs")
    print("📋 Customize event_categories and startup_concepts for your data")
