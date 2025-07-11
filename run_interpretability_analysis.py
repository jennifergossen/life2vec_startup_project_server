# run_interpretability_analysis.py
"""
ROBUST INTERPRETABILITY ANALYSIS FOR LARGE DATASETS
Handles class imbalance, large sample sizes, and provides comprehensive insights
"""

import pickle
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_extracted_data(data_dir="interpretability_results"):
    """Load the extracted interpretability data"""
    
    print("📂 Loading extracted interpretability data...")
    
    try:
        # Load main interpretability data
        data_path = os.path.join(data_dir, 'interpretability_data.pkl')
        with open(data_path, 'rb') as f:
            interpretability_data = pickle.load(f)
        
        # Load test data with metadata
        test_data_path = os.path.join(data_dir, 'test_data_with_metadata.pkl')
        test_data = pd.read_pickle(test_data_path)
        
        print(f"✅ Data loaded successfully")
        print(f"📊 Samples: {len(interpretability_data['predictions']):,}")
        print(f"📏 Embedding dim: {interpretability_data['startup_embeddings'].shape[1]}")
        print(f"📚 Vocab size: {len(interpretability_data['vocab_to_idx']):,}")
        
        # Check data distribution
        if 'data_distribution' in interpretability_data:
            dist = interpretability_data['data_distribution']
            if 'survival_counts' in dist:
                survival_counts = dist['survival_counts']
                total = survival_counts['survived'] + survival_counts['failed']
                survival_rate = survival_counts['survived'] / total if total > 0 else 0
                
                print(f"📈 Data distribution:")
                print(f"  ✅ Survived: {survival_counts['survived']:,} ({survival_rate:.1%})")
                print(f"  ❌ Failed: {survival_counts['failed']:,} ({1-survival_rate:.1%})")
                
                if survival_rate > 0.95:
                    print(f"⚠️ WARNING: Very high survival rate - analysis will focus on prediction confidence")
        
        return interpretability_data, test_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def analyze_model_performance(predictions, true_labels, test_data):
    """Comprehensive model performance analysis"""
    
    print(f"\n📊 === MODEL PERFORMANCE ANALYSIS ===")
    
    # Basic metrics
    survival_rate = np.mean(true_labels)
    pred_rate = np.mean(predictions)
    accuracy = np.mean((predictions > 0.5) == true_labels)
    
    print(f"📈 Overall Performance:")
    print(f"  Actual survival rate: {survival_rate:.2%}")
    print(f"  Predicted survival rate: {pred_rate:.2%}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Prediction distribution analysis
    pred_stats = {
        'min': predictions.min(),
        'max': predictions.max(),
        'mean': predictions.mean(),
        'std': predictions.std(),
        'median': np.median(predictions)
    }
    
    print(f"\n📊 Prediction Distribution:")
    print(f"  Range: [{pred_stats['min']:.3f}, {pred_stats['max']:.3f}]")
    print(f"  Mean: {pred_stats['mean']:.3f}")
    print(f"  Median: {pred_stats['median']:.3f}")
    print(f"  Std Dev: {pred_stats['std']:.3f}")
    
    # Confidence analysis
    confident_high = np.sum(predictions > 0.8)
    confident_low = np.sum(predictions < 0.2)
    uncertain = np.sum((predictions >= 0.4) & (predictions <= 0.6))
    
    print(f"\n🎯 Prediction Confidence:")
    print(f"  High confidence (>0.8): {confident_high:,} ({confident_high/len(predictions)*100:.1f}%)")
    print(f"  Low confidence (<0.2): {confident_low:,} ({confident_low/len(predictions)*100:.1f}%)")
    print(f"  Uncertain (0.4-0.6): {uncertain:,} ({uncertain/len(predictions)*100:.1f}%)")
    
    # Classification report (if we have both classes)
    if len(np.unique(true_labels)) > 1:
        pred_classes = (predictions > 0.5).astype(int)
        print(f"\n📋 Classification Report:")
        print(classification_report(true_labels, pred_classes, target_names=['Failed', 'Survived']))
    
    return {
        'basic_metrics': {
            'survival_rate': survival_rate,
            'predicted_rate': pred_rate,
            'accuracy': accuracy
        },
        'prediction_stats': pred_stats,
        'confidence_analysis': {
            'high_confidence': confident_high,
            'low_confidence': confident_low,
            'uncertain': uncertain
        }
    }

def analyze_by_characteristics(predictions, true_labels, test_data):
    """Analyze model performance across different startup characteristics"""
    
    print(f"\n🔍 === PERFORMANCE BY STARTUP CHARACTERISTICS ===")
    
    results = {}
    
    # 1. Industry Analysis
    if 'industry' in test_data.columns:
        print(f"\n📊 Performance by Industry:")
        industry_results = []
        
        for industry in test_data['industry'].unique():
            mask = test_data['industry'] == industry
            if mask.sum() >= 10:  # At least 10 samples
                ind_accuracy = np.mean((predictions[mask] > 0.5) == true_labels[mask])
                ind_survival = np.mean(true_labels[mask])
                ind_pred_rate = np.mean(predictions[mask])
                ind_count = mask.sum()
                
                industry_results.append({
                    'industry': industry,
                    'count': ind_count,
                    'accuracy': ind_accuracy,
                    'actual_survival': ind_survival,
                    'predicted_survival': ind_pred_rate
                })
                
                print(f"  {industry:12}: {ind_count:5,} samples | "
                      f"Acc: {ind_accuracy:.2%} | "
                      f"Actual: {ind_survival:.2%} | "
                      f"Predicted: {ind_pred_rate:.2%}")
        
        results['industry'] = industry_results
    
    # 2. Funding Stage Analysis
    if 'funding_stage' in test_data.columns:
        print(f"\n💰 Performance by Funding Stage:")
        funding_results = []
        
        for stage in test_data['funding_stage'].unique():
            mask = test_data['funding_stage'] == stage
            if mask.sum() >= 10:
                stage_accuracy = np.mean((predictions[mask] > 0.5) == true_labels[mask])
                stage_survival = np.mean(true_labels[mask])
                stage_pred_rate = np.mean(predictions[mask])
                stage_count = mask.sum()
                
                funding_results.append({
                    'stage': stage,
                    'count': stage_count,
                    'accuracy': stage_accuracy,
                    'actual_survival': stage_survival,
                    'predicted_survival': stage_pred_rate
                })
                
                print(f"  {stage:12}: {stage_count:5,} samples | "
                      f"Acc: {stage_accuracy:.2%} | "
                      f"Actual: {stage_survival:.2%} | "
                      f"Predicted: {stage_pred_rate:.2%}")
        
        results['funding'] = funding_results
    
    # 3. Sequence Length Analysis
    if 'sequence_length' in test_data.columns:
        print(f"\n📏 Performance by Sequence Length:")
        
        sequence_lengths = test_data['sequence_length'].values
        unique_lengths = len(np.unique(sequence_lengths))
        
        if unique_lengths >= 3:
            try:
                # Create length bins
                test_data['length_bin'] = pd.qcut(sequence_lengths, q=3, 
                                                labels=['Short', 'Medium', 'Long'], 
                                                duplicates='drop')
                
                length_results = []
                for bin_name in ['Short', 'Medium', 'Long']:
                    if bin_name in test_data['length_bin'].values:
                        mask = test_data['length_bin'] == bin_name
                        if mask.sum() > 0:
                            bin_accuracy = np.mean((predictions[mask] > 0.5) == true_labels[mask])
                            bin_survival = np.mean(true_labels[mask])
                            bin_pred_rate = np.mean(predictions[mask])
                            bin_count = mask.sum()
                            
                            length_results.append({
                                'bin': bin_name,
                                'count': bin_count,
                                'accuracy': bin_accuracy,
                                'actual_survival': bin_survival,
                                'predicted_survival': bin_pred_rate
                            })
                            
                            print(f"  {bin_name:12}: {bin_count:5,} samples | "
                                  f"Acc: {bin_accuracy:.2%} | "
                                  f"Actual: {bin_survival:.2%} | "
                                  f"Predicted: {bin_pred_rate:.2%}")
                
                results['sequence_length'] = length_results
                
            except Exception as e:
                print(f"  ⚠️ Could not create length bins: {e}")
        else:
            print(f"  ⚠️ Only {unique_lengths} unique lengths - insufficient for binning")
    
    # 4. Company Age Analysis
    if 'company_age_at_prediction' in test_data.columns:
        print(f"\n🕐 Performance by Company Age:")
        
        ages = test_data['company_age_at_prediction'].values
        valid_ages = ages[ages > 0]  # Remove invalid ages
        
        if len(valid_ages) > 100:
            try:
                # Create age bins
                age_bins = pd.cut(valid_ages, bins=4, labels=['Very Young', 'Young', 'Mature', 'Old'])
                valid_mask = ages > 0
                
                age_results = []
                for bin_name in ['Very Young', 'Young', 'Mature', 'Old']:
                    if bin_name in age_bins.values:
                        # Map back to original indices
                        bin_mask = (age_bins == bin_name) & valid_mask[valid_mask]
                        full_mask = np.zeros(len(test_data), dtype=bool)
                        full_mask[valid_mask] = (age_bins == bin_name)
                        
                        if full_mask.sum() > 0:
                            age_accuracy = np.mean((predictions[full_mask] > 0.5) == true_labels[full_mask])
                            age_survival = np.mean(true_labels[full_mask])
                            age_pred_rate = np.mean(predictions[full_mask])
                            age_count = full_mask.sum()
                            
                            age_results.append({
                                'age_bin': bin_name,
                                'count': age_count,
                                'accuracy': age_accuracy,
                                'actual_survival': age_survival,
                                'predicted_survival': age_pred_rate
                            })
                            
                            print(f"  {bin_name:12}: {age_count:5,} samples | "
                                  f"Acc: {age_accuracy:.2%} | "
                                  f"Actual: {age_survival:.2%} | "
                                  f"Predicted: {age_pred_rate:.2%}")
                
                results['company_age'] = age_results
                
            except Exception as e:
                print(f"  ⚠️ Could not create age bins: {e}")
    
    return results

def analyze_embeddings(embeddings, predictions, true_labels, sample_size=5000):
    """Analyze startup embeddings using dimensionality reduction and clustering"""
    
    print(f"\n🧠 === EMBEDDING SPACE ANALYSIS ===")
    
    # Sample for visualization if dataset is too large
    if len(embeddings) > sample_size:
        print(f"📊 Sampling {sample_size:,} startups for embedding analysis...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        predictions_sample = predictions[indices]
        true_labels_sample = true_labels[indices]
    else:
        embeddings_sample = embeddings
        predictions_sample = predictions
        true_labels_sample = true_labels
    
    results = {}
    
    # 1. PCA Analysis
    print(f"\n📊 Principal Component Analysis:")
    pca = PCA(n_components=min(50, embeddings_sample.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_sample)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"  First 5 components explain: {explained_variance[:5].sum():.1%} of variance")
    print(f"  First 10 components explain: {explained_variance[:10].sum():.1%} of variance")
    print(f"  Components for 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
    
    results['pca'] = {
        'explained_variance': explained_variance[:20].tolist(),
        'cumulative_variance': cumulative_variance[:20].tolist()
    }
    
    # 2. Clustering Analysis
    print(f"\n🔄 Clustering Analysis:")
    
    # Use PCA embeddings for clustering (first 20 components)
    cluster_embeddings = pca_embeddings[:, :20]
    
    # Try different numbers of clusters
    best_k = 5
    silhouette_scores = []
    
    for k in range(2, min(11, len(embeddings_sample)//50)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_embeddings)
            
            # Simple within-cluster sum of squares as a proxy for silhouette score
            wcss = kmeans.inertia_
            silhouette_scores.append((k, wcss))
            
        except Exception as e:
            print(f"  ⚠️ Error with k={k}: {e}")
            continue
    
    if silhouette_scores:
        # Use elbow method (simple version)
        best_k = min(silhouette_scores, key=lambda x: x[1])[0]
        best_k = min(best_k, 8)  # Cap at 8 clusters for interpretability
    
    print(f"  Optimal number of clusters: {best_k}")
    
    # Final clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_embeddings)
    
    print(f"  Cluster distribution:")
    cluster_results = []
    for i in range(best_k):
        mask = cluster_labels == i
        if mask.sum() > 0:
            cluster_accuracy = np.mean((predictions_sample[mask] > 0.5) == true_labels_sample[mask])
            cluster_survival = np.mean(true_labels_sample[mask])
            cluster_pred_rate = np.mean(predictions_sample[mask])
            cluster_size = mask.sum()
            
            cluster_results.append({
                'cluster': i,
                'size': cluster_size,
                'accuracy': cluster_accuracy,
                'actual_survival': cluster_survival,
                'predicted_survival': cluster_pred_rate
            })
            
            print(f"    Cluster {i}: {cluster_size:4,} startups | "
                  f"Acc: {cluster_accuracy:.2%} | "
                  f"Survival: {cluster_survival:.2%}")
    
    results['clustering'] = {
        'optimal_k': best_k,
        'cluster_results': cluster_results
    }
    
    # 3. t-SNE Visualization (for smaller samples)
    if len(embeddings_sample) <= 2000:
        print(f"\n🎨 Creating t-SNE visualization...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)//4))
            tsne_embeddings = tsne.fit_transform(cluster_embeddings)
            
            # Save visualization data
            results['tsne'] = {
                'embeddings': tsne_embeddings.tolist(),
                'predictions': predictions_sample.tolist(),
                'true_labels': true_labels_sample.tolist(),
                'clusters': cluster_labels.tolist()
            }
            
            print(f"  ✅ t-SNE visualization ready")
            
        except Exception as e:
            print(f"  ⚠️ t-SNE failed: {e}")
    
    return results

def generate_insights_and_recommendations(performance_results, characteristic_results, embedding_results):
    """Generate actionable insights and recommendations"""
    
    print(f"\n💡 === INSIGHTS AND RECOMMENDATIONS ===")
    
    insights = []
    recommendations = []
    
    # 1. Performance Insights
    basic_metrics = performance_results['basic_metrics']
    survival_rate = basic_metrics['survival_rate']
    accuracy = basic_metrics['accuracy']
    
    if survival_rate > 0.95:
        insights.append(f"Dataset is heavily imbalanced ({survival_rate:.1%} survival rate)")
        recommendations.append("Consider collecting more failed startup data for balanced analysis")
        recommendations.append("Focus on prediction confidence rather than just accuracy")
    
    if accuracy > 0.95 and survival_rate > 0.95:
        insights.append("Model may be learning to always predict 'survived'")
        recommendations.append("Investigate if model is actually learning meaningful patterns")
        recommendations.append("Check training data balance and model calibration")
    
    # 2. Characteristic Insights
    if 'industry' in characteristic_results:
        industry_results = characteristic_results['industry']
        industry_accuracies = [r['accuracy'] for r in industry_results]
        
        if max(industry_accuracies) - min(industry_accuracies) > 0.05:
            best_industry = max(industry_results, key=lambda x: x['accuracy'])
            worst_industry = min(industry_results, key=lambda x: x['accuracy'])
            
            insights.append(f"Significant performance variation across industries")
            insights.append(f"Best: {best_industry['industry']} ({best_industry['accuracy']:.2%})")
            insights.append(f"Worst: {worst_industry['industry']} ({worst_industry['accuracy']:.2%})")
            
            recommendations.append("Investigate why certain industries have better predictions")
            recommendations.append("Consider industry-specific features or models")
    
    # 3. Embedding Insights
    if 'clustering' in embedding_results:
        cluster_results = embedding_results['clustering']['cluster_results']
        cluster_accuracies = [r['accuracy'] for r in cluster_results]
        
        if len(cluster_accuracies) > 1 and max(cluster_accuracies) - min(cluster_accuracies) > 0.05:
            insights.append("Startup embeddings form distinct clusters with varying performance")
            recommendations.append("Explore what characterizes high-performing clusters")
            recommendations.append("Consider cluster-aware prediction strategies")
    
    # 4. Data Quality Insights
    confidence_analysis = performance_results['confidence_analysis']
    uncertain_pct = confidence_analysis['uncertain'] / (confidence_analysis['high_confidence'] + 
                                                        confidence_analysis['low_confidence'] + 
                                                        confidence_analysis['uncertain']) * 100
    
    if uncertain_pct > 20:
        insights.append(f"High uncertainty in predictions ({uncertain_pct:.1f}% in 0.4-0.6 range)")
        recommendations.append("Investigate sources of model uncertainty")
        recommendations.append("Consider uncertainty quantification methods")
    
    # Print insights
    print(f"\n🔍 Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    print(f"\n📋 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return {'insights': insights, 'recommendations': recommendations}

def save_comprehensive_results(performance_results, characteristic_results, embedding_results, 
                             insights, output_file="interpretability_results/comprehensive_analysis.txt"):
    """Save comprehensive analysis results"""
    
    print(f"\n💾 Saving comprehensive analysis results...")
    
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE STARTUP2VEC INTERPRETABILITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Performance Summary
        f.write("📊 MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        basic_metrics = performance_results['basic_metrics']
        f.write(f"Overall Accuracy: {basic_metrics['accuracy']:.2%}\n")
        f.write(f"Survival Rate (Actual): {basic_metrics['survival_rate']:.2%}\n")
        f.write(f"Survival Rate (Predicted): {basic_metrics['predicted_rate']:.2%}\n\n")
        
        # Characteristics Analysis
        if characteristic_results:
            f.write("🔍 PERFORMANCE BY CHARACTERISTICS\n")
            f.write("-" * 40 + "\n")
            
            if 'industry' in characteristic_results:
                f.write("By Industry:\n")
                for result in characteristic_results['industry']:
                    f.write(f"  {result['industry']}: {result['accuracy']:.2%} accuracy "
                           f"({result['count']:,} samples)\n")
                f.write("\n")
            
            if 'funding' in characteristic_results:
                f.write("By Funding Stage:\n")
                for result in characteristic_results['funding']:
                    f.write(f"  {result['stage']}: {result['accuracy']:.2%} accuracy "
                           f"({result['count']:,} samples)\n")
                f.write("\n")
        
        # Embedding Analysis
        if embedding_results:
            f.write("🧠 EMBEDDING SPACE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            if 'pca' in embedding_results:
                pca_results = embedding_results['pca']
                f.write(f"PCA: First 5 components explain "
                       f"{sum(pca_results['explained_variance'][:5]):.1%} of variance\n")
            
            if 'clustering' in embedding_results:
                cluster_results = embedding_results['clustering']
                f.write(f"Clustering: {cluster_results['optimal_k']} distinct startup clusters found\n")
            
            f.write("\n")
        
        # Insights and Recommendations
        f.write("💡 KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        for i, insight in enumerate(insights['insights'], 1):
            f.write(f"{i}. {insight}\n")
        f.write("\n")
        
        f.write("📋 RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for i, rec in enumerate(insights['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")
        
        f.write("Note: This analysis is based on the complete validation dataset.\n")
        f.write("For detailed visualizations and further analysis, refer to the embedding results.\n")
    
    print(f"✅ Results saved to {output_file}")

def main():
    """Main comprehensive interpretability analysis"""
    
    print("🔍 COMPREHENSIVE STARTUP2VEC INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    print("Designed for large datasets with robust handling of class imbalance")
    
    # 1. Load data
    interpretability_data, test_data = load_extracted_data()
    
    if interpretability_data is None:
        print("❌ Could not load data")
        print("💡 Run the full extraction script first: python extract_startup2vec_data_FULL.py")
        return 1
    
    # Extract arrays
    predictions = interpretability_data['predictions']
    true_labels = interpretability_data['true_labels']
    embeddings = interpretability_data['startup_embeddings']
    
    print(f"\n🎯 Analyzing {len(predictions):,} startup predictions...")
    
    # 2. Model Performance Analysis
    performance_results = analyze_model_performance(predictions, true_labels, test_data)
    
    # 3. Characteristics Analysis
    characteristic_results = analyze_by_characteristics(predictions, true_labels, test_data)
    
    # 4. Embedding Analysis
    embedding_results = analyze_embeddings(embeddings, predictions, true_labels)
    
    # 5. Generate Insights
    insights = generate_insights_and_recommendations(
        performance_results, characteristic_results, embedding_results
    )
    
    # 6. Save Results
    save_comprehensive_results(
        performance_results, characteristic_results, embedding_results, insights
    )
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"📊 Analyzed {len(predictions):,} startups across multiple dimensions")
    print(f"📁 Results saved to interpretability_results/comprehensive_analysis.txt")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)