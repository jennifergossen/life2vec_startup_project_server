# class_imbalance_analysis.py
"""
ADVANCED CLASS IMBALANCE ANALYSIS & INTERPRETABILITY
Investigates why the model learned to always predict "survived" despite class weights
Provides sophisticated interpretability analysis for imbalanced datasets
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def analyze_class_imbalance_issues(interpretability_data, test_data):
    """Analyze why model failed despite class weights"""
    
    print("🔍 ADVANCED CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    
    predictions = interpretability_data['predictions']
    true_labels = interpretability_data['true_labels']
    
    # 1. Check actual class distribution
    print("\n📊 CLASS DISTRIBUTION ANALYSIS:")
    total_samples = len(true_labels)
    survived_count = np.sum(true_labels == 1)
    died_count = np.sum(true_labels == 0)
    
    print(f"  Total samples: {total_samples:,}")
    print(f"  Died (0): {died_count:,} ({died_count/total_samples*100:.2f}%)")
    print(f"  Survived (1): {survived_count:,} ({survived_count/total_samples*100:.2f}%)")
    print(f"  Imbalance ratio: {survived_count/died_count:.1f}:1")
    
    # 2. Prediction distribution analysis
    print("\n📈 PREDICTION DISTRIBUTION:")
    print(f"  Min prediction: {predictions.min():.4f}")
    print(f"  Max prediction: {predictions.max():.4f}")
    print(f"  Mean prediction: {predictions.mean():.4f}")
    print(f"  Std prediction: {predictions.std():.4f}")
    
    # Check if predictions are actually varied
    unique_preds = len(np.unique(np.round(predictions, 3)))
    print(f"  Unique predictions (rounded): {unique_preds}")
    
    if unique_preds < 10:
        print("  ⚠️ Very few unique predictions - model may be collapsed")
    
    # 3. Performance metrics that handle imbalance
    print("\n🎯 IMBALANCE-AWARE METRICS:")
    
    # AUC-ROC (works well with imbalance)
    try:
        auc_roc = roc_auc_score(true_labels, predictions)
        print(f"  AUC-ROC: {auc_roc:.4f}")
        
        if auc_roc < 0.6:
            print("    ⚠️ Poor discrimination ability")
        elif auc_roc < 0.7:
            print("    📊 Moderate discrimination")
        else:
            print("    ✅ Good discrimination")
            
    except Exception as e:
        print(f"  ❌ Could not calculate AUC-ROC: {e}")
        auc_roc = None
    
    # Precision-Recall AUC (better for imbalanced data)
    try:
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = np.trapz(precision, recall)
        print(f"  PR-AUC: {pr_auc:.4f}")
        
        # Random baseline for PR-AUC
        baseline_pr = survived_count / total_samples
        print(f"  PR-AUC baseline: {baseline_pr:.4f}")
        
        if pr_auc > baseline_pr + 0.05:
            print("    ✅ Better than random baseline")
        else:
            print("    ⚠️ Close to random performance")
            
    except Exception as e:
        print(f"  ❌ Could not calculate PR-AUC: {e}")
        pr_auc = None
    
    # 4. Calibration analysis
    print("\n🎯 CALIBRATION ANALYSIS:")
    try:
        prob_true, prob_pred = calibration_curve(true_labels, predictions, n_bins=10)
        
        print(f"  Calibration bins:")
        for i, (true_prob, pred_prob) in enumerate(zip(prob_true, prob_pred)):
            print(f"    Bin {i}: Predicted {pred_prob:.3f}, Actual {true_prob:.3f}")
        
        # Perfect calibration would have prob_true ≈ prob_pred
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"  Mean calibration error: {calibration_error:.4f}")
        
        if calibration_error < 0.05:
            print("    ✅ Well calibrated")
        elif calibration_error < 0.1:
            print("    📊 Moderately calibrated")
        else:
            print("    ⚠️ Poorly calibrated")
            
    except Exception as e:
        print(f"  ❌ Could not analyze calibration: {e}")
    
    # 5. Threshold analysis
    print("\n🎚️ THRESHOLD OPTIMIZATION:")
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_f1 = 0
    
    print(f"  Threshold analysis:")
    for threshold in thresholds:
        pred_classes = (predictions > threshold).astype(int)
        
        # Calculate F1 for minority class (died)
        tp = np.sum((pred_classes == 0) & (true_labels == 0))
        fp = np.sum((pred_classes == 0) & (true_labels == 1))
        fn = np.sum((pred_classes == 1) & (true_labels == 0))
        
        precision_died = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_died = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_died = 2 * precision_died * recall_died / (precision_died + recall_died) if (precision_died + recall_died) > 0 else 0
        
        print(f"    {threshold:.1f}: F1={f1_died:.3f}, P={precision_died:.3f}, R={recall_died:.3f}")
        
        if f1_died > best_f1:
            best_f1 = f1_died
            best_threshold = threshold
    
    print(f"  Best threshold: {best_threshold:.1f} (F1={best_f1:.3f})")
    
    return {
        'class_distribution': {'died': died_count, 'survived': survived_count, 'ratio': survived_count/died_count},
        'prediction_stats': {'min': predictions.min(), 'max': predictions.max(), 'mean': predictions.mean(), 'std': predictions.std()},
        'metrics': {'auc_roc': auc_roc, 'pr_auc': pr_auc},
        'best_threshold': best_threshold,
        'best_f1': best_f1
    }

def analyze_minority_class_patterns(interpretability_data, test_data):
    """Focus analysis on the minority class (failed startups)"""
    
    print("\n🔍 MINORITY CLASS (FAILED STARTUPS) ANALYSIS")
    print("=" * 60)
    
    predictions = interpretability_data['predictions']
    true_labels = interpretability_data['true_labels']
    embeddings = interpretability_data['startup_embeddings']
    
    # Find failed startups
    failed_mask = true_labels == 0
    failed_count = np.sum(failed_mask)
    
    if failed_count == 0:
        print("❌ No failed startups in dataset!")
        return None
    
    print(f"📊 Analyzing {failed_count:,} failed startups...")
    
    # 1. Prediction distribution for failed startups
    failed_predictions = predictions[failed_mask]
    survived_predictions = predictions[~failed_mask]
    
    print(f"\n📈 PREDICTION PATTERNS:")
    print(f"  Failed startups - predictions:")
    print(f"    Mean: {failed_predictions.mean():.4f}")
    print(f"    Std: {failed_predictions.std():.4f}")
    print(f"    Range: [{failed_predictions.min():.4f}, {failed_predictions.max():.4f}]")
    
    print(f"  Survived startups - predictions:")
    print(f"    Mean: {survived_predictions.mean():.4f}")
    print(f"    Std: {survived_predictions.std():.4f}")
    
    # Check if model can distinguish at all
    separation = survived_predictions.mean() - failed_predictions.mean()
    print(f"  Separation: {separation:.4f}")
    
    if separation > 0.01:
        print("    ✅ Model shows some discrimination")
    else:
        print("    ⚠️ Model shows little discrimination")
    
    # 2. Find well-predicted failed startups
    correctly_predicted_failed = failed_predictions < 0.5
    well_predicted_count = np.sum(correctly_predicted_failed)
    
    print(f"\n🎯 WELL-PREDICTED FAILED STARTUPS:")
    print(f"  Count: {well_predicted_count} ({well_predicted_count/failed_count*100:.1f}%)")
    
    if well_predicted_count > 0:
        well_predicted_probs = failed_predictions[correctly_predicted_failed]
        print(f"  Their prediction range: [{well_predicted_probs.min():.4f}, {well_predicted_probs.max():.4f}]")
        print(f"  Mean prediction: {well_predicted_probs.mean():.4f}")
    
    # 3. Characteristics of failed vs survived startups
    print(f"\n📊 CHARACTERISTICS COMPARISON:")
    
    # Sequence length analysis
    if 'sequence_length' in test_data.columns:
        failed_seq_lengths = test_data.loc[failed_mask, 'sequence_length']
        survived_seq_lengths = test_data.loc[~failed_mask, 'sequence_length']
        
        print(f"  Sequence lengths:")
        print(f"    Failed: {failed_seq_lengths.mean():.1f} ± {failed_seq_lengths.std():.1f}")
        print(f"    Survived: {survived_seq_lengths.mean():.1f} ± {survived_seq_lengths.std():.1f}")
        
        if abs(failed_seq_lengths.mean() - survived_seq_lengths.mean()) > 5:
            print(f"    📊 Significant difference detected")
    
    # Company age analysis
    if 'company_age_at_prediction' in test_data.columns:
        failed_ages = test_data.loc[failed_mask, 'company_age_at_prediction']
        survived_ages = test_data.loc[~failed_mask, 'company_age_at_prediction']
        
        # Filter valid ages
        valid_failed_ages = failed_ages[failed_ages > 0]
        valid_survived_ages = survived_ages[survived_ages > 0]
        
        if len(valid_failed_ages) > 0 and len(valid_survived_ages) > 0:
            print(f"  Company ages:")
            print(f"    Failed: {valid_failed_ages.mean():.1f} ± {valid_failed_ages.std():.1f}")
            print(f"    Survived: {valid_survived_ages.mean():.1f} ± {valid_survived_ages.std():.1f}")
    
    # 4. Embedding analysis for failed startups
    print(f"\n🧠 EMBEDDING ANALYSIS:")
    
    try:
        # PCA to see if failed startups cluster differently
        pca = PCA(n_components=2)
        
        # Sample for visualization if too large
        if len(embeddings) > 10000:
            sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = true_labels[sample_indices]
            sample_failed_mask = sample_labels == 0
        else:
            sample_embeddings = embeddings
            sample_failed_mask = failed_mask
        
        pca_embeddings = pca.fit_transform(sample_embeddings)
        
        # Check if failed startups are clustered
        failed_pca = pca_embeddings[sample_failed_mask]
        survived_pca = pca_embeddings[~sample_failed_mask]
        
        print(f"  PCA analysis:")
        print(f"    Failed startups PC1: {failed_pca[:, 0].mean():.3f} ± {failed_pca[:, 0].std():.3f}")
        print(f"    Survived startups PC1: {survived_pca[:, 0].mean():.3f} ± {survived_pca[:, 0].std():.3f}")
        
        # Calculate separation in embedding space
        failed_center = np.mean(failed_pca, axis=0)
        survived_center = np.mean(survived_pca, axis=0)
        embedding_separation = np.linalg.norm(failed_center - survived_center)
        
        print(f"    Embedding separation: {embedding_separation:.3f}")
        
        if embedding_separation > 0.1:
            print(f"    ✅ Failed startups form distinct cluster")
        else:
            print(f"    ⚠️ Failed startups not well separated")
        
    except Exception as e:
        print(f"  ❌ Embedding analysis failed: {e}")
    
    return {
        'failed_count': failed_count,
        'separation': separation,
        'well_predicted': well_predicted_count,
        'embedding_separation': embedding_separation if 'embedding_separation' in locals() else 0
    }

def identify_model_insights_despite_imbalance(interpretability_data, test_data):
    """Extract meaningful insights despite class imbalance"""
    
    print("\n�� EXTRACTING INSIGHTS DESPITE CLASS IMBALANCE")
    print("=" * 60)
    
    predictions = interpretability_data['predictions']
    true_labels = interpretability_data['true_labels']
    embeddings = interpretability_data['startup_embeddings']
    
    insights = []
    
    # 1. Prediction confidence analysis
    print(f"\n🔍 PREDICTION CONFIDENCE INSIGHTS:")
    
    # Find uncertain predictions (these are most interesting)
    uncertain_mask = (predictions >= 0.4) & (predictions <= 0.6)
    uncertain_count = np.sum(uncertain_mask)
    
    print(f"  Uncertain predictions (0.4-0.6): {uncertain_count:,}")
    
    if uncertain_count > 0:
        uncertain_accuracy = np.mean(true_labels[uncertain_mask] == (predictions[uncertain_mask] > 0.5))
        print(f"  Accuracy on uncertain cases: {uncertain_accuracy:.2%}")
        
        # What makes predictions uncertain?
        if 'sequence_length' in test_data.columns:
            uncertain_seq_len = test_data.loc[uncertain_mask, 'sequence_length'].mean()
            all_seq_len = test_data['sequence_length'].mean()
            print(f"  Uncertain cases avg sequence length: {uncertain_seq_len:.1f} vs {all_seq_len:.1f}")
        
        insights.append(f"Model is uncertain about {uncertain_count:,} startups ({uncertain_count/len(predictions)*100:.1f}%)")
    
    # 2. Sequence length insights
    if 'sequence_length' in test_data.columns:
        print(f"\n📏 SEQUENCE LENGTH INSIGHTS:")
        
        # Bin by sequence length
        seq_lengths = test_data['sequence_length'].values
        
        # Create meaningful bins
        short_mask = seq_lengths <= np.percentile(seq_lengths, 33)
        long_mask = seq_lengths >= np.percentile(seq_lengths, 67)
        medium_mask = ~(short_mask | long_mask)
        
        for mask, name in [(short_mask, 'Short'), (medium_mask, 'Medium'), (long_mask, 'Long')]:
            if np.sum(mask) > 0:
                group_accuracy = np.mean((predictions[mask] > 0.5) == true_labels[mask])
                group_survival_rate = np.mean(true_labels[mask])
                group_pred_rate = np.mean(predictions[mask])
                group_confidence = np.mean(np.abs(predictions[mask] - 0.5))
                
                print(f"  {name} sequences ({np.sum(mask):,} startups):")
                print(f"    Accuracy: {group_accuracy:.2%}")
                print(f"    Actual survival: {group_survival_rate:.2%}")
                print(f"    Predicted survival: {group_pred_rate:.2%}")
                print(f"    Avg confidence: {group_confidence:.3f}")
                
                if group_confidence > 0.4:
                    insights.append(f"{name} sequences show high model confidence")
                
                if abs(group_survival_rate - group_pred_rate) > 0.05:
                    insights.append(f"{name} sequences show prediction bias")
    
    # 3. Embedding clusters insights
    print(f"\n🧠 EMBEDDING CLUSTERS INSIGHTS:")
    
    try:
        from sklearn.cluster import KMeans
        
        # Sample for clustering if dataset is large
        if len(embeddings) > 50000:
            sample_indices = np.random.choice(len(embeddings), 50000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_predictions = predictions[sample_indices]
            sample_labels = true_labels[sample_indices]
        else:
            sample_embeddings = embeddings
            sample_predictions = predictions
            sample_labels = true_labels
        
        # Use PCA first for efficiency
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(sample_embeddings)
        
        # Cluster into meaningful groups
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pca_embeddings)
        
        # Analyze each cluster
        interesting_clusters = []
        
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 100:  # Only analyze substantial clusters
                cluster_accuracy = np.mean((sample_predictions[cluster_mask] > 0.5) == sample_labels[cluster_mask])
                cluster_survival = np.mean(sample_labels[cluster_mask])
                cluster_confidence = np.mean(np.abs(sample_predictions[cluster_mask] - 0.5))
                cluster_pred_rate = np.mean(sample_predictions[cluster_mask])
                
                print(f"  Cluster {i} ({cluster_size:,} startups):")
                print(f"    Accuracy: {cluster_accuracy:.2%}")
                print(f"    Survival rate: {cluster_survival:.2%}")
                print(f"    Predicted rate: {cluster_pred_rate:.2%}")
                print(f"    Confidence: {cluster_confidence:.3f}")
                
                # Flag interesting clusters
                if cluster_survival < 0.9:  # Lower survival rate
                    interesting_clusters.append(i)
                    insights.append(f"Cluster {i} has lower survival rate ({cluster_survival:.1%})")
                
                if cluster_confidence < 0.3:  # Lower confidence
                    interesting_clusters.append(i)
                    insights.append(f"Cluster {i} shows low model confidence")
                
                if abs(cluster_survival - cluster_pred_rate) > 0.1:  # Prediction bias
                    insights.append(f"Cluster {i} shows prediction bias")
        
        print(f"  Interesting clusters for further analysis: {interesting_clusters}")
        
    except Exception as e:
        print(f"  ❌ Clustering analysis failed: {e}")
    
    # 4. Summary insights
    print(f"\n💡 KEY INSIGHTS SUMMARY:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    if not insights:
        insights.append("Model shows limited discrimination despite class imbalance handling")
        insights.append("Consider examining training data quality and preprocessing")
    
    return insights

def generate_actionable_recommendations(analysis_results, minority_results, insights):
    """Generate actionable recommendations for improving the model"""
    
    print(f"\n📋 ACTIONABLE RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    # Based on class imbalance analysis
    if analysis_results['metrics']['auc_roc'] and analysis_results['metrics']['auc_roc'] < 0.7:
        recommendations.append("Model discrimination is poor - consider different architectures or features")
    
    if analysis_results['best_f1'] < 0.1:
        recommendations.append("Extremely poor minority class performance - investigate data quality")
    
    # Based on minority class analysis
    if minority_results and minority_results['separation'] < 0.01:
        recommendations.append("Model cannot distinguish failed startups - check if survival labels are correct")
    
    if minority_results and minority_results['well_predicted'] < minority_results['failed_count'] * 0.1:
        recommendations.append("Less than 10% of failed startups predicted correctly - consider focal loss or SMOTE")
    
    # General recommendations
    recommendations.extend([
        "Focus interpretability on prediction confidence rather than binary predictions",
        "Analyze the uncertain predictions (0.4-0.6 range) - these may reveal model insights",
        "Consider ensemble methods or threshold optimization for better minority class recall",
        "Investigate if temporal bias affects the dataset (successful companies stay in dataset longer)",
        "Use precision-recall curves instead of accuracy for evaluation",
        "Consider contrastive learning to better separate failed from successful startups"
    ])
    
    print(f"📝 PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:8], 1):  # Top 8 recommendations
        print(f"  {i}. {rec}")
    
    return recommendations

def main():
    """Main analysis function"""
    
    print("🔍 ADVANCED CLASS IMBALANCE & INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # Load data
    try:
        with open('interpretability_results/interpretability_data.pkl', 'rb') as f:
            interpretability_data = pickle.load(f)
        
        test_data = pd.read_pickle('interpretability_results/test_data_with_metadata.pkl')
        
        print("✅ Data loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Run comprehensive analysis
    analysis_results = analyze_class_imbalance_issues(interpretability_data, test_data)
    minority_results = analyze_minority_class_patterns(interpretability_data, test_data)
    insights = identify_model_insights_despite_imbalance(interpretability_data, test_data)
    recommendations = generate_actionable_recommendations(analysis_results, minority_results, insights)
    
    # Save results
    results = {
        'class_imbalance_analysis': analysis_results,
        'minority_class_analysis': minority_results,
        'insights': insights,
        'recommendations': recommendations
    }
    
    with open('interpretability_results/advanced_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Advanced analysis results saved to interpretability_results/advanced_analysis.pkl")
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Despite class imbalance, extracted {len(insights)} key insights")
    print(f"📋 Generated {len(recommendations)} actionable recommendations")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
