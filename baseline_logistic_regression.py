#!/usr/bin/env python3
"""
Logistic Regression Baseline for Startup Survival Prediction
Uses bag-of-words representation of event sequences.
Uses the same balanced train split and test evaluation as the main model.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def load_balanced_train_ids():
    """Load the balanced train company IDs"""
    try:
        # Try to load from saved file
        with open("balanced_train_ids.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ balanced_train_ids.pkl not found!")
        print("   Run 'python save_balanced_train_ids.py' first")
        return None

def load_company_data():
    """Load company data and create survival labels"""
    company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.csv")
    if not company_path.exists():
        company_path = Path("data/cleaned/cleaned_startup/company_base_cleaned.pkl")
        company_df = pd.read_pickle(company_path)
    else:
        company_df = pd.read_csv(company_path, low_memory=False)
    
    # Create survival labels (same logic as data module)
    company_df = company_df[['COMPANY_ID', 'status', 'founded_on', 'closed_on']].copy()
    company_df['founded_on'] = pd.to_datetime(company_df['founded_on'], errors='coerce')
    company_df['closed_on'] = pd.to_datetime(company_df.get('closed_on'), errors='coerce')
    
    founded_valid_mask = company_df['founded_on'].notna()
    survived_mask = (
        company_df['status'].isin(['operating', 'acquired', 'ipo']) & founded_valid_mask
    )
    died_mask = (
        (company_df['status'] == 'closed') & (company_df['closed_on'].notna()) & founded_valid_mask
    )
    
    company_df['survival_label'] = None
    company_df.loc[survived_mask, 'survival_label'] = 1
    company_df.loc[died_mask, 'survival_label'] = 0
    
    valid_mask = company_df['survival_label'].notna()
    company_df = company_df.loc[valid_mask, ['COMPANY_ID', 'survival_label']]
    
    return company_df

def load_data_split():
    """Load the train/val/test split"""
    split_path = Path("data/processed/populations/startups/data_split/result.pkl")
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    return split

def load_event_sequences():
    """Load event sequences for all companies"""
    # Load the processed event sequences
    sequences_path = Path("data/processed/populations/startups/sequences.pkl")
    if sequences_path.exists():
        with open(sequences_path, "rb") as f:
            sequences = pickle.load(f)
        return sequences
    
    # Fallback: try to load from other locations
    print("⚠️  sequences.pkl not found, trying alternative loading...")
    
    # You may need to implement alternative loading based on your data structure
    # For now, we'll create a simple example
    print("❌ Please ensure event sequences are available")
    return None

def create_bow_features(sequences, company_ids):
    """Create bag-of-words features from event sequences"""
    print("Creating bag-of-words features...")
    
    # Convert token sequences to text for CountVectorizer
    texts = []
    for company_id in company_ids:
        if company_id in sequences:
            # Convert token IDs to text representation
            seq = sequences[company_id]
            # Convert to space-separated string
            text = " ".join([str(token) for token in seq])
            texts.append(text)
        else:
            texts.append("")  # Empty sequence
    
    # Create bag-of-words features
    vectorizer = CountVectorizer(
        max_features=1000,  # Limit features to prevent overfitting
        min_df=2,           # Minimum document frequency
        max_df=0.95         # Maximum document frequency
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Created {X.shape[1]} features from {len(texts)} companies")
    return X, vectorizer

def main():
    """Main function"""
    print("🎯 LOGISTIC REGRESSION BASELINE")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    company_df = load_company_data()
    split = load_data_split()
    balanced_train_ids = load_balanced_train_ids()
    
    if balanced_train_ids is None:
        return
    
    # Get test companies (unbalanced, as in main model)
    test_ids = set(split.test)
    test_companies = company_df[company_df['COMPANY_ID'].isin(test_ids)]
    
    print(f"Balanced train companies: {len(balanced_train_ids)}")
    print(f"Test companies: {len(test_companies)}")
    
    # Load event sequences
    sequences = load_event_sequences()
    if sequences is None:
        print("❌ Could not load event sequences")
        return
    
    # Create features
    print("Creating features...")
    X_train, vectorizer = create_bow_features(sequences, balanced_train_ids)
    X_test, _ = create_bow_features(sequences, test_companies['COMPANY_ID'].tolist())
    
    # Get labels
    train_labels = []
    for company_id in balanced_train_ids:
        label = company_df[company_df['COMPANY_ID'] == company_id]['survival_label'].iloc[0]
        train_labels.append(label)
    
    test_labels = test_companies['survival_label'].values
    
    print(f"Train features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    print(f"Train labels: {np.bincount(train_labels)}")
    print(f"Test labels: {np.bincount(test_labels)}")
    
    # Train logistic regression
    print("\nTraining logistic regression...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    lr_model.fit(X_train, train_labels)
    
    # Predictions
    train_pred = lr_model.predict(X_train)
    test_pred = lr_model.predict(X_test)
    test_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("\n=== LOGISTIC REGRESSION BASELINE RESULTS ===")
    print(f"Test set size: {len(test_companies)}")
    print(f"Accuracy: {accuracy_score(test_labels, test_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(test_labels, test_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(test_labels, test_proba):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    print(f"Confusion Matrix (rows: true, cols: pred):")
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'coefficient': lr_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save results
    results = {
        'accuracy': accuracy_score(test_labels, test_pred),
        'balanced_accuracy': balanced_accuracy_score(test_labels, test_pred),
        'roc_auc': roc_auc_score(test_labels, test_proba),
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }
    
    with open("logistic_regression_baseline_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Results saved to logistic_regression_baseline_results.pkl")

if __name__ == "__main__":
    main() 