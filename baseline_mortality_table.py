#!/usr/bin/env python3
"""
Mortality/Survival Table Baseline for Startup Survival Prediction
- Uses the exact same train/val/test split as the main model
- Computes the overall survival rate in the training set
- Predicts this rate for all test companies
- Evaluates accuracy, balanced accuracy, ROC-AUC, and confusion matrix
"""
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

# --- 1. Load company metadata ---
company_path = "data/cleaned/cleaned_startup/company_base_cleaned.csv"
company_df = pd.read_csv(company_path, low_memory=False)
company_df = company_df.set_index('COMPANY_ID')

# --- 1b. Create survival_label column (same logic as pipeline) ---
company_df['founded_on'] = pd.to_datetime(company_df['founded_on'], errors='coerce')
company_df['closed_on'] = pd.to_datetime(company_df.get('closed_on'), errors='coerce')
company_df['survival_label'] = np.nan
founded_valid_mask = company_df['founded_on'].notna()
survived_mask = (
    company_df['status'].isin(['operating', 'acquired', 'ipo']) & founded_valid_mask
)
died_mask = (
    (company_df['status'] == 'closed') & (company_df['closed_on'].notna()) & founded_valid_mask
)
company_df.loc[survived_mask, 'survival_label'] = 1
company_df.loc[died_mask, 'survival_label'] = 0

# --- 2. Load train/val/test split ---
sys.path.append('src')  # Needed for unpickling DataSplit
split_path = "data/processed/populations/startups/data_split/result.pkl"
with open(split_path, "rb") as f:
    split = pickle.load(f)
train_ids = set(split.train)
val_ids = set(split.val)
test_ids = set(split.test)

# --- 3. Get survival labels for each split ---
train_df = company_df.loc[company_df.index.intersection(train_ids)]
test_df = company_df.loc[company_df.index.intersection(test_ids)]

# Only keep companies with valid survival labels
train_df = train_df[train_df['survival_label'].notna()]
test_df = test_df[test_df['survival_label'].notna()]

# --- 4. Compute overall survival rate in training set ---
train_survival_rate = train_df['survival_label'].mean()
print(f"Training set survival rate: {train_survival_rate:.3f}")

# --- 5. Predict for test set ---
test_true = test_df['survival_label'].astype(int).values
# Predict the same probability for all
test_pred_prob = np.full_like(test_true, fill_value=train_survival_rate, dtype=float)
# Binary prediction with threshold 0.5
test_pred = (test_pred_prob >= 0.5).astype(int)

# --- 6. Evaluate ---
acc = accuracy_score(test_true, test_pred)
bal_acc = balanced_accuracy_score(test_true, test_pred)
try:
    roc_auc = roc_auc_score(test_true, test_pred_prob)
except Exception:
    roc_auc = float('nan')
cm = confusion_matrix(test_true, test_pred)

print("\n=== Mortality Table Baseline Results ===")
print(f"Test set size: {len(test_true)}")
print(f"Accuracy: {acc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("Confusion Matrix (rows: true, cols: pred):")
print(cm) 