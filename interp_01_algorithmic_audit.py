#!/usr/bin/env python3
"""
ALGORITHMIC AUDIT SCRIPT: Evaluate model performance across subgroups (employee size, industry, country, business model, technology type, investment event presence)
Outputs metrics per subgroup as CSV and printed table.

INSTRUCTIONS:
- This script parses all subgroup characteristics directly from the input token sequence for each test sample (no DataFrame merges).
- To test, set TEST_MODE = True to run on a small sample. Set to False for full test set.
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, accuracy_score
)
from collections import defaultdict

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# === CONFIGURATION ===
CHECKPOINT_PATH = "survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"
OUTPUT_CSV = "algorithmic_audit_results.csv"
BATCH_SIZE = 1024
TEST_MODE = False  # Set to False for full test set
TEST_BATCHES = 2  # Number of batches to process in test mode

# === METRICS ===
METRICS = [
    ('AUC', roc_auc_score),
    ('Balanced Accuracy', balanced_accuracy_score),
    ('F1', f1_score),
    ('Precision', precision_score),
    ('Recall', recall_score),
    ('MCC', matthews_corrcoef),
    ('Accuracy', accuracy_score),
]

# === TOKEN PARSING LOGIC (from complete_interpretability_5.py) ===
def parse_token_categories(vocab_to_idx):
    categories = {
        'company_country': {},          
        'company_category': {},         
        'company_employee_size': {},    
        'company_industry': {},         
        'company_business_model': {},   
        'company_technology': {},       
        'event_types': {},              
        'event_categories': {},         
        'event_terms': {},              
        'event_roles': {},              
        'people_jobs': {},              
        'people_terms': {},             
        'people_job_titles': {},        
        'education_degree_type': {},    
        'education_institution': {},    
        'education_subject': {},        
        'investment_types': {},         
        'investment_investor_types': {},
        'investment_amounts': {},       
        'investment_fund_sizes': {},    
        'investment_counts': {},        
        'investment_valuations': {},    
        'investment_other': {},         
        'acquisition_types': {},        
        'acquisition_prices': {},       
        'acquisition_other': {},        
        'ipo_exchanges': {},            
        'ipo_money_raised': {},         
        'ipo_share_prices': {},         
        'ipo_valuations': {},           
        'ipo_other': {},                
        'days_since_founding': {},      
    }
    for token_str, token_id in vocab_to_idx.items():
        if token_str.startswith('COUNTRY_'):
            categories['company_country'][token_id] = token_str
        elif token_str.startswith('CATEGORY_'):
            categories['company_category'][token_id] = token_str
        elif token_str.startswith('EMPLOYEE_'):
            categories['company_employee_size'][token_id] = token_str
        elif token_str.startswith('INDUSTRY_'):
            categories['company_industry'][token_id] = token_str
        elif token_str.startswith('MODEL_'):
            categories['company_business_model'][token_id] = token_str
        elif token_str.startswith('TECH_'):
            categories['company_technology'][token_id] = token_str
        elif token_str.startswith('EVT_TYPE_'):
            categories['event_types'][token_id] = token_str
        elif token_str.startswith('EVT_CAT_'):
            categories['event_categories'][token_id] = token_str
        elif token_str.startswith('EVT_TERM_'):
            categories['event_terms'][token_id] = token_str
        elif token_str.startswith('EVENT_ROLES_'):
            categories['event_roles'][token_id] = token_str
        elif token_str.startswith('PPL_JOB_'):
            categories['people_jobs'][token_id] = token_str
        elif token_str.startswith('PPL_TERM_'):
            categories['people_terms'][token_id] = token_str
        elif token_str.startswith('PEOPLE_JOB_TITLE_'):
            categories['people_job_titles'][token_id] = token_str
        elif token_str.startswith('EDU_DEGREE_TYPE_'):
            categories['education_degree_type'][token_id] = token_str
        elif token_str.startswith('EDU_INSTITUTION_'):
            categories['education_institution'][token_id] = token_str
        elif token_str.startswith('EDU_SUBJECT_'):
            categories['education_subject'][token_id] = token_str
        elif token_str.startswith('INV_INVESTMENT_TYPE_'):
            categories['investment_types'][token_id] = token_str
        elif token_str.startswith('INV_INVESTOR_TYPES_'):
            categories['investment_investor_types'][token_id] = token_str
        elif token_str.startswith('INV_RAISED_AMOUNT_USD_'):
            categories['investment_amounts'][token_id] = token_str
        elif token_str.startswith('INV_FUND_SIZE_USD_'):
            categories['investment_fund_sizes'][token_id] = token_str
        elif token_str.startswith('INV_INVESTOR_COUNT_'):
            categories['investment_counts'][token_id] = token_str
        elif token_str.startswith('INV_POST_MONEY_VALUATION_USD_'):
            categories['investment_valuations'][token_id] = token_str
        elif token_str.startswith('INV_'):
            categories['investment_other'][token_id] = token_str
        elif token_str.startswith('ACQ_ACQUISITION_TYPE_'):
            categories['acquisition_types'][token_id] = token_str
        elif token_str.startswith('ACQ_PRICE_USD_'):
            categories['acquisition_prices'][token_id] = token_str
        elif token_str.startswith('ACQ_'):
            categories['acquisition_other'][token_id] = token_str
        elif token_str.startswith('IPO_EXCHANGE_'):
            categories['ipo_exchanges'][token_id] = token_str
        elif token_str.startswith('IPO_MONEY_RAISED_USD_'):
            categories['ipo_money_raised'][token_id] = token_str
        elif token_str.startswith('IPO_SHARE_PRICE_USD_'):
            categories['ipo_share_prices'][token_id] = token_str
        elif token_str.startswith('IPO_VALUATION_USD_'):
            categories['ipo_valuations'][token_id] = token_str
        elif token_str.startswith('IPO_'):
            categories['ipo_other'][token_id] = token_str
        elif token_str.startswith('DAYS_'):
            categories['days_since_founding'][token_id] = token_str
    return categories

def parse_sequence_characteristics(sequence, idx_to_vocab, token_categories):
    characteristics = {
        'country': 'Unknown', 'industry_category': 'Unknown', 'employee_size': 'Unknown',
        'business_model': 'Unknown', 'technology_type': 'Unknown',
        'has_investment_events': False, 'has_acquisition_events': False,
        'has_ipo_events': False, 'has_education_events': False, 'has_people_events': False,
        'investment_event_count': 0, 'people_job_count': 0, 'education_event_count': 0,
        'unique_token_count': 0, 'token_diversity': 0.0
    }
    try:
        clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
        characteristics['unique_token_count'] = len(set(clean_sequence))
        characteristics['token_diversity'] = len(set(clean_sequence)) / len(clean_sequence) if len(clean_sequence) > 0 else 0
        for token_id in clean_sequence:
            token_str = idx_to_vocab.get(int(token_id), "")
            if token_str.startswith('COUNTRY_'):
                characteristics['country'] = token_str.replace('COUNTRY_', '')
            elif token_str.startswith('CATEGORY_') or token_str.startswith('INDUSTRY_'):
                characteristics['industry_category'] = token_str.split('_', 1)[1] if '_' in token_str else 'Unknown'
            elif token_str.startswith('EMPLOYEE_'):
                characteristics['employee_size'] = token_str.replace('EMPLOYEE_', '')
            elif token_str.startswith('MODEL_'):
                characteristics['business_model'] = token_str.replace('MODEL_', '')
            elif token_str.startswith('TECH_'):
                characteristics['technology_type'] = token_str.replace('TECH_', '')
            if int(token_id) in token_categories['investment_types'] or int(token_id) in token_categories['investment_amounts']:
                characteristics['has_investment_events'] = True
                characteristics['investment_event_count'] += 1
            elif int(token_id) in token_categories['acquisition_types']:
                characteristics['has_acquisition_events'] = True
            elif int(token_id) in token_categories['ipo_exchanges'] or int(token_id) in token_categories['ipo_money_raised']:
                characteristics['has_ipo_events'] = True
            elif (int(token_id) in token_categories['education_degree_type'] or 
                  int(token_id) in token_categories['education_institution'] or 
                  int(token_id) in token_categories['education_subject']):
                characteristics['has_education_events'] = True
                characteristics['education_event_count'] += 1
            elif int(token_id) in token_categories['people_jobs'] or int(token_id) in token_categories['people_job_titles']:
                characteristics['has_people_events'] = True
                characteristics['people_job_count'] += 1
    except Exception as e:
        print(f"Warning: Could not parse sequence characteristics: {e}")
    return characteristics

# === SUBGROUP DEFINITIONS (use parsed fields) ===
SUBGROUPS = {
    'employee_size': lambda meta: meta['employee_size'],
    'industry': lambda meta: meta['industry_category'],
    'country': lambda meta: meta['country'],
    'business_model': lambda meta: meta['business_model'],
    'technology_type': lambda meta: meta['technology_type'],
    'investment_event': lambda meta: meta['has_investment_events'],
}

# === MAIN SCRIPT ===
def load_model_and_data():
    from models.survival_model import FixedStartupSurvivalModel
    from dataloaders.survival_datamodule import SurvivalDataModule
    print("Loading model from:", CHECKPOINT_PATH)
    model = FixedStartupSurvivalModel.load_from_checkpoint(CHECKPOINT_PATH, map_location='cpu')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    datamodule = SurvivalDataModule(
        corpus_name="startup_corpus",
        vocab_name="startup_vocab",
        batch_size=BATCH_SIZE,
        num_workers=4,
        prediction_windows=[1, 2, 3, 4]
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    # Get vocab
    vocab_to_idx = datamodule.vocabulary.token2index
    idx_to_vocab = datamodule.vocabulary.index2token
    return model, test_loader, device, vocab_to_idx, idx_to_vocab

def get_predictions_and_metadata(model, test_loader, device, idx_to_vocab, vocab_to_idx, test_mode=TEST_MODE, test_batches=TEST_BATCHES):
    token_categories = parse_token_categories(vocab_to_idx)
    all_logits = []
    all_labels = []
    all_metadata = []
    n_batches = test_batches if test_mode else None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['survival_label'].squeeze().to(device)
            logits = model(input_ids=input_ids, padding_mask=padding_mask)['survival_logits']
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            # Parse characteristics for each sample in batch
            for j in range(input_ids.size(0)):
                meta = parse_sequence_characteristics(input_ids[j, 0, :], idx_to_vocab, token_categories)
                all_metadata.append(meta)
            if test_mode and i + 1 >= test_batches:
                break
            if i % 10 == 0:
                print(f"Processed batch {i}")
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # Probability of survival
    preds = (probs > 0.5).astype(int)
    return preds, probs, labels.numpy(), all_metadata

def audit_by_subgroup(metadata, preds, probs, labels):
    results = []
    meta_df = pd.DataFrame(metadata)
    for subgroup_name, subgroup_fn in SUBGROUPS.items():
        print(f"\n=== Auditing by {subgroup_name} ===")
        groups = meta_df.groupby(meta_df.apply(subgroup_fn, axis=1))
        for group_value, group_df in groups:
            idx = group_df.index.values
            if len(idx) < 10:
                continue  # Skip tiny groups
            group_labels = labels[idx]
            group_preds = preds[idx]
            group_probs = probs[idx]
            safe_metrics = {}
            for metric_name, metric_fn in METRICS:
                try:
                    if metric_name == 'AUC':
                        if len(np.unique(group_labels)) < 2:
                            print(f"Warning: Skipping AUC for {subgroup_name}={group_value} (only one class present)")
                            safe_metrics[metric_name] = np.nan
                        else:
                            safe_metrics[metric_name] = metric_fn(group_labels, group_probs)
                    else:
                        safe_metrics[metric_name] = metric_fn(group_labels, group_preds)
                except Exception:
                    safe_metrics[metric_name] = np.nan
            pred_survival_rate = group_preds.mean()
            actual_survival_rate = group_labels.mean()
            results.append({
                'Subgroup': subgroup_name,
                'Group': group_value,
                'N': len(idx),
                **safe_metrics,
                'Predicted Survival Rate': pred_survival_rate,
                'Actual Survival Rate': actual_survival_rate,
            })
    return results

def print_results_table(results):
    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    print("\n=== Algorithmic Audit Results ===")
    print(df.to_string(index=False, float_format="{:.3f}".format))

def save_results_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    print("Loading model and data...")
    model, test_loader, device, vocab_to_idx, idx_to_vocab = load_model_and_data()
    print("Model and data loaded. Getting predictions and metadata...")
    preds, probs, labels, metadata = get_predictions_and_metadata(model, test_loader, device, idx_to_vocab, vocab_to_idx, TEST_MODE, TEST_BATCHES)
    print(f"Test set size: {len(labels)} samples")
    # Print class counts and probability stats
    print("Label counts (ground truth):", np.bincount(labels.astype(int)))
    print("Prediction counts:", np.bincount(preds.astype(int)))
    print("Mean predicted survival probability:", np.mean(probs))
    print("Std of predicted survival probability:", np.std(probs))
    print("Predictions and metadata obtained. Auditing by subgroup...")
    results = audit_by_subgroup(metadata, preds, probs, labels)
    print("Audit complete. Printing results table...")
    print_results_table(results)
    print("Saving results to CSV...")
    save_results_csv(results, OUTPUT_CSV)
    print("Done.")

if __name__ == "__main__":
    main()