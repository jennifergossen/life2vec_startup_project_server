#!/usr/bin/env python3
"""
DATA CONTRIBUTION ANALYSIS: Evaluate how different event types contribute to model predictions.
Compatible with the current balanced-finetuned model and test set.
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# === CONFIGURATION ===
CHECKPOINT_PATH = "survival_checkpoints_FIXED/finetune-v2/best-epoch=03-val/balanced_acc=0.6041.ckpt"
OUTPUT_DIR = "data_contribution_results"
BATCH_SIZE = 1024
TEST_MODE = False  # Set to False for full test set
TEST_BATCHES = 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === TOKEN PARSING LOGIC (same as interp_01_algorithmic_audit.py) ===
def parse_token_categories(vocab_to_idx):
    categories = {
        'investment_types': {},         
        'investment_amounts': {},       
        'investment_fund_sizes': {},    
        'investment_counts': {},        
        'investment_valuations': {},    
        'acquisition_types': {},        
        'acquisition_prices': {},       
        'ipo_exchanges': {},            
        'ipo_money_raised': {},         
        'ipo_share_prices': {},         
        'ipo_valuations': {},           
        'education_degree_type': {},    
        'education_institution': {},    
        'education_subject': {},        
        'people_jobs': {},              
        'people_job_titles': {},        
    }
    for token_str, token_id in vocab_to_idx.items():
        if token_str.startswith('INV_INVESTMENT_TYPE_'):
            categories['investment_types'][token_id] = token_str
        elif token_str.startswith('INV_RAISED_AMOUNT_USD_'):
            categories['investment_amounts'][token_id] = token_str
        elif token_str.startswith('INV_FUND_SIZE_USD_'):
            categories['investment_fund_sizes'][token_id] = token_str
        elif token_str.startswith('INV_INVESTOR_COUNT_'):
            categories['investment_counts'][token_id] = token_str
        elif token_str.startswith('INV_POST_MONEY_VALUATION_USD_'):
            categories['investment_valuations'][token_id] = token_str
        elif token_str.startswith('ACQ_ACQUISITION_TYPE_'):
            categories['acquisition_types'][token_id] = token_str
        elif token_str.startswith('ACQ_PRICE_USD_'):
            categories['acquisition_prices'][token_id] = token_str
        elif token_str.startswith('IPO_EXCHANGE_'):
            categories['ipo_exchanges'][token_id] = token_str
        elif token_str.startswith('IPO_MONEY_RAISED_USD_'):
            categories['ipo_money_raised'][token_id] = token_str
        elif token_str.startswith('IPO_SHARE_PRICE_USD_'):
            categories['ipo_share_prices'][token_id] = token_str
        elif token_str.startswith('IPO_VALUATION_USD_'):
            categories['ipo_valuations'][token_id] = token_str
        elif token_str.startswith('EDU_DEGREE_TYPE_'):
            categories['education_degree_type'][token_id] = token_str
        elif token_str.startswith('EDU_INSTITUTION_'):
            categories['education_institution'][token_id] = token_str
        elif token_str.startswith('EDU_SUBJECT_'):
            categories['education_subject'][token_id] = token_str
        elif token_str.startswith('PPL_JOB_'):
            categories['people_jobs'][token_id] = token_str
        elif token_str.startswith('PEOPLE_JOB_TITLE_'):
            categories['people_job_titles'][token_id] = token_str
    return categories

# === EVENT TYPE PARSING ===
def parse_event_types(sequence, idx_to_vocab, token_categories):
    """Return a dict of specific event type presence/counts for a token sequence."""
    event_presence = {
        'has_investment_events': False,
        'has_acquisition_events': False,
        'has_ipo_events': False,
        'has_education_events': False,
        'has_people_events': False,
    }
    event_counts = {
        'investment_event_count': 0,
        'acquisition_event_count': 0,
        'ipo_event_count': 0,
        'education_event_count': 0,
        'people_event_count': 0,
    }
    
    clean_sequence = sequence[sequence > 0].cpu().numpy() if torch.is_tensor(sequence) else sequence[sequence > 0]
    
    for token_id in clean_sequence:
        # Check for ALL event types (not exclusive)
        # Investment events
        if (int(token_id) in token_categories['investment_types'] or 
            int(token_id) in token_categories['investment_amounts']):
            event_presence['has_investment_events'] = True
            event_counts['investment_event_count'] += 1
        
        # Acquisition events
        if int(token_id) in token_categories['acquisition_types']:
            event_presence['has_acquisition_events'] = True
            event_counts['acquisition_event_count'] += 1
        
        # IPO events
        if (int(token_id) in token_categories['ipo_exchanges'] or 
            int(token_id) in token_categories['ipo_money_raised']):
            event_presence['has_ipo_events'] = True
            event_counts['ipo_event_count'] += 1
        
        # Education events
        if (int(token_id) in token_categories['education_degree_type'] or 
            int(token_id) in token_categories['education_institution'] or 
            int(token_id) in token_categories['education_subject']):
            event_presence['has_education_events'] = True
            event_counts['education_event_count'] += 1
        
        # People events
        if (int(token_id) in token_categories['people_jobs'] or 
            int(token_id) in token_categories['people_job_titles']):
            event_presence['has_people_events'] = True
            event_counts['people_event_count'] += 1
    
    return event_presence, event_counts

# === MODEL/DATA LOADING ===
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
    vocab_to_idx = datamodule.vocabulary.token2index
    idx_to_vocab = datamodule.vocabulary.index2token
    return model, test_loader, device, vocab_to_idx, idx_to_vocab

# === DATA EXTRACTION ===
def get_predictions_and_event_metadata(model, test_loader, device, idx_to_vocab, vocab_to_idx, test_mode=TEST_MODE, test_batches=TEST_BATCHES):
    token_categories = parse_token_categories(vocab_to_idx)
    all_preds, all_probs, all_labels, all_event_presence, all_event_counts, all_sequences = [], [], [], [], [], []
    n_batches = test_batches if test_mode else None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['survival_label'].squeeze().to(device)
            logits = model(input_ids=input_ids, padding_mask=padding_mask)['survival_logits']
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            for j in range(input_ids.size(0)):
                presence, counts = parse_event_types(input_ids[j, 0, :], idx_to_vocab, token_categories)
                all_event_presence.append(presence)
                all_event_counts.append(counts)
                # Store the raw token sequence for granular analysis
                all_sequences.append(input_ids[j, 0, :].cpu().numpy())
            if test_mode and i + 1 >= test_batches:
                break
            if i % 10 == 0:
                print(f"Processed batch {i}")
    return np.array(all_preds), np.array(all_probs), np.array(all_labels), all_event_presence, all_event_counts, all_sequences

# === GRANULAR EVENT ANALYSIS ===
def analyze_granular_events(preds, probs, labels, event_metadata, token_categories, all_sequences):
    """Analyze contribution of specific event subcategories."""
    results = []
    subcategories = {
        'investment_types': list(token_categories['investment_types'].keys()),
        'acquisition_types': list(token_categories['acquisition_types'].keys()),
        'education_degree_type': list(token_categories['education_degree_type'].keys()),
        'people_jobs': list(token_categories['people_jobs'].keys()),
    }
    print(f"\n=== GRANULAR EVENT ANALYSIS ===")
    for subcategory_name, token_ids in subcategories.items():
        print(f"\n--- {subcategory_name.upper()} ---")
        for token_id in token_ids:
            token_str = token_categories[subcategory_name][token_id]
            companies_with_token = []
            companies_without_token = []
            for i, seq in enumerate(all_sequences):
                # Check if this specific token_id is present in the sequence
                if int(token_id) in seq:
                    companies_with_token.append(i)
                else:
                    companies_without_token.append(i)
            if len(companies_with_token) < 10:
                continue  # Skip if too few companies have this token
            with_token_labels = labels[companies_with_token]
            with_token_preds = preds[companies_with_token]
            with_token_probs = probs[companies_with_token]
            without_token_labels = labels[companies_without_token]
            without_token_preds = preds[companies_without_token]
            without_token_probs = probs[companies_without_token]
            accuracy_with = accuracy_score(with_token_labels, with_token_preds)
            actual_survival_with = np.mean(with_token_labels)
            predicted_survival_with = np.mean(with_token_preds)
            mean_prob_with = np.mean(with_token_probs)
            accuracy_without = accuracy_score(without_token_labels, without_token_preds)
            actual_survival_without = np.mean(without_token_labels)
            predicted_survival_without = np.mean(without_token_preds)
            mean_prob_without = np.mean(without_token_probs)
            results.append({
                'subcategory': subcategory_name,
                'token_id': token_id,
                'token_str': token_str,
                'n_with_token': len(companies_with_token),
                'n_without_token': len(companies_without_token),
                'accuracy_with': accuracy_with,
                'actual_survival_with': actual_survival_with,
                'predicted_survival_with': predicted_survival_with,
                'mean_prob_with': mean_prob_with,
                'accuracy_without': accuracy_without,
                'actual_survival_without': actual_survival_without,
                'predicted_survival_without': predicted_survival_without,
                'mean_prob_without': mean_prob_without,
                'survival_rate_diff': actual_survival_with - actual_survival_without,
                'prediction_diff': predicted_survival_with - predicted_survival_without,
                'prob_diff': mean_prob_with - mean_prob_without,
            })
            print(f"  {token_str}: {len(companies_with_token)} companies")
            print(f"    Accuracy: {accuracy_with:.3f} vs {accuracy_without:.3f}")
            print(f"    Actual survival: {actual_survival_with:.3f} vs {actual_survival_without:.3f}")
            print(f"    Predicted survival: {predicted_survival_with:.3f} vs {predicted_survival_without:.3f}")
            print(f"    Mean probability: {mean_prob_with:.3f} vs {mean_prob_without:.3f}")
    return results

# === MAIN ===
def main():
    print("Loading model and data...")
    model, test_loader, device, vocab_to_idx, idx_to_vocab = load_model_and_data()
    print("Model and data loaded. Getting predictions and event metadata...")
    token_categories = parse_token_categories(vocab_to_idx)
    print("\n=== TOKEN CATEGORY COUNTS ===")
    for category, tokens in token_categories.items():
        print(f"{category}: {len(tokens)} tokens")
    preds, probs, labels, event_presence, event_counts, all_sequences = get_predictions_and_event_metadata(
        model, test_loader, device, idx_to_vocab, vocab_to_idx, TEST_MODE, TEST_BATCHES)
    print(f"Test set size: {len(preds)} samples")
    print(f"\nLabel counts: {np.bincount(labels)}")
    print(f"Prediction counts: {np.bincount(preds)}")
    print(f"Mean predicted survival probability: {np.mean(probs):.3f}")
    print(f"Std of predicted survival probability: {np.std(probs):.3f}")
    print(f"\n=== EVENT PRESENCE COUNTS ===")
    df_presence = pd.DataFrame(event_presence)
    for event_type in df_presence.columns:
        count = df_presence[event_type].sum()
        percentage = (count / len(df_presence)) * 100
        print(f"{event_type}: {count} companies ({percentage:.1f}%)")
    print("\nAnalyzing granular event contributions...")
    granular_results = analyze_granular_events(preds, probs, labels, event_counts, token_categories, all_sequences)
    granular_df = pd.DataFrame(granular_results)
    granular_df.to_csv(f"{OUTPUT_DIR}/granular_event_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/granular_event_report.txt", "w") as f:
        f.write("GRANULAR EVENT CONTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for _, row in granular_df.iterrows():
            f.write(f"Token: {row['token_str']} (ID: {row['token_id']})\n")
            f.write(f"Subcategory: {row['subcategory']}\n")
            f.write(f"Companies with token: {row['n_with_token']}\n")
            f.write(f"Companies without token: {row['n_without_token']}\n")
            f.write(f"Accuracy: {row['accuracy_with']:.3f} (with) vs {row['accuracy_without']:.3f} (without)\n")
            f.write(f"Actual survival: {row['actual_survival_with']:.3f} (with) vs {row['actual_survival_without']:.3f} (without)\n")
            f.write(f"Predicted survival: {row['predicted_survival_with']:.3f} (with) vs {row['predicted_survival_without']:.3f} (without)\n")
            f.write(f"Mean probability: {row['mean_prob_with']:.3f} (with) vs {row['mean_prob_without']:.3f} (without)\n")
            f.write(f"Survival rate difference: {row['survival_rate_diff']:.3f}\n")
            f.write(f"Prediction difference: {row['prediction_diff']:.3f}\n")
            f.write(f"Probability difference: {row['prob_diff']:.3f}\n")
            f.write("-" * 40 + "\n\n")
    print(f"\nResults saved to {OUTPUT_DIR}/granular_event_analysis.csv and {OUTPUT_DIR}/granular_event_report.txt")

    # === WELL FUNDED ANALYSIS ===
    print("\nAnalyzing 'well funded' companies (series B-J or growth, excluding series A)...")
    well_funded_terms = [
        'series_b', 'series_c', 'series_d', 'series_e', 'series_f',
        'series_g', 'series_h', 'series_i', 'series_j', 'growth'
    ]
    # Find all token IDs matching these terms (case-insensitive)
    well_funded_token_ids = set()
    for token_str, token_id in vocab_to_idx.items():
        if any(term in token_str.lower() for term in well_funded_terms):
            well_funded_token_ids.add(token_id)
    well_funded_companies = []
    not_well_funded_companies = []
    for i, seq in enumerate(all_sequences):
        if any(int(token_id) in seq for token_id in well_funded_token_ids):
            well_funded_companies.append(i)
        else:
            not_well_funded_companies.append(i)
    # Compute stats
    wf_labels = labels[well_funded_companies]
    wf_preds = preds[well_funded_companies]
    wf_probs = probs[well_funded_companies]
    not_wf_labels = labels[not_well_funded_companies]
    not_wf_preds = preds[not_well_funded_companies]
    not_wf_probs = probs[not_well_funded_companies]
    well_funded_stats = {
        'n_well_funded': len(well_funded_companies),
        'n_not_well_funded': len(not_well_funded_companies),
        'accuracy_well_funded': accuracy_score(wf_labels, wf_preds) if len(wf_labels) > 0 else np.nan,
        'actual_survival_well_funded': np.mean(wf_labels) if len(wf_labels) > 0 else np.nan,
        'predicted_survival_well_funded': np.mean(wf_preds) if len(wf_preds) > 0 else np.nan,
        'mean_prob_well_funded': np.mean(wf_probs) if len(wf_probs) > 0 else np.nan,
        'accuracy_not_well_funded': accuracy_score(not_wf_labels, not_wf_preds) if len(not_wf_labels) > 0 else np.nan,
        'actual_survival_not_well_funded': np.mean(not_wf_labels) if len(not_wf_labels) > 0 else np.nan,
        'predicted_survival_not_well_funded': np.mean(not_wf_preds) if len(not_wf_preds) > 0 else np.nan,
        'mean_prob_not_well_funded': np.mean(not_wf_probs) if len(not_wf_probs) > 0 else np.nan,
        'survival_rate_diff': np.mean(wf_labels) - np.mean(not_wf_labels) if len(wf_labels) > 0 and len(not_wf_labels) > 0 else np.nan,
        'prediction_diff': np.mean(wf_preds) - np.mean(not_wf_preds) if len(wf_preds) > 0 and len(not_wf_preds) > 0 else np.nan,
        'prob_diff': np.mean(wf_probs) - np.mean(not_wf_probs) if len(wf_probs) > 0 and len(not_wf_probs) > 0 else np.nan,
    }
    pd.DataFrame([well_funded_stats]).to_csv(f"{OUTPUT_DIR}/well_funded_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/well_funded_report.txt", "w") as f:
        f.write("WELL FUNDED COMPANY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for k, v in well_funded_stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to {OUTPUT_DIR}/well_funded_analysis.csv and {OUTPUT_DIR}/well_funded_report.txt")

    # === TEAM ANALYSIS (PPL_ tokens, threshold 10) ===
    print("\nAnalyzing 'team' companies (>=10 unique PPL_ tokens)...")
    # Find all token IDs starting with 'PPL_'
    ppl_token_ids = set()
    for token_str, token_id in vocab_to_idx.items():
        if token_str.startswith('PPL_'):
            ppl_token_ids.add(token_id)
    team_companies = []
    not_team_companies = []
    for i, seq in enumerate(all_sequences):
        unique_ppl_tokens = set([int(token_id) for token_id in seq if int(token_id) in ppl_token_ids])
        if len(unique_ppl_tokens) >= 10:
            team_companies.append(i)
        else:
            not_team_companies.append(i)
    team_labels = labels[team_companies]
    team_preds = preds[team_companies]
    team_probs = probs[team_companies]
    not_team_labels = labels[not_team_companies]
    not_team_preds = preds[not_team_companies]
    not_team_probs = probs[not_team_companies]
    team_stats = {
        'n_team': len(team_companies),
        'n_not_team': len(not_team_companies),
        'accuracy_team': accuracy_score(team_labels, team_preds) if len(team_labels) > 0 else np.nan,
        'actual_survival_team': np.mean(team_labels) if len(team_labels) > 0 else np.nan,
        'predicted_survival_team': np.mean(team_preds) if len(team_preds) > 0 else np.nan,
        'mean_prob_team': np.mean(team_probs) if len(team_probs) > 0 else np.nan,
        'accuracy_not_team': accuracy_score(not_team_labels, not_team_preds) if len(not_team_labels) > 0 else np.nan,
        'actual_survival_not_team': np.mean(not_team_labels) if len(not_team_labels) > 0 else np.nan,
        'predicted_survival_not_team': np.mean(not_team_preds) if len(not_team_preds) > 0 else np.nan,
        'mean_prob_not_team': np.mean(not_team_probs) if len(not_team_probs) > 0 else np.nan,
        'survival_rate_diff': np.mean(team_labels) - np.mean(not_team_labels) if len(team_labels) > 0 and len(not_team_labels) > 0 else np.nan,
        'prediction_diff': np.mean(team_preds) - np.mean(not_team_preds) if len(team_preds) > 0 and len(not_team_preds) > 0 else np.nan,
        'prob_diff': np.mean(team_probs) - np.mean(not_team_probs) if len(team_probs) > 0 and len(not_team_probs) > 0 else np.nan,
    }
    pd.DataFrame([team_stats]).to_csv(f"{OUTPUT_DIR}/team_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/team_report.txt", "w") as f:
        f.write("TEAM COMPANY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for k, v in team_stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to {OUTPUT_DIR}/team_analysis.csv and {OUTPUT_DIR}/team_report.txt")

    # === HIGH GROWTH ANALYSIS (INV_INVESTMENT_TYPE + growth) ===
    print("\nAnalyzing 'high growth' companies (INV_INVESTMENT_TYPE + growth)...")
    high_growth_token_ids = set()
    for token_str, token_id in vocab_to_idx.items():
        if 'inv_investment_type' in token_str.lower() and 'growth' in token_str.lower():
            high_growth_token_ids.add(token_id)
    high_growth_companies = []
    not_high_growth_companies = []
    for i, seq in enumerate(all_sequences):
        if any(int(token_id) in seq for token_id in high_growth_token_ids):
            high_growth_companies.append(i)
        else:
            not_high_growth_companies.append(i)
    hg_labels = labels[high_growth_companies]
    hg_preds = preds[high_growth_companies]
    hg_probs = probs[high_growth_companies]
    not_hg_labels = labels[not_high_growth_companies]
    not_hg_preds = preds[not_high_growth_companies]
    not_hg_probs = probs[not_high_growth_companies]
    high_growth_stats = {
        'n_high_growth': len(high_growth_companies),
        'n_not_high_growth': len(not_high_growth_companies),
        'accuracy_high_growth': accuracy_score(hg_labels, hg_preds) if len(hg_labels) > 0 else np.nan,
        'actual_survival_high_growth': np.mean(hg_labels) if len(hg_labels) > 0 else np.nan,
        'predicted_survival_high_growth': np.mean(hg_preds) if len(hg_preds) > 0 else np.nan,
        'mean_prob_high_growth': np.mean(hg_probs) if len(hg_probs) > 0 else np.nan,
        'accuracy_not_high_growth': accuracy_score(not_hg_labels, not_hg_preds) if len(not_hg_labels) > 0 else np.nan,
        'actual_survival_not_high_growth': np.mean(not_hg_labels) if len(not_hg_labels) > 0 else np.nan,
        'predicted_survival_not_high_growth': np.mean(not_hg_preds) if len(not_hg_preds) > 0 else np.nan,
        'mean_prob_not_high_growth': np.mean(not_hg_probs) if len(not_hg_probs) > 0 else np.nan,
        'survival_rate_diff': np.mean(hg_labels) - np.mean(not_hg_labels) if len(hg_labels) > 0 and len(not_hg_labels) > 0 else np.nan,
        'prediction_diff': np.mean(hg_preds) - np.mean(not_hg_preds) if len(hg_preds) > 0 and len(not_hg_preds) > 0 else np.nan,
        'prob_diff': np.mean(hg_probs) - np.mean(not_hg_probs) if len(hg_probs) > 0 and len(not_hg_probs) > 0 else np.nan,
    }
    pd.DataFrame([high_growth_stats]).to_csv(f"{OUTPUT_DIR}/high_growth_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/high_growth_report.txt", "w") as f:
        f.write("HIGH GROWTH COMPANY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for k, v in high_growth_stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to {OUTPUT_DIR}/high_growth_analysis.csv and {OUTPUT_DIR}/high_growth_report.txt")

    # === EARLY STAGE ANALYSIS (seed, angel, pre_series, series_a) ===
    print("\nAnalyzing 'early stage' companies (seed, angel, pre_series, series_a)...")
    early_stage_terms = ['seed', 'angel', 'pre_series', 'series_a']
    early_stage_token_ids = set()
    for token_str, token_id in vocab_to_idx.items():
        if any(term in token_str.lower() for term in early_stage_terms):
            early_stage_token_ids.add(token_id)
    early_stage_companies = []
    not_early_stage_companies = []
    for i, seq in enumerate(all_sequences):
        if any(int(token_id) in seq for token_id in early_stage_token_ids):
            early_stage_companies.append(i)
        else:
            not_early_stage_companies.append(i)
    es_labels = labels[early_stage_companies]
    es_preds = preds[early_stage_companies]
    es_probs = probs[early_stage_companies]
    not_es_labels = labels[not_early_stage_companies]
    not_es_preds = preds[not_early_stage_companies]
    not_es_probs = probs[not_early_stage_companies]
    early_stage_stats = {
        'n_early_stage': len(early_stage_companies),
        'n_not_early_stage': len(not_early_stage_companies),
        'accuracy_early_stage': accuracy_score(es_labels, es_preds) if len(es_labels) > 0 else np.nan,
        'actual_survival_early_stage': np.mean(es_labels) if len(es_labels) > 0 else np.nan,
        'predicted_survival_early_stage': np.mean(es_preds) if len(es_preds) > 0 else np.nan,
        'mean_prob_early_stage': np.mean(es_probs) if len(es_probs) > 0 else np.nan,
        'accuracy_not_early_stage': accuracy_score(not_es_labels, not_es_preds) if len(not_es_labels) > 0 else np.nan,
        'actual_survival_not_early_stage': np.mean(not_es_labels) if len(not_es_labels) > 0 else np.nan,
        'predicted_survival_not_early_stage': np.mean(not_es_preds) if len(not_es_preds) > 0 else np.nan,
        'mean_prob_not_early_stage': np.mean(not_es_probs) if len(not_es_probs) > 0 else np.nan,
        'survival_rate_diff': np.mean(es_labels) - np.mean(not_es_labels) if len(es_labels) > 0 and len(not_es_labels) > 0 else np.nan,
        'prediction_diff': np.mean(es_preds) - np.mean(not_es_preds) if len(es_preds) > 0 and len(not_es_preds) > 0 else np.nan,
        'prob_diff': np.mean(es_probs) - np.mean(not_es_probs) if len(es_probs) > 0 and len(not_es_probs) > 0 else np.nan,
    }
    pd.DataFrame([early_stage_stats]).to_csv(f"{OUTPUT_DIR}/early_stage_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/early_stage_report.txt", "w") as f:
        f.write("EARLY STAGE COMPANY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for k, v in early_stage_stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to {OUTPUT_DIR}/early_stage_analysis.csv and {OUTPUT_DIR}/early_stage_report.txt")
    print("Done.")

if __name__ == "__main__":
    main()
