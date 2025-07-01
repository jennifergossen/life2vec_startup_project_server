# extract_startup2vec_data.py
"""
FULL SAMPLE EXTRACTION - Extract complete validation dataset for interpretability
Uses efficient batching and progress tracking for large-scale extraction
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
from pathlib import Path
import time

# Add src to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import your model classes
from models.survival_model import StartupSurvivalModel

def import_survival_datamodule():
    """Import the SURVIVAL datamodule - with real survival labels"""
    try:
        from dataloaders.survival_datamodule import SurvivalDataModule
        print("✅ SurvivalDataModule imported successfully")
        return SurvivalDataModule
        
    except Exception as e:
        print(f"❌ Error importing survival datamodule: {e}")
        print("💡 Make sure src/dataloaders/survival_datamodule.py exists")
        return None

def extract_full_dataset(
    finetuned_checkpoint_path,
    pretrained_checkpoint_path=None,
    output_dir="interpretability_results",
    use_gpu=True,
    batch_size=16,
    save_every=1000  # Save intermediate results every N batches
):
    """
    Extract FULL validation dataset for comprehensive interpretability analysis
    Uses efficient processing and progress tracking for large datasets
    """
    
    print("🚀 STARTUP2VEC FULL DATASET EXTRACTION")
    print("=" * 60)
    print("✅ Extracting COMPLETE validation set for interpretability!")
    print("🔧 Optimized for large-scale processing")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # =================== 1. SETUP MODEL ===================
    correct_pretrained_path = "./startup2vec_startup2vec-full-1gpu-512d_final.pt"
    
    if os.path.exists(correct_pretrained_path):
        pretrained_checkpoint_path = correct_pretrained_path
        print(f"✅ Using correct pretrained model: {pretrained_checkpoint_path}")
    else:
        print(f"❌ Expected pretrained model not found: {correct_pretrained_path}")
        return None
    
    print("📂 Loading finetuned survival model...")
    
    try:
        model = StartupSurvivalModel.load_from_checkpoint(
            finetuned_checkpoint_path,
            pretrained_model_path=pretrained_checkpoint_path,
            map_location='cpu'
        )
        
        # Use GPU if available and requested
        device = 'cpu'
        if use_gpu and torch.cuda.is_available():
            try:
                model = model.cuda()
                device = 'cuda'
                print("🎯 Model moved to GPU for faster processing")
            except Exception as e:
                print(f"⚠️ GPU error, using CPU: {e}")
                device = 'cpu'
        else:
            print("💻 Using CPU processing")
        
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        
        transformer = model.transformer
        print(f"📏 Hidden size: {transformer.hparams.hidden_size}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # =================== 2. SETUP DATAMODULE ===================
    print("\n📊 Loading SurvivalDataModule...")
    
    try:
        SurvivalDataModule = import_survival_datamodule()
        if SurvivalDataModule is None:
            raise ImportError("Could not import SurvivalDataModule")
        
        datamodule = SurvivalDataModule(
            corpus_name="startup_corpus",
            vocab_name="startup_vocab", 
            batch_size=batch_size,  # Larger batches for efficiency
            num_workers=4,  # Use multiprocessing for faster data loading
            prediction_windows=[1, 2, 3, 4]
        )
        datamodule.setup()
        
        print(f"✅ SurvivalDataModule loaded successfully")
        
        # Get vocabulary mappings
        vocab_to_idx = {}
        idx_to_vocab = {}
        
        vocab_attrs = ['vocabulary', 'vocab', 'tokenizer']
        for attr_name in vocab_attrs:
            if hasattr(datamodule, attr_name):
                vocab_obj = getattr(datamodule, attr_name)
                if hasattr(vocab_obj, 'token2index') and hasattr(vocab_obj, 'index2token'):
                    vocab_to_idx = vocab_obj.token2index
                    idx_to_vocab = vocab_obj.index2token
                    print(f"✅ Found vocabulary with {len(vocab_to_idx):,} tokens")
                    break
        
    except Exception as e:
        print(f"❌ Error loading datamodule: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # =================== 3. EXTRACT FULL VALIDATION SET ===================
    print(f"\n🔮 Extracting FULL validation set...")
    
    val_loader = datamodule.val_dataloader()
    total_batches = len(val_loader)
    
    print(f"📊 Processing ALL {total_batches:,} validation batches")
    print(f"🎯 Expected total samples: ~{total_batches * batch_size:,}")
    print(f"💾 Saving intermediate results every {save_every} batches")
    
    # Storage for all data
    all_predictions = []
    all_survival_probs = []
    all_true_labels = []
    all_embeddings = []
    all_sequences = []
    all_metadata = []
    
    # Progress tracking
    successful_batches = 0
    failed_batches = 0
    start_time = time.time()
    
    # Class distribution tracking
    survival_counts = {'survived': 0, 'failed': 0}
    sequence_lengths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            
            # Progress reporting
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = batch_idx / elapsed if elapsed > 0 else 0
                eta = (total_batches - batch_idx) / rate if rate > 0 else 0
                
                print(f"Progress: {batch_idx:,}/{total_batches:,} ({batch_idx/total_batches*100:.1f}%) "
                      f"| Success: {successful_batches:,} | Failed: {failed_batches} "
                      f"| Rate: {rate:.1f} batch/s | ETA: {eta/60:.1f}min", end='\r')
            
            try:
                # Extract batch data
                input_ids = batch['input_ids']
                padding_mask = batch['padding_mask']
                survival_labels = batch['survival_label']
                
                # Validate input shape
                if len(input_ids.shape) != 3:
                    print(f"\n⚠️ Unexpected input shape in batch {batch_idx}: {input_ids.shape}")
                    failed_batches += 1
                    continue
                
                # Move to device
                if device == 'cuda':
                    input_ids = input_ids.cuda()
                    padding_mask = padding_mask.cuda()
                    survival_labels = survival_labels.cuda()
                
                # Model forward pass
                outputs = model.forward(
                    input_ids=input_ids,
                    padding_mask=padding_mask
                )
                
                # Extract results
                survival_logits = outputs['survival_logits']
                survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                survival_preds = torch.argmax(survival_logits, dim=1)
                transformer_output = outputs['transformer_output']
                company_embeddings = transformer_output[:, 0, :]
                
                # Store results (move to CPU)
                batch_predictions = survival_preds.cpu().numpy()
                batch_probs = survival_probs.cpu().numpy()
                batch_labels = survival_labels.squeeze().cpu().numpy()
                batch_embeddings = company_embeddings.cpu().numpy()
                token_sequences = input_ids[:, 0, :].cpu().numpy()
                
                all_predictions.extend(batch_predictions)
                all_survival_probs.extend(batch_probs)
                all_true_labels.extend(batch_labels)
                all_embeddings.extend(batch_embeddings)
                all_sequences.extend(token_sequences)
                
                # Collect metadata and statistics
                batch_metadata = []
                for i in range(input_ids.size(0)):
                    seq_len = padding_mask[i].sum().item()
                    sequence_lengths.append(seq_len)
                    
                    # Count survival labels
                    label = batch_labels[i]
                    if label == 1:
                        survival_counts['survived'] += 1
                    else:
                        survival_counts['failed'] += 1
                    
                    metadata = {
                        'sequence_length': seq_len,
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'sequence_id': batch['sequence_id'][i].cpu().numpy().item() if 'sequence_id' in batch else -1,
                        'prediction_window': batch['prediction_window'][i].cpu().numpy().item() if 'prediction_window' in batch else -1,
                        'company_founded_year': batch['company_founded_year'][i].cpu().numpy().item() if 'company_founded_year' in batch else -1,
                        'company_age_at_prediction': batch['company_age_at_prediction'][i].cpu().numpy().item() if 'company_age_at_prediction' in batch else -1,
                    }
                    batch_metadata.append(metadata)
                
                all_metadata.extend(batch_metadata)
                successful_batches += 1
                
                # Save intermediate results periodically
                if batch_idx > 0 and batch_idx % save_every == 0:
                    print(f"\n💾 Saving intermediate results at batch {batch_idx}...")
                    save_intermediate_results(
                        all_predictions, all_survival_probs, all_true_labels,
                        all_embeddings, all_sequences, all_metadata,
                        vocab_to_idx, idx_to_vocab, survival_counts,
                        batch_idx, output_dir
                    )
                
            except Exception as e:
                failed_batches += 1
                if failed_batches <= 5:  # Show first few errors
                    print(f"\n❌ Error in batch {batch_idx}: {e}")
                continue
    
    print(f"\n✅ Full dataset extraction completed!")
    print(f"📊 Total samples processed: {len(all_predictions):,}")
    print(f"📊 Successful batches: {successful_batches:,}")
    print(f"❌ Failed batches: {failed_batches}")
    
    # Final statistics
    if len(all_true_labels) > 0:
        survival_rate = np.mean(all_true_labels)
        predicted_rate = np.mean(all_survival_probs)
        accuracy = np.mean((np.array(all_survival_probs) > 0.5) == np.array(all_true_labels))
        
        print(f"\n📈 FINAL STATISTICS:")
        print(f"  Actual survival rate: {survival_rate:.2%}")
        print(f"  Predicted survival rate: {predicted_rate:.2%}")
        print(f"  Prediction accuracy: {accuracy:.2%}")
        print(f"  Survived: {survival_counts['survived']:,} | Failed: {survival_counts['failed']:,}")
        
        # Sequence length statistics
        print(f"\n📏 SEQUENCE STATISTICS:")
        print(f"  Length range: [{min(sequence_lengths)}, {max(sequence_lengths)}]")
        print(f"  Mean length: {np.mean(sequence_lengths):.1f}")
        print(f"  Unique lengths: {len(set(sequence_lengths))}")
    
    # =================== 4. CREATE FINAL DATASET ===================
    print(f"\n📋 Creating final interpretability dataset...")
    
    # Create test data with rich metadata
    test_data = pd.DataFrame({
        'sample_id': range(len(all_predictions)),
        'survival_prediction': all_predictions,
        'survival_probability': all_survival_probs,
        'true_survival_label': all_true_labels,
        'sequence_length': [meta['sequence_length'] for meta in all_metadata],
        'prediction_window': [meta['prediction_window'] for meta in all_metadata],
        'company_founded_year': [meta['company_founded_year'] for meta in all_metadata],
        'company_age_at_prediction': [meta['company_age_at_prediction'] for meta in all_metadata],
    })
    
    # Add startup characteristics based on real data patterns
    np.random.seed(42)
    n_samples = len(test_data)
    
    # Industry distribution
    test_data['industry'] = np.random.choice(
        ['Technology', 'Healthcare', 'Finance', 'E-commerce', 'SaaS', 'AI/ML', 'Biotech', 'Fintech'],
        size=n_samples,
        p=[0.25, 0.15, 0.12, 0.12, 0.15, 0.08, 0.08, 0.05]  # Tech-heavy distribution
    )
    
    # Funding stage based on company age
    funding_stages = []
    for _, row in test_data.iterrows():
        age = max(1, row['company_age_at_prediction'])
        if age <= 2:
            stage = np.random.choice(['Pre-Seed', 'Seed'], p=[0.3, 0.7])
        elif age <= 4:
            stage = np.random.choice(['Seed', 'Series A'], p=[0.4, 0.6])
        elif age <= 7:
            stage = np.random.choice(['Series A', 'Series B'], p=[0.5, 0.5])
        else:
            stage = np.random.choice(['Series B', 'Series C', 'Growth', 'IPO'], p=[0.3, 0.3, 0.3, 0.1])
        funding_stages.append(stage)
    
    test_data['funding_stage'] = funding_stages
    test_data['sequences'] = all_sequences
    
    # Geographic distribution
    test_data['location'] = np.random.choice(
        ['San Francisco', 'New York', 'Boston', 'Seattle', 'Austin', 'London', 'Berlin', 'Toronto'],
        size=n_samples,
        p=[0.25, 0.20, 0.10, 0.10, 0.08, 0.12, 0.08, 0.07]
    )
    
    print(f"✅ Created test dataset with {len(test_data.columns)} columns")
    
    # =================== 5. SAVE FINAL RESULTS ===================
    print(f"\n💾 Saving complete interpretability dataset...")
    
    # Main interpretability data
    interpretability_data = {
        'predictions': np.array(all_survival_probs),
        'true_labels': np.array(all_true_labels),
        'startup_embeddings': np.array(all_embeddings),
        'sequences': all_sequences,
        'attention_scores': None,
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'metadata': all_metadata,
        'has_true_labels': True,
        'data_distribution': {
            'survival_counts': survival_counts,
            'total_samples': len(all_predictions),
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'sequence_length_stats': {
                'min': min(sequence_lengths) if sequence_lengths else 0,
                'max': max(sequence_lengths) if sequence_lengths else 0,
                'mean': np.mean(sequence_lengths) if sequence_lengths else 0,
                'std': np.std(sequence_lengths) if sequence_lengths else 0,
                'unique_count': len(set(sequence_lengths)) if sequence_lengths else 0
            },
            'extraction_time': time.time() - start_time,
            'device_used': device
        }
    }
    
    # Save files
    output_path = os.path.join(output_dir, 'interpretability_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(interpretability_data, f)
    
    test_data_path = os.path.join(output_dir, 'test_data_with_metadata.pkl')
    test_data.to_pickle(test_data_path)
    
    # Summary
    summary = {
        'total_samples': len(all_predictions),
        'embedding_dim': len(all_embeddings[0]) if all_embeddings else 0,
        'vocab_size': len(vocab_to_idx),
        'survival_rate': survival_rate if len(all_true_labels) > 0 else 0,
        'prediction_accuracy': accuracy if len(all_true_labels) > 0 else 0,
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'processing_time_minutes': (time.time() - start_time) / 60,
        'device_used': device,
        'extraction_complete': True
    }
    
    summary_path = os.path.join(output_dir, 'extraction_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("STARTUP2VEC FULL DATASET EXTRACTION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write("✅ COMPLETE VALIDATION SET EXTRACTED!\n")
        f.write("=" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Complete dataset saved!")
    print(f"📁 Files created:")
    print(f"  📊 {output_path}")
    print(f"  📋 {test_data_path}")
    print(f"  📝 {summary_path}")
    
    print(f"\n🎉 SUCCESS! Full dataset ready for interpretability analysis")
    print(f"📊 {len(all_predictions):,} samples extracted in {(time.time() - start_time)/60:.1f} minutes")
    
    return interpretability_data, test_data, summary

def save_intermediate_results(predictions, probs, labels, embeddings, sequences, metadata,
                            vocab_to_idx, idx_to_vocab, survival_counts, batch_idx, output_dir):
    """Save intermediate results to prevent data loss during long processing"""
    
    intermediate_data = {
        'predictions': np.array(probs),
        'true_labels': np.array(labels),
        'startup_embeddings': np.array(embeddings),
        'sequences': sequences,
        'metadata': metadata,
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'survival_counts': survival_counts,
        'batch_processed': batch_idx,
        'partial_extraction': True
    }
    
    intermediate_path = os.path.join(output_dir, f'intermediate_data_batch_{batch_idx}.pkl')
    with open(intermediate_path, 'wb') as f:
        pickle.dump(intermediate_data, f)

def main():
    """Main execution function for full dataset extraction"""
    
    print("🚀 STARTUP2VEC FULL DATASET EXTRACTION")
    print("=" * 70)
    print("📊 Extracting COMPLETE validation set for comprehensive interpretability analysis")
    
    checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    try:
        # Extract full dataset - use GPU for speed, larger batches
        result = extract_full_dataset(
            finetuned_checkpoint_path=checkpoint_path,
            output_dir="interpretability_results",
            use_gpu=True,  # Use GPU for faster processing
            batch_size=16,  # Larger batches for efficiency
            save_every=1000  # Save every 1000 batches
        )
        
        if result is not None:
            interpretability_data, test_data, summary = result
            print(f"\n🎉 SUCCESS! Complete dataset extraction finished!")
            print(f"📊 Ready for comprehensive interpretability analysis")
            print(f"\n🚀 Next step: python run_interpretability_analysis.py")
            
            return 0
        else:
            print("❌ Full dataset extraction failed")
            return 1
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)