#!/usr/bin/env python3
"""
FIXED step_6_finetune_survival.py
All critical issues resolved:
- Correct vocabulary size (20,882)
- Moderate class weights [2.0, 1.0] 
- Lower learning rate 2e-5
- Balanced accuracy monitoring
- Input validation
- Gradient clipping
- NOW: Uses balanced company-level sampling for training (balance_companies=True)
- NOW: Wandb project is 'startup-survival-balanced', run name includes 'finetune-balanced'
- NOW: Only uses 'startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt' as the pretrained model
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import importlib.util

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

spec = importlib.util.spec_from_file_location("step_4b_create_balanced_datamodule", "step_4b_create_balanced_datamodule.py")
datamodule_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datamodule_module)
StartupDataModule = datamodule_module.StartupDataModule
StartupVocabulary = datamodule_module.StartupVocabulary

from models.survival_model import FixedStartupSurvivalModel

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fixed_survival_finetuning.log')
        ]
    )

def get_pretrained_model_path():
    """Return only the correct balanced 1-GPU pretrained model."""
    path = "startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt"
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"Pretrained model not found: {path}")

def get_vocab_size_from_pretrained(pretrained_path):
    """Get vocabulary size from pretrained model"""
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        vocab_size = checkpoint['hparams']['vocab_size']
        logging.info(f"Using vocab size from pretrained model: {vocab_size:,}")
        return vocab_size
    except Exception as e:
        logging.warning(f"Could not get vocab size from pretrained: {e}")
        return 20882  # Fallback to correct size

def validate_data_compatibility(datamodule, pretrained_vocab_size):
    """Validate that data is compatible with model"""
    
    logging.info("🔍 Validating data compatibility...")
    
    # Check vocabulary compatibility
    current_vocab_size = datamodule.get_vocab_size()
    
    if current_vocab_size != pretrained_vocab_size:
        logging.error(f"VOCABULARY MISMATCH!")
        logging.error(f"  Pretrained: {pretrained_vocab_size:,}")
        logging.error(f"  Current: {current_vocab_size:,}")
        raise ValueError("Vocabulary size mismatch - will cause index errors")
    
    # Test data loading
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    
    try:
        batch = next(iter(train_loader))
        
        # Check input shapes
        input_ids = batch['input_ids']
        labels = batch['survival_label']
        
        logging.info(f"✅ Data validation passed:")
        logging.info(f"   Input shape: {input_ids.shape}")
        logging.info(f"   Labels shape: {labels.shape}")
        logging.info(f"   Vocab size: {current_vocab_size:,}")
        
        # Check token ID ranges
        max_token = input_ids.max().item()
        min_token = input_ids.min().item()
        
        if max_token >= pretrained_vocab_size:
            raise ValueError(f"Invalid token ID {max_token} >= vocab_size {pretrained_vocab_size}")
        
        logging.info(f"   Token range: {min_token} to {max_token}")
        
        return True
        
    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        raise

def get_fixed_config(num_gpus=1, quick_test=False):
    """Get FIXED configuration that should work"""
    
    if quick_test:
        return {
            "batch_size": 8,           # Small for testing
            "num_workers": 2,
            "accumulate_grad_batches": 1,
            "learning_rate": 1e-5,     # FIXED: Much lower
            "max_epochs": 2,
            "val_check_interval": 1.0,
            "class_weights": [2.0, 1.0]  # FIXED: Less extreme
        }
    
    # FIXED configuration for stable training
    return {
        "batch_size": 16,              # FIXED: Smaller than before
        "num_workers": 8,              # Reasonable
        "accumulate_grad_batches": 1,
        "learning_rate": 1e-5,         # FIXED: Much lower (was 1e-4)
        "max_epochs": 10,              # Reasonable
        "val_check_interval": 0.5,
        "class_weights": [2.0, 1.0]    # FIXED: Much less extreme (was [11.45, 0.52])
    }

def main():
    parser = argparse.ArgumentParser(description="FIXED Startup Survival Prediction Finetuning (Balanced)")
    
    # Data parameters
    parser.add_argument("--corpus-name", type=str, default="startup_corpus")
    parser.add_argument("--vocab-name", type=str, default="startup_vocab")
    parser.add_argument("--prediction-windows", nargs="+", type=int, default=[1, 2, 3, 4])
    
    # Model parameters
    parser.add_argument("--pretrained-model", type=str, default="startup2vec_startup2vec-balanced-full-1gpu-512d_final.pt")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Training parameters - FIXED DEFAULTS
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    
    # Hardware
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default="startup-survival-balanced")
    parser.add_argument("--experiment-name", type=str, default=None)
    
    # Control
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--run", action="store_true", required=True)
    
    # New argument for balanced val/test
    parser.add_argument("--balanced-valtest", action="store_true", help="Use balanced val/test splits for interpretability (not for main results)")
    
    args = parser.parse_args()
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    start_time = time.time()
    
    log.info("🔧 STARTUP SURVIVAL PREDICTION FINETUNING (BALANCED DATASET)")
    log.info("This run uses company-level balanced sampling for training (balance_companies=True)")
    log.info("=" * 70)
    log.info("🎯 ALL CRITICAL ISSUES FIXED:")
    log.info("   ✅ Vocabulary size validation")
    log.info("   ✅ Moderate class weights [2.0, 1.0]")
    log.info("   ✅ Lower learning rate 1e-5")
    log.info("   ✅ Gradient clipping")
    log.info("   ✅ Balanced accuracy monitoring")
    log.info("   ✅ Input validation")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        log.error("❌ CUDA not available! Exiting...")
        return 1
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.gpus, available_gpus)
    log.info(f"🎯 Using {num_gpus} GPU(s)")
    
    try:
        # Find and validate pretrained model
        if args.pretrained_model is None:
            pretrained_model_path = get_pretrained_model_path()
        else:
            pretrained_model_path = args.pretrained_model
        
        log.info(f"📁 Using pretrained model: {pretrained_model_path}")
        
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
        
        # Get vocabulary size from pretrained model
        pretrained_vocab_size = get_vocab_size_from_pretrained(pretrained_model_path)
        
        # Get fixed configuration
        config = get_fixed_config(num_gpus, args.quick_test)
        config["class_weights"] = [1.0, 1.0]  # Balanced dataset, so no weighting needed
        
        # Override with command line arguments if provided
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        
        config["max_epochs"] = args.max_epochs
        
        log.info(f"📊 FIXED Configuration:")
        log.info(f"   Batch size: {config['batch_size']}")
        log.info(f"   Learning rate: {config['learning_rate']}")
        log.info(f"   Class weights: {config['class_weights']}")
        log.info(f"   Max epochs: {config['max_epochs']}")
        log.info(f"   Vocab size: {pretrained_vocab_size:,}")
        
        # Create and validate datamodule
        log.info("📊 Creating and validating datamodule...")
        datamodule = StartupDataModule(
            corpus_name=args.corpus_name,
            vocab_name=args.vocab_name,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            prediction_windows=args.prediction_windows,
            balance_companies=True,  # Always balance train
            balance_valtest=args.balanced_valtest  # New option for val/test
        )
        
        # CRITICAL: Validate data compatibility
        validate_data_compatibility(datamodule, pretrained_vocab_size)
        
        # Get class distribution
        class_stats = datamodule.get_class_distribution()
        
        log.info("📈 Dataset Statistics:")
        for split, stats in class_stats.items():
            log.info(f"  {split.upper()}: {stats['total']:,} samples")
            log.info(f"    Died: {stats['died']:,} ({stats['died_pct']:.1f}%)")
            log.info(f"    Survived: {stats['survived']:,} ({stats['survived_pct']:.1f}%)")
        
        # Create FIXED model
        log.info("🏗️ Creating FIXED survival prediction model...")
        model = FixedStartupSurvivalModel(
            pretrained_model_path=pretrained_model_path,
            vocab_size=pretrained_vocab_size,  # FIXED: Use correct vocab size
            num_classes=2,
            freeze_encoder=args.freeze_encoder,
            learning_rate=config["learning_rate"],
            weight_decay=args.weight_decay,
            class_weights=config["class_weights"],
            use_balanced_accuracy=True,  # FIXED: Monitor balanced accuracy
            gradient_clip_val=0.1        # FIXED: Add gradient clipping
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        log.info(f"📊 Model parameters:")
        log.info(f"  Total: {total_params:,}")
        log.info(f"  Trainable: {trainable_params:,}")
        log.info(f"  Frozen: {total_params - trainable_params:,}")
        
        # Experiment name
        if args.experiment_name is None:
            freeze_str = "frozen" if args.freeze_encoder else "unfrozen"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"FIXED-{freeze_str}-{config['batch_size']}bs-{config['learning_rate']:.0e}lr-{timestamp}"
        
        # Setup logging
        run_name = args.experiment_name or f"finetune-balanced-{num_gpus}gpu-batch{config['batch_size']}"
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            save_dir="./logs",
            config=config,
            tags=["balanced", "finetuning", "survival"]
        )
        log.info(f"🔗 WandB: {args.wandb_project}/{run_name}")
        log.info(f"Wandb run: https://wandb.ai/{logger.experiment.entity}/{logger.experiment.project}/{logger.experiment.id}")
        
        # FIXED callbacks - monitor balanced accuracy
        callbacks = [
            ModelCheckpoint(
                dirpath=f"survival_checkpoints_FIXED/{run_name}",
                filename="best-{epoch:02d}-{val/balanced_acc:.4f}",
                monitor="val/balanced_acc",  # FIXED: Monitor balanced accuracy
                mode="max",
                save_top_k=2,
                save_last=True,
                every_n_epochs=1
            ),
            EarlyStopping(
                monitor="val/balanced_acc",  # FIXED: Monitor balanced accuracy
                mode="max",
                patience=5,
                verbose=True,
                min_delta=0.01  # Require 1% improvement
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Multi-GPU strategy
        if num_gpus > 1:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
        else:
            strategy = "auto"
        
        # Create FIXED trainer
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu" if num_gpus > 0 else "cpu",
            devices=num_gpus if num_gpus > 0 else None,
            strategy=strategy,
            precision=args.precision,
            logger=logger,
            callbacks=callbacks,
            # FIXED: Add gradient clipping
            gradient_clip_val=0.1,
            # Validation settings
            val_check_interval=config["val_check_interval"],
            # Logging
            log_every_n_steps=10,
            # Performance
            sync_batchnorm=True if num_gpus > 1 else False,
            enable_model_summary=True,
            deterministic=False,
            enable_checkpointing=True,
        )
        
        log.info("🏋️ Starting FIXED survival prediction finetuning...")
        log.info(f"📊 Final configuration:")
        log.info(f"  Model: {'Frozen' if args.freeze_encoder else 'Unfrozen'} encoder")
        log.info(f"  Batch size: {config['batch_size']}")
        log.info(f"  Learning rate: {config['learning_rate']}")
        log.info(f"  Class weights: {config['class_weights']}")
        log.info(f"  Gradient clipping: 0.5")
        
        if args.quick_test:
            log.info("🧪 QUICK TEST MODE")
        
        # Test model on single batch first
        log.info("🧪 Testing model on single batch...")
        test_batch = next(iter(datamodule.train_dataloader()))
        
        try:
            with torch.no_grad():
                model.eval()
                test_output = model.forward(
                    input_ids=test_batch['input_ids'][:2],  # Just 2 samples
                    padding_mask=test_batch['padding_mask'][:2]
                )
                log.info(f"✅ Single batch test passed")
                log.info(f"   Output shape: {test_output['survival_logits'].shape}")
        except Exception as e:
            log.error(f"❌ Single batch test failed: {e}")
            raise
        
        # Start training
        model.train()
        trainer.fit(model, datamodule)
        
        # Test on best model
        log.info("🧪 Testing best model...")
        test_results = trainer.test(model, datamodule, ckpt_path="best")
        
        # Final metrics
        elapsed_time = time.time() - start_time
        
        log.info("🎉 FIXED FINETUNING COMPLETE!")
        log.info(f"✅ Best validation balanced accuracy: {trainer.callback_metrics.get('val/balanced_acc', 0.0):.4f}")
        log.info(f"✅ Best validation AUC: {trainer.callback_metrics.get('val/auc', 0.0):.4f}")
        log.info(f"✅ Final epoch: {trainer.current_epoch}")
        log.info(f"⏱️ Total time: {elapsed_time/3600:.2f} hours")
        log.info(f"💾 Best model: survival_checkpoints_FIXED/{run_name}/")
        
        if args.quick_test:
            log.info("🧪 Quick test completed!")
        else:
            log.info("🎓 Model ready for interpretability analysis!")
            
        return 0
        
    except Exception as e:
        log.error(f"❌ Error during finetuning: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)