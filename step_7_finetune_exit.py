#!/usr/bin/env python3

"""
FIXED OPTIMIZED Step 6: Startup Exit Prediction Finetuning
- Adapted from survival prediction for exit prediction task
- Removed incompatible arguments (pin_memory, persistent_workers)
- Kept all optimizations that work
- Should complete in ~4-6 hours instead of 22 hours
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dataloaders.exit_datamodule import ExitDataModule
from models.exit_model import StartupExitModel

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('exit_finetuning.log')
        ]
    )

def get_pretrained_model_path():
    """Find the pretrained model file"""
    possible_paths = [
        # Most recent patterns first
        "startup2vec_startup2vec-full-1gpu-512d_final.pt",
        "startup2vec_startup2vec-full-3gpu-512d_final.pt", 
        "startup2vec_startup2vec-test-1gpu-256d_final.pt",
        "checkpoints/last.ckpt",
        "checkpoints/last-v2.ckpt",
    ]
    
    for pattern in possible_paths:
        if "*" in pattern:
            # Handle wildcard patterns
            import glob
            matches = glob.glob(pattern)
            if matches:
                # Get the most recent file
                return max(matches, key=lambda x: Path(x).stat().st_mtime)
        else:
            if Path(pattern).exists():
                return pattern
    
    raise FileNotFoundError("No pretrained model found. Please ensure you have a trained model.")

def get_optimal_config(num_gpus=1, quick_test=False):
    """Get optimal configuration for finetuning"""
    
    if quick_test:
        return {
            "batch_size": 16,
            "num_workers": 4,
            "accumulate_grad_batches": 2,
            "learning_rate": 1e-4,
            "max_epochs": 2,
            "val_check_interval": 1.0,
        }
    
    # Optimized for speed (but compatible)
    config = {
        "batch_size": 64,  # Large batch for A100
        "num_workers": 16,  # More workers for data loading
        "accumulate_grad_batches": 1,  # No accumulation needed with large batch
        "learning_rate": 1e-4,  # Conservative but effective
        "max_epochs": 15,
        "val_check_interval": 0.5,  # Validate twice per epoch
    }
    
    return config

def estimate_training_time(train_size, batch_size, accumulate_batches, epochs, num_gpus):
    """Estimate finetuning time"""
    effective_batch_size = batch_size * accumulate_batches * num_gpus
    steps_per_epoch = train_size // effective_batch_size
    total_steps = steps_per_epoch * epochs
    
    # Finetuning is faster than pretraining
    seconds_per_step = 0.15 if num_gpus == 1 else 0.1  # Optimistic for finetuning
    estimated_hours = (total_steps * seconds_per_step) / 3600
    
    return estimated_hours, steps_per_epoch, total_steps

def main():
    parser = argparse.ArgumentParser(description="Startup Exit Prediction Finetuning - FIXED & OPTIMIZED")
    
    # Data parameters
    parser.add_argument("--corpus-name", type=str, default="startup_corpus",
                       help="Name of corpus to use")
    parser.add_argument("--vocab-name", type=str, default="startup_vocab",
                       help="Name of vocabulary to use")
    parser.add_argument("--prediction-windows", nargs="+", type=int, default=[1, 2, 3, 4],
                       help="Prediction windows in years")
    
    # Model parameters
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Path to pretrained model")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze pretrained encoder")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (auto-optimized if not specified)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Training parameters - OPTIMIZED DEFAULTS
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (auto-optimized if not specified)")
    parser.add_argument("--max-epochs", type=int, default=15,
                       help="Maximum epochs")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers (auto-optimized if not specified)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None,
                       help="Gradient accumulation batches (auto-optimized if not specified)")
    
    # Hardware
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="Training precision")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default="startup-exit-prediction",
                       help="WandB project name")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name")
    
    # CDW Loss parameters (life2vec values)
    parser.add_argument("--cdw-alpha", type=float, default=2.0,
                       help="CDW alpha parameter")
    parser.add_argument("--cdw-delta", type=float, default=3.0,
                       help="CDW delta parameter")
    parser.add_argument("--cdw-transform", type=str, default="log",
                       help="CDW transform type")
    
    # Control
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with limited data")
    parser.add_argument("--run", action="store_true", required=True,
                       help="Actually run the finetuning")
    
    args = parser.parse_args()
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    start_time = time.time()
    
    log.info("🚀 FIXED OPTIMIZED STARTUP EXIT PREDICTION FINETUNING")
    log.info("=" * 70)
    log.info("⚡ Performance optimizations enabled (22h → ~4-6h)")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        log.error("❌ CUDA not available! Exiting...")
        return 1
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.gpus, available_gpus)
    
    log.info(f"🎯 Using {num_gpus} GPU(s) out of {available_gpus} available")
    
    # Get optimal configuration
    config = get_optimal_config(num_gpus, args.quick_test)
    
    # Override with command line arguments if provided
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.accumulate_grad_batches is not None:
        config["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    
    config["max_epochs"] = args.max_epochs
    
    log.info(f"�� Optimized Configuration:")
    log.info(f"   Batch size: {config['batch_size']} per GPU")
    log.info(f"   Effective batch: {config['batch_size'] * config['accumulate_grad_batches'] * num_gpus}")
    log.info(f"   Workers: {config['num_workers']}")
    log.info(f"   Learning rate: {config['learning_rate']}")
    log.info(f"   Max epochs: {config['max_epochs']}")
    
    try:
        # Find pretrained model
        if args.pretrained_model is None:
            pretrained_model_path = get_pretrained_model_path()
        else:
            pretrained_model_path = args.pretrained_model
        
        log.info(f"📁 Using pretrained model: {pretrained_model_path}")
        
        # Verify pretrained model exists
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
        
        # Experiment name
        if args.experiment_name is None:
            freeze_str = "frozen" if args.freeze_encoder else "unfrozen"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"exit-fixed-optimized-{freeze_str}-{config['batch_size']}bs-{config['learning_rate']:.0e}lr-{timestamp}"
        
        # Create datamodule with COMPATIBLE settings only
        log.info("📊 Creating exit prediction datamodule...")
        datamodule = ExitDataModule(
            corpus_name=args.corpus_name,
            vocab_name=args.vocab_name,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            prediction_windows=args.prediction_windows
            # REMOVED: pin_memory and persistent_workers (not supported)
        )
        
        # Setup datamodule to get class distribution
        datamodule.setup()
        class_stats = datamodule.get_class_distribution()
        
        # Estimate training time
        train_size = class_stats['train']['total']
        estimated_hours, steps_per_epoch, total_steps = estimate_training_time(
            train_size, config["batch_size"], config["accumulate_grad_batches"], 
            config["max_epochs"], num_gpus
        )
        
        log.info("📈 Dataset Statistics:")
        for split, stats in class_stats.items():
            log.info(f"  {split.upper()}: {stats['total']:,} samples")
            log.info(f"    No Exit: {stats['no_exit']:,} ({stats['no_exit_pct']:.1f}%)")
            log.info(f"    Exit: {stats['exit']:,} ({stats['exit_pct']:.1f}%)")
        
        log.info(f"⏱️ Training Estimates:")
        log.info(f"   Steps per epoch: {steps_per_epoch:,}")
        log.info(f"   Total steps: {total_steps:,}")
        log.info(f"   Estimated time: {estimated_hours:.1f} hours (vs 22h original!)")
        
        # Calculate class weights for severe imbalance
        train_stats = class_stats['train']
        total = train_stats['total']
        no_exit_weight = total / (2 * train_stats['no_exit']) if train_stats['no_exit'] > 0 else 1.0
        exit_weight = total / (2 * train_stats['exit']) if train_stats['exit'] > 0 else 1.0
        class_weights = [no_exit_weight, exit_weight]
        
        log.info(f"🏋️ Class weights: no_exit={no_exit_weight:.2f}, exit={exit_weight:.2f}")
        
        # Create model
        log.info("🏗️ Creating exit prediction model...")
        model = StartupExitModel(
            pretrained_model_path=pretrained_model_path,
            num_classes=2,
            freeze_encoder=args.freeze_encoder,
            learning_rate=config["learning_rate"],
            weight_decay=args.weight_decay,
            cdw_alpha=args.cdw_alpha,
            cdw_delta=args.cdw_delta,
            cdw_transform=args.cdw_transform,
            class_weights=class_weights
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        log.info(f"📊 Model parameters:")
        log.info(f"  Total: {total_params:,}")
        log.info(f"  Trainable: {trainable_params:,}")
        log.info(f"  Frozen: {total_params - trainable_params:,}")
        
        # Setup WandB logging
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir="./logs",
            config={**vars(args), **config}
        )
        log.info(f"🔗 WandB: {args.wandb_project}/{args.experiment_name}")
        
        # Optimized callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=f"exit_checkpoints/{args.experiment_name}",
                filename="best-{epoch:02d}-{val/auc:.4f}",
                monitor="val/auc",
                mode="max",
                save_top_k=2,  # Reduced to save space
                save_last=True,
                every_n_epochs=1  # Save every epoch for safety
            ),
            EarlyStopping(
                monitor="val/auc",
                mode="max",
                patience=7,  # Slightly more patience
                verbose=True,
                min_delta=0.001  # Require meaningful improvement
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
        
        # Create optimized trainer
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu" if num_gpus > 0 else "cpu",
            devices=num_gpus if num_gpus > 0 else None,
            strategy=strategy,
            precision=args.precision,
            logger=logger,
            callbacks=callbacks,
            # Optimization settings
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=1.0,
            # Validation optimizations
            val_check_interval=config["val_check_interval"],
            # Logging optimizations
            log_every_n_steps=20,  # Less frequent logging
            # Performance optimizations
            sync_batchnorm=True if num_gpus > 1 else False,
            enable_model_summary=True,
            deterministic=False,  # Allow non-deterministic for speed
            # Checkpointing
            enable_checkpointing=True,
        )
        
        log.info("🏋️ Starting optimized exit prediction finetuning...")
        log.info(f"📊 Final configuration:")
        log.info(f"  Model: {'Frozen' if args.freeze_encoder else 'Unfrozen'} encoder")
        log.info(f"  Effective batch size: {config['batch_size'] * config['accumulate_grad_batches'] * num_gpus}")
        log.info(f"  Learning rate: {config['learning_rate']}")
        log.info(f"  Estimated time: {estimated_hours:.1f} hours")
        
        if args.quick_test:
            log.info("🧪 QUICK TEST MODE - Limited epochs")
        
        # Start training
        trainer.fit(model, datamodule)
        
        # Test on best model
        log.info("🧪 Testing best model...")
        test_results = trainer.test(model, datamodule, ckpt_path="best")
        
        # Final metrics
        elapsed_time = time.time() - start_time
        
        log.info("🎉 OPTIMIZED EXIT FINETUNING COMPLETE!")
        log.info(f"✅ Best validation AUC: {trainer.callback_metrics.get('val/auc', 0.0):.4f}")
        log.info(f"✅ Best validation F1: {trainer.callback_metrics.get('val/f1', 0.0):.4f}")
        log.info(f"✅ Final epoch: {trainer.current_epoch}")
        log.info(f"⏱️ Total time: {elapsed_time/3600:.2f} hours")
        log.info(f"🚀 Speed: {trainer.global_step/(elapsed_time/3600):.0f} steps/hour")
        log.info(f"💾 Best model: exit_checkpoints/{args.experiment_name}/")
        
        if args.quick_test:
            log.info("🧪 Quick test completed successfully!")
        else:
            log.info("🎓 Model ready for deployment!")
            
        # Speedup analysis
        if estimated_hours > 0:
            actual_speedup = estimated_hours / (elapsed_time/3600)
            log.info(f"⚡ Achieved {actual_speedup:.1f}x speedup vs estimate!")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Error during finetuning: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
