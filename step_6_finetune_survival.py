#!/usr/bin/env python3

"""
Step 6: Startup Survival Prediction Finetuning

This script finetunes the pretrained startup2vec model for survival prediction
following life2vec methodology for binary classification.
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dataloaders.survival_datamodule import SurvivalDataModule
from models.survival_model import StartupSurvivalModel

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('survival_finetuning.log')
        ]
    )

def get_pretrained_model_path():
    """Find the pretrained model file"""
    possible_paths = [
        "startup2vec_startup2vec-full-1gpu-512d_final.pt",
        "checkpoints/last.ckpt",
        "checkpoints/last-v2.ckpt",
        "checkpoints/startup2vec-full-1gpu-512d-epoch=02-step=032517.ckpt"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("No pretrained model found. Please ensure you have a trained model.")

def main():
    parser = argparse.ArgumentParser(description="Startup Survival Prediction Finetuning")
    
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
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=20,
                       help="Maximum epochs")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of workers")
    parser.add_argument("--accumulate-grad-batches", type=int, default=4,
                       help="Gradient accumulation batches")
    
    # Hardware
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="Training precision")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default="startup-survival-prediction",
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
    
    log.info("🚀 STARTUP SURVIVAL PREDICTION FINETUNING")
    log.info("=" * 60)
    log.info("Following life2vec methodology for binary classification")
    
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
            args.experiment_name = f"survival-{freeze_str}-{args.batch_size}bs-{args.learning_rate}lr"
        
        # Create datamodule
        log.info("📊 Creating survival prediction datamodule...")
        datamodule = SurvivalDataModule(
            corpus_name=args.corpus_name,
            vocab_name=args.vocab_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prediction_windows=args.prediction_windows
        )
        
        # Setup datamodule to get class distribution
        datamodule.setup()
        class_stats = datamodule.get_class_distribution()
        
        log.info("📈 Class distribution:")
        for split, stats in class_stats.items():
            log.info(f"  {split.upper()}:")
            log.info(f"    Total: {stats['total']:,}")
            log.info(f"    Died: {stats['died']:,} ({stats['died_pct']:.1f}%)")
            log.info(f"    Survived: {stats['survived']:,} ({stats['survived_pct']:.1f}%)")
        
        # Calculate class weights for severe imbalance
        train_stats = class_stats['train']
        total = train_stats['total']
        died_weight = total / (2 * train_stats['died']) if train_stats['died'] > 0 else 1.0
        survived_weight = total / (2 * train_stats['survived']) if train_stats['survived'] > 0 else 1.0
        class_weights = [died_weight, survived_weight]
        
        log.info(f"🏋️ Calculated class weights: died={died_weight:.2f}, survived={survived_weight:.2f}")
        
        # Create model
        log.info("🏗️ Creating survival prediction model...")
        model = StartupSurvivalModel(
            pretrained_model_path=pretrained_model_path,
            num_classes=2,
            freeze_encoder=args.freeze_encoder,
            learning_rate=args.learning_rate,
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
        
        # Setup WandB logging (always use WandB)
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir="./logs",
            config=vars(args)
        )
        log.info(f"🔗 WandB logging: {args.wandb_project}/{args.experiment_name}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=f"survival_checkpoints/{args.experiment_name}",
                filename="best-{epoch:02d}-{val/auc:.4f}",
                monitor="val/auc",
                mode="max",
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor="val/auc",
                mode="max",
                patience=5,
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if args.gpus > 0 else "cpu",
            devices=args.gpus if args.gpus > 0 else None,
            precision=args.precision,
            logger=logger,
            callbacks=callbacks,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            # Validation
            val_check_interval=1.0,
            # Quick test settings
            limit_train_batches=10 if args.quick_test else None,
            limit_val_batches=5 if args.quick_test else None,
            # Logging
            log_every_n_steps=10,
        )
        
        log.info("🏋️ Starting survival prediction finetuning...")
        log.info(f"📊 Configuration:")
        log.info(f"  Encoder: {'Frozen' if args.freeze_encoder else 'Unfrozen'}")
        log.info(f"  Batch size: {args.batch_size} (accumulated: {args.batch_size * args.accumulate_grad_batches})")
        log.info(f"  Learning rate: {args.learning_rate}")
        log.info(f"  Max epochs: {args.max_epochs}")
        log.info(f"  Prediction windows: {args.prediction_windows}")
        log.info(f"  CDW Loss: alpha={args.cdw_alpha}, delta={args.cdw_delta}, transform={args.cdw_transform}")
        
        if args.quick_test:
            log.info("🧪 QUICK TEST MODE - Limited batches")
        
        # Start training
        trainer.fit(model, datamodule)
        
        # Test on best model
        log.info("🧪 Testing best model...")
        trainer.test(model, datamodule, ckpt_path="best")
        
        # Save final metrics
        final_metrics = {
            'best_val_auc': trainer.callback_metrics.get('val/auc', 0.0).item(),
            'best_val_f1': trainer.callback_metrics.get('val/f1', 0.0).item(),
            'final_epoch': trainer.current_epoch,
            'total_training_time': time.time() - start_time
        }
        
        elapsed_time = time.time() - start_time
        
        log.info("🎉 FINETUNING COMPLETE!")
        log.info(f"✅ Best validation AUC: {final_metrics['best_val_auc']:.4f}")
        log.info(f"✅ Best validation F1: {final_metrics['best_val_f1']:.4f}")
        log.info(f"✅ Total epochs: {final_metrics['final_epoch']}")
        log.info(f"⏱️ Total time: {elapsed_time/3600:.2f} hours")
        log.info(f"💾 Checkpoints saved to: survival_checkpoints/{args.experiment_name}/")
        
        if args.quick_test:
            log.info("�� Quick test completed successfully!")
            log.info("💡 Ready for full training with --max-epochs 20")
        else:
            log.info("🎓 Model ready for deployment and analysis!")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Error during finetuning: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
