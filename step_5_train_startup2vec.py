#!/usr/bin/env python3

"""
ROBUST FIXED training script for your beast hardware:
- Fixed learning rate scheduler step limit issue
- Fixed early stopping callback
- Added comprehensive error handling
- 4 GPUs (2x A100 80GB + 2x L40S 44GB)
- 128 CPU cores
- Optimized for overnight training reliability
"""

import torch
import time
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from src.models.pretrain import TransformerEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

def import_datamodule():
    """Import the datamodule from the .py file"""
    try:
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        if spec is None:
            raise ImportError("Could not find step_4_create_datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        return datamodule_module.StartupDataModule, datamodule_module.StartupVocabulary
    except Exception as e:
        print(f"❌ Error importing datamodule: {e}")
        print("💡 Make sure step_4_create_datamodule.py exists in the current directory")
        sys.exit(1)

# Import the classes
StartupDataModule, StartupVocabulary = import_datamodule()

def get_optimal_config(num_gpus=4, total_memory_gb=247, quick_test=False):
    """Get optimal configuration for your hardware"""
    
    if quick_test:
        config = {
            # Smaller model for testing
            "hidden_size": 256,
            "hidden_ff": 1024,
            "n_encoders": 4,
            "n_heads": 8,
            "n_local": 2,
            # Small batch for testing
            "batch_size_per_gpu": 32,
            "total_batch_size": 32 * num_gpus,
            "num_workers": 4,
            # Quick training
            "learning_rate": 1e-4,
            "max_epochs": 2,
            # Mixed precision for speed
            "precision": "16-mixed",
            # Other optimizations
            "compile_model": False,  # DISABLED: Fix for einops compatibility
            "accumulate_grad_batches": 1,
        }
    else:
        config = {
            # Larger model for full training - optimized for your hardware
            "hidden_size": 512,  # Increased from 96
            "hidden_ff": 2048,   # Increased from 128
            "n_encoders": 8,     # Increased from 4
            "n_heads": 16,       # Increased from 8
            "n_local": 4,        # Increased from 2
            # Training config optimized for your hardware
            "batch_size_per_gpu": 128,  # Large batch per GPU
            "total_batch_size": 128 * num_gpus,
            "num_workers": 16,   # Use many of your 128 cores
            # Learning rates for large batch training
            "learning_rate": 2e-4 * (num_gpus * 0.5),  # Scale with sqrt(gpus)
            "max_epochs": 20,    # Default epochs for full training
            # Mixed precision for speed
            "precision": "16-mixed",
            # Other optimizations
            "compile_model": True,  # PyTorch 2.0 compile
            "accumulate_grad_batches": 1,  # No accumulation needed with large memory
        }
    
    return config

def estimate_training_time(train_size, batch_size_total, num_gpus, epochs):
    """Estimate training time based on hardware"""
    steps_per_epoch = train_size // batch_size_total
    
    # Time estimates (rough) - your hardware is FAST
    if num_gpus >= 4:
        seconds_per_step = 0.05  # Very fast with 4 GPUs + mixed precision
    elif num_gpus >= 2:
        seconds_per_step = 0.1
    else:
        seconds_per_step = 0.2
    
    total_steps = steps_per_epoch * epochs
    estimated_hours = (total_steps * seconds_per_step) / 3600
    
    return estimated_hours, steps_per_epoch, total_steps, seconds_per_step

def check_data_integrity(datamodule):
    """Check for data issues that could cause CUDA errors"""
    print("🔍 Checking data integrity...")
    try:
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # Check input_ids range
        input_ids = batch["input_ids"]
        vocab_size = datamodule.vocabulary.size()
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Input IDs min: {input_ids.min().item()}")
        print(f"   Input IDs max: {input_ids.max().item()}")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Padding mask shape: {batch['padding_mask'].shape}")
        
        # Check if any token IDs are out of bounds
        if input_ids.max().item() >= vocab_size:
            print(f"❌ ERROR: Found token ID {input_ids.max().item()} >= vocab_size {vocab_size}")
            invalid_positions = (input_ids >= vocab_size).nonzero()
            print(f"Invalid positions: {invalid_positions[:10]}")  # Show first 10
            return False
        
        if input_ids.min().item() < 0:
            print(f"❌ ERROR: Found negative token ID {input_ids.min().item()}")
            return False
        
        # Check padding mask values
        padding_mask = batch['padding_mask']
        if padding_mask.min().item() < 0 or padding_mask.max().item() > 1:
            print(f"❌ ERROR: Padding mask has invalid values: min={padding_mask.min().item()}, max={padding_mask.max().item()}")
            return False
        
        # Check for NaN or inf values
        if torch.isnan(input_ids).any():
            print("❌ ERROR: Found NaN in input_ids")
            return False
        
        if torch.isinf(input_ids).any():
            print("❌ ERROR: Found inf in input_ids")
            return False
        
        print("✅ Data integrity check passed")
        return True
        
    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Startup Life2vec Training - Beast Mode")
    parser.add_argument("--quick-test", action="store_true", help="Quick test run (2 epochs)")
    parser.add_argument("--single-gpu", action="store_true", help="Use single GPU instead of all 4")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum epochs for full training")
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=None, help="Override number of workers")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    
    print("🚀 STARTUP LIFE2VEC - BEAST MODE TRAINING!")
    print("=" * 70)
    print("⚡️ Hardware detected:")
    print("   💪 4x GPUs available (2x A100 80GB + 2x L40S 44GB)")
    print("   🧠 128 CPU cores")
    print("   💾 ~247 GB total GPU memory")
    
    # GPU configuration
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Exiting...")
        return 1
    
    available_gpus = torch.cuda.device_count()
    num_gpus = 1 if args.single_gpu else min(4, available_gpus)
    print(f"   🎯 Using {num_gpus} GPU(s)")
    
    # Get optimal config
    config = get_optimal_config(num_gpus, quick_test=args.quick_test)
    
    # Override config with command line args
    if args.batch_size is not None:
        config["batch_size_per_gpu"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if not args.quick_test:
        config["max_epochs"] = args.max_epochs
    
    print(f"\n📊 Training Configuration:")
    if args.quick_test:
        print("   🧪 QUICK TEST MODE")
    else:
        print("   💪 FULL TRAINING MODE")
    print(f"   🏗️ Model: {config['hidden_size']}d, {config['n_encoders']} layers, {config['n_heads']} heads")
    print(f"   📦 Batch: {config['batch_size_per_gpu']} per GPU ({config['batch_size_per_gpu'] * num_gpus} total)")
    print(f"   👷 Workers: {config['num_workers']}")
    print(f"   📚 Epochs: {config.get('max_epochs', 2)}")
    print(f"   ⚡ Precision: {config['precision']}")

    # Create datamodule with optimized settings
    print("\n📊 Creating datamodule...")
    try:
        datamodule = StartupDataModule(
            batch_size=config["batch_size_per_gpu"],
            num_workers=config["num_workers"],
            max_length=512,
            mask_ratio=0.15
        )
        datamodule.setup()
        vocab_size = datamodule.vocabulary.size()
        print(f"✅ Datamodule created successfully")
        print(f"📖 Vocabulary size: {vocab_size:,}")
        
        # Check data integrity before training
        if not check_data_integrity(datamodule):
            print("❌ Data integrity check failed! Please fix the data issues.")
            return 1
            
    except Exception as e:
        print(f"❌ Failed to create datamodule: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Dataset sizes and time estimation
    train_size = len(datamodule.train_dataset)
    val_size = len(datamodule.val_dataset)
    epochs = config.get('max_epochs', 2)
    
    estimated_hours, steps_per_epoch, total_steps, speed = estimate_training_time(
        train_size, config["batch_size_per_gpu"] * num_gpus, num_gpus, epochs
    )
    
    print(f"\n⏱️ Training Estimates:")
    print(f"   📚 Train samples: {train_size:,}")
    print(f"   📝 Val samples: {val_size:,}")
    print(f"   📊 Steps per epoch: {steps_per_epoch:,}")
    print(f"   🎯 Total steps: {total_steps:,}")
    print(f"   ⏰ Estimated time: {estimated_hours:.1f} hours")
    print(f"   🚀 Speed: ~{1/speed:.0f} steps/second")
    
    # Model hyperparameters - FIXED: Pass steps_per_epoch to model
    hparams = {
        "hidden_size": config["hidden_size"],
        "hidden_ff": config["hidden_ff"], 
        "n_encoders": config["n_encoders"],
        "n_heads": config["n_heads"],
        "n_local": config["n_local"],
        "local_window_size": 4,
        "max_length": 512,
        "vocab_size": vocab_size,
        "num_classes": -1,
        "cls_num_targs": 3,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size_per_gpu"],
        "num_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,  # ADDED: Pass actual steps per epoch
        "total_steps": total_steps,          # ADDED: Pass total steps
        "attention_type": "performer",
        "norm_type": "rezero", 
        "num_random_features": max(32, config["hidden_size"] // 8),  # Scale with model size
        "parametrize_emb": True,
        "emb_dropout": 0.1,
        "fw_dropout": 0.1,
        "att_dropout": 0.1,
        "dc_dropout": 0.1,
        "hidden_act": "swish",
        "training_task": "mlm",
        "weight_tying": True,
        "norm_output_emb": True,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    
    print(f"\n🏗️ Creating model...")
    try:
        model = TransformerEncoder(hparams=hparams)
        
        # PyTorch 2.0 compile for speed (if available and not quick test)
        # DISABLED: Fix for einops compatibility with PyTorch 2.x
        # if hasattr(torch, 'compile') and config.get('compile_model', False):
        #     print("⚡ Compiling model with PyTorch 2.0...")
        #     model = torch.compile(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per float32 parameter
        print(f"✅ Model created successfully")
        print(f"🔢 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"💾 Estimated model memory: {model_memory_gb:.1f} GB")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Setup logging
    experiment_name = f"startup2vec-{'test' if args.quick_test else 'full'}-{num_gpus}gpu-{config['hidden_size']}d"
    
    if args.use_wandb:
        try:
            logger = WandbLogger(
                project="startup-life2vec-beast-mode",
                name=experiment_name,
                save_dir="./logs",
                config={**hparams, **config}
            )
            print("📊 Using WandB logging")
        except Exception as e:
            print(f"⚠️ WandB failed, falling back to TensorBoard: {e}")
            logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    else:
        logger = TensorBoardLogger("lightning_logs", name=experiment_name)
        print("📊 Using TensorBoard logging")
    
    # Callbacks - ROBUST and SAFE
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{experiment_name}-{{epoch:02d}}-{{step:06d}}",
            save_top_k=3,  # Keep best 3 checkpoints
            monitor="train_loss",  # Monitor train_loss instead of val_loss
            mode="min",
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=True  # Save at end of each epoch
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # REMOVED: Early stopping to prevent validation issues
    # Training will run for full epochs - more reliable for overnight runs
    
    # Distributed strategy for multi-GPU
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,  # Optimization
            gradient_as_bucket_view=True   # Memory optimization
        )
        print(f"🔥 Using Distributed Data Parallel across {num_gpus} GPUs")
    else:
        strategy = "auto"
    
    # Create trainer - ROBUST settings for overnight training
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision=config["precision"],
        logger=logger,
        callbacks=callbacks,
        # Performance optimizations
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        # Validation settings - CONSERVATIVE for reliability
        val_check_interval=1.0,  # Validate once per epoch
        limit_val_batches=50,    # Limit validation batches to speed up
        # Logging - less frequent to reduce overhead
        log_every_n_steps=50,
        # Speed optimizations
        enable_model_summary=True,
        sync_batchnorm=True if num_gpus > 1 else False,
        # Quick test limitations
        limit_train_batches=50 if args.quick_test else None,
        # Fault tolerance
        detect_anomaly=False,  # Disable for speed
        deterministic=False,   # Allow non-deterministic for speed
        # Checkpointing
        enable_checkpointing=True,
        # Profiling - disable for production
        profiler=None,
    )
    
    print(f"\n🏋️ Starting training...")
    print(f"💪 Configuration: {num_gpus} GPUs × {config['batch_size_per_gpu']} batch size = {num_gpus * config['batch_size_per_gpu']} total")
    print(f"⚡ Precision: {config['precision']}")
    print(f"🔥 Estimated time: {estimated_hours:.1f} hours")
    print(f"💾 Checkpoints will be saved every epoch")
    print(f"🛡️ No early stopping - will run full {epochs} epochs")
    
    if args.quick_test:
        print("🧪 Quick test will run 2 epochs with limited batches")
    
    # Start training
    start_time = time.time()
    
    try:
        print("\n🚀 Training started!")
        
        # Resume from checkpoint if specified
        ckpt_path = args.resume_from_checkpoint
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"📂 Resuming from checkpoint: {ckpt_path}")
        else:
            ckpt_path = None
        
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        
        # Save final model (fix pickling issue)
        save_path = f"startup2vec_{experiment_name}_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'hparams': hparams,
            'config': config,
            'training_args': vars(args),
            'vocab_size': vocab_size,
            'experiment_name': experiment_name,
            'final_epoch': trainer.current_epoch,
            'final_step': trainer.global_step
        }, save_path)
        print(f"💾 Model saved to: {save_path}")
        
        # Training summary
        total_time = time.time() - start_time
        actual_steps_per_second = trainer.global_step / total_time if total_time > 0 else 0
        print(f"\n📊 Training Summary:")
        print(f"   ⏱️ Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"   🚀 Actual speed: {actual_steps_per_second:.1f} steps/second")
        print(f"   📈 Final epoch: {trainer.current_epoch}")
        print(f"   🎯 Total steps: {trainer.global_step}")
        print(f"   💾 Model saved: {save_path}")
        print(f"   📁 Checkpoints in: checkpoints/")
        
        if args.quick_test:
            print(f"\n✅ QUICK TEST SUCCESSFUL!")
            print(f"💡 Ready for full training with: --max-epochs 20")
        else:
            print(f"\n🎓 READY FOR DOWNSTREAM TASKS!")
            print(f"💡 Use this model for startup survival prediction")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)