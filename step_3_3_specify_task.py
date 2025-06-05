#!/usr/bin/env python3

"""
Step 3.3: Specify Task for Startup Life2Vec

This script sets up the MLM (Masked Language Modeling) task for startup data.
The task will be used for pre-training the transformer model.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# FIXED: Import 
from src.dataloaders.tasks.pretrain_startup import MLM, MLMEncodedDocument

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def create_mlm_task(
    max_length: int = 512,
    mask_ratio: float = 0.15,
    no_sep: bool = False,
    # Augmentation parameters
    p_sequence_timecut: float = 0.01,
    p_sequence_resample: float = 0.01,
    p_sequence_abspos_noise: float = 0.1,
    p_sequence_hide_background: float = 0.01,
    p_sentence_drop_tokens: float = 0.01,
    shuffle_within_sentences: bool = True,
    smart_masking: bool = False
):
    """
    Create MLM task with specified parameters
    
    Args:
        max_length: Maximum sequence length for the model
        mask_ratio: Fraction of tokens to mask (0.15 is BERT standard)
        no_sep: Whether to exclude [SEP] tokens between sentences
        p_sequence_timecut: Probability of random time cutting
        p_sequence_resample: Probability of resampling events
        p_sequence_abspos_noise: Probability of adding noise to absolute positions
        p_sequence_hide_background: Probability of hiding background info
        p_sentence_drop_tokens: Probability of dropping tokens from sentences
        shuffle_within_sentences: Whether to shuffle tokens within sentences
        smart_masking: Whether to use smart masking (group-aware)
    
    Returns:
        MLM task instance
    """
    
    task = MLM(
        name="mlm_startup",
        max_length=max_length,
        no_sep=no_sep,
        # Augmentation parameters
        p_sequence_timecut=p_sequence_timecut,
        p_sequence_resample=p_sequence_resample,
        p_sequence_abspos_noise=p_sequence_abspos_noise,
        p_sequence_hide_background=p_sequence_hide_background,
        p_sentence_drop_tokens=p_sentence_drop_tokens,
        shuffle_within_sentences=shuffle_within_sentences,
        # MLM specific parameters
        mask_ratio=mask_ratio,
        smart_masking=smart_masking
    )
    
    return task

def validate_task(task):
    """
    Validate the task configuration
    
    Args:
        task: MLM task instance
    """
    log = logging.getLogger(__name__)
    
    log.info("🔍 Validating task configuration...")
    
    # Check task parameters
    assert 0.0 < task.mask_ratio < 1.0, f"Invalid mask_ratio: {task.mask_ratio}"
    assert task.max_length > 0, f"Invalid max_length: {task.max_length}"
    assert 0.0 <= task.p_sequence_timecut <= 1.0, f"Invalid p_sequence_timecut: {task.p_sequence_timecut}"
    assert 0.0 <= task.p_sequence_resample <= 1.0, f"Invalid p_sequence_resample: {task.p_sequence_resample}"
    assert 0.0 <= task.p_sequence_abspos_noise <= 1.0, f"Invalid p_sequence_abspos_noise: {task.p_sequence_abspos_noise}"
    assert 0.0 <= task.p_sequence_hide_background <= 1.0, f"Invalid p_sequence_hide_background: {task.p_sequence_hide_background}"
    assert 0.0 <= task.p_sentence_drop_tokens <= 1.0, f"Invalid p_sentence_drop_tokens: {task.p_sentence_drop_tokens}"
    
    log.info("✅ Task configuration is valid")
    
    # Log task summary
    log.info("📋 TASK CONFIGURATION SUMMARY:")
    log.info(f"   📛 Name: {task.name}")
    log.info(f"   📏 Max length: {task.max_length}")
    log.info(f"   🎭 Mask ratio: {task.mask_ratio}")
    log.info(f"   🔤 No SEP tokens: {task.no_sep}")
    log.info(f"   🎲 Smart masking: {task.smart_masking}")
    log.info(f"   🔀 Shuffle within sentences: {task.shuffle_within_sentences}")
    log.info("   🎛️  AUGMENTATION PROBABILITIES:")
    log.info(f"      ✂️  Time cut: {task.p_sequence_timecut}")
    log.info(f"      🎯 Resample: {task.p_sequence_resample}")
    log.info(f"      📍 Position noise: {task.p_sequence_abspos_noise}")
    log.info(f"      🙈 Hide background: {task.p_sequence_hide_background}")
    log.info(f"      💧 Drop tokens: {task.p_sentence_drop_tokens}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 3.3: Specify MLM Task for Startup Data")
    
    # Task configuration
    parser.add_argument("--max-length", type=int, default=512, 
                       help="Maximum sequence length (default: 512)")
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                       help="Fraction of tokens to mask (default: 0.15)")
    parser.add_argument("--no-sep", action="store_true",
                       help="Exclude [SEP] tokens between sentences")
    parser.add_argument("--smart-masking", action="store_true",
                       help="Enable smart masking (group-aware)")
    
    # Augmentation parameters
    parser.add_argument("--p-timecut", type=float, default=0.01,
                       help="Probability of sequence time cutting (default: 0.01)")
    parser.add_argument("--p-resample", type=float, default=0.01, 
                       help="Probability of sequence resampling (default: 0.01)")
    parser.add_argument("--p-pos-noise", type=float, default=0.1,
                       help="Probability of position noise (default: 0.1)")
    parser.add_argument("--p-hide-bg", type=float, default=0.01,
                       help="Probability of hiding background (default: 0.01)")
    parser.add_argument("--p-drop-tokens", type=float, default=0.01,
                       help="Probability of dropping tokens (default: 0.01)")
    parser.add_argument("--no-shuffle", action="store_true",
                       help="Disable shuffling within sentences")
    
    # Control flags
    parser.add_argument("--run", action="store_true", required=True,
                       help="Actually run the task creation (safety flag)")
    parser.add_argument("--output-info", action="store_true",
                       help="Output detailed task information")
    
    args = parser.parse_args()
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    start_time = time.time()
    
    log.info("🚀 CREATING STARTUP MLM TASK")
    log.info("=" * 60)
    
    try:
        # Create the task
        log.info("🔧 Setting up MLM task...")
        task = create_mlm_task(
            max_length=args.max_length,
            mask_ratio=args.mask_ratio,
            no_sep=args.no_sep,
            p_sequence_timecut=args.p_timecut,
            p_sequence_resample=args.p_resample,
            p_sequence_abspos_noise=args.p_pos_noise,
            p_sequence_hide_background=args.p_hide_bg,
            p_sentence_drop_tokens=args.p_drop_tokens,
            shuffle_within_sentences=not args.no_shuffle,
            smart_masking=args.smart_masking
        )
        
        # Validate the task
        validate_task(task)
        
        if args.output_info:
            log.info("📊 DETAILED TASK INFORMATION:")
            log.info(f"   🏷️  Task class: {task.__class__.__name__}")
            log.info(f"   📁 Task module: {task.__class__.__module__}")
            log.info(f"   🔢 Expected vocab requirements: GENERAL, MONTH, YEAR, plus startup-specific tokens")
            log.info(f"   📝 Encoded document type: {MLMEncodedDocument.__name__}")
        
        # Success message
        processing_time = time.time() - start_time
        log.info("🎉 TASK CREATION COMPLETE!")
        log.info(f"✅ MLM task '{task.name}' ready for datamodule integration")
        log.info(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        log.info("")
        log.info("🎉 SUCCESS!")
        log.info("✅ Task ready for datamodule and training!")
        log.info("🚀 Next step: Create datamodule and start training")

        # Quick usage example
        log.info("")
        log.info("💡 USAGE EXAMPLE:")
        log.info("   from src.dataloaders.tasks.pretrain import MLM")
        log.info(f"   task = MLM(name='{task.name}', max_length={task.max_length}, mask_ratio={task.mask_ratio})")
        
        return 0

    except Exception as e:
        log.error(f"❌ Error creating task: {str(e)}")
        log.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
