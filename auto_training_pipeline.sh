#!/bin/bash

# SIMPLIFIED AUTOMATED TRAINING PIPELINE
# GPUs: 0 (L40S), 1 (L40S), 3 (A100)
# Directory: /data/kebl8110/life2vec_startup_project
# No complex time calculations - just get the job done!

set -e  # Exit on any error

# Configuration - A100 ONLY (safest approach)
PRETRAIN_GPUS="3"      # Only A100 (80GB, plenty of memory)
FINETUNE_GPU="3"       # Same A100
LOG_FILE="training_pipeline_$(date +%Y%m%d_%H%M%S).log"
PROJECT_DIR="/data/kebl8110/life2vec_startup_project"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# Check GPUs
check_gpus() {
    log "🔍 Checking GPU availability..."
    
    for gpu in 3; do
        if ! nvidia-smi -i $gpu &>/dev/null; then
            error "GPU $gpu not available!"
            return 1
        fi
    done
    
    log "✅ GPU 3 (A100) is available and has plenty of memory"
    return 0
}

# Show GPU status
show_gpu_status() {
    log "📊 Current GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F, '$1 ~ /^(0|1|3)$/ {printf "   GPU %s (%s): %s%% util, %s/%sMB\n", $1, $2, $3, $4, $5}'
}

# Main pipeline
run_pipeline() {
    local start_time=$(date)
    
    log "🚀 STARTING SIMPLIFIED TRAINING PIPELINE"
    log "========================================"
    log "🕐 Started at: $start_time"
    log "📁 Directory: $PROJECT_DIR"
    log "📝 Log file: $LOG_FILE"
    log "🎯 Pretraining: GPUs $PRETRAIN_GPUS"
    log "🎯 Finetuning: GPU $FINETUNE_GPU (A100)"
    
    # Change to project directory
    cd "$PROJECT_DIR" || {
        error "Failed to change to project directory: $PROJECT_DIR"
        return 1
    }
    
    # Check GPUs
    if ! check_gpus; then
        error "GPU check failed"
        return 1
    fi
    
    # Verify scripts exist
    if [[ ! -f "step_5_train_startup2vec.py" ]] || [[ ! -f "step_6_finetune_survival.py" ]]; then
        error "Training scripts not found in $PROJECT_DIR"
        return 1
    fi
    
    log "✅ All checks passed, starting training..."
    
    # PHASE 1: PRETRAINING
    log ""
    log "🔥 PHASE 1: PRETRAINING"
    log "======================="
    log "🎯 Using GPU: $PRETRAIN_GPUS (A100 only - 80GB, no memory conflicts)"
    log "📊 Batch size: 400 (optimized for single A100)"
    log "👷 Workers: 24"
    log "🧠 Why A100 only? L40S GPUs have competing processes using memory"
    
    show_gpu_status
    
    # Quick test modifications
    local epochs=15
    local quick_flag=""
    
    if [ "$QUICK_TEST" = true ]; then
        epochs=2
        quick_flag="--quick-test"
        log "🧪 QUICK TEST: 2 epochs with quick-test flag"
    fi

    CUDA_VISIBLE_DEVICES=$PRETRAIN_GPUS python step_5_train_startup2vec.py \
        --max-epochs $epochs \
        --use-wandb \
        --batch-size 400 \
        --num-workers 24 \
        --single-gpu \
        $quick_flag \
        2>&1 | tee -a "$LOG_FILE" &
    
    local pretrain_pid=$!
    log "🏃 Pretraining started (PID: $pretrain_pid)"
    
    # Simple monitoring - check every 10 minutes
    while kill -0 $pretrain_pid 2>/dev/null; do
        sleep 600  # 10 minutes
        log "⏱️ Pretraining still running..."
        show_gpu_status
    done
    
    # Check result
    wait $pretrain_pid
    local pretrain_exit_code=$?
    
    if [ $pretrain_exit_code -ne 0 ]; then
        error "Pretraining failed with exit code $pretrain_exit_code"
        return 1
    fi
    
    success "✅ Pretraining completed successfully!"
    
    # FIND PRETRAINED MODEL
    log ""
    log "🔍 FINDING PRETRAINED MODEL"
    log "=========================="
    
    local pretrained_model=""
    
    # Look for model files in order of preference
    declare -a patterns=(
        "startup2vec_*3gpu*_final.pt"
        "startup2vec_*final.pt"
        "checkpoints/last.ckpt"
        "checkpoints/*3gpu*.ckpt" 
        "checkpoints/*epoch*.ckpt"
    )
    
    for pattern in "${patterns[@]}"; do
        local found=$(find . -name "$pattern" -type f 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            pretrained_model="$found"
            break
        fi
    done
    
    if [ -z "$pretrained_model" ]; then
        error "No pretrained model found!"
        log "📁 Available files:"
        find . -name "*.pt" -o -name "*.ckpt" | head -10
        return 1
    fi
    
    log "✅ Found pretrained model: $pretrained_model"
    
    # PHASE 2: FINETUNING
    log ""
    log "🎯 PHASE 2: FINETUNING" 
    log "====================="
    log "🎯 Using GPU: $FINETUNE_GPU (A100 - single GPU optimized)"
    log "📊 Batch size: 64 (optimized for A100)"
    log "👷 Workers: 16"
    log "🧠 Why single GPU? Better convergence + no communication overhead"
    
    show_gpu_status
    
    CUDA_VISIBLE_DEVICES=$FINETUNE_GPU python step_6_finetune_survival.py \
        --pretrained-model "$pretrained_model" \
        --batch-size 64 \
        --max-epochs 15 \
        --learning-rate 1e-4 \
        --accumulate-grad-batches 1 \
        --num-workers 16 \
        --wandb-project "startup-survival-prediction" \
        --experiment-name "single-a100-optimized-$(date +%Y%m%d_%H%M%S)" \
        --run \
        2>&1 | tee -a "$LOG_FILE" &
    
    local finetune_pid=$!
    log "🏃 Finetuning started (PID: $finetune_pid)"
    
    # Simple monitoring
    while kill -0 $finetune_pid 2>/dev/null; do
        sleep 300  # 5 minutes
        log "⏱️ Finetuning still running..."
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F, '$1 == "3" {printf "   GPU %s (%s): %s%% util, %s/%sMB\n", $1, $2, $3, $4, $5}'
    done
    
    # Check result
    wait $finetune_pid
    local finetune_exit_code=$?
    
    if [ $finetune_exit_code -ne 0 ]; then
        error "Finetuning failed with exit code $finetune_exit_code"
        return 1
    fi
    
    success "✅ Finetuning completed successfully!"
    
    # FINAL SUMMARY
    local end_time=$(date)
    
    log ""
    log "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
    log "=================================="
    log "🕐 Started: $start_time"
    log "🕐 Finished: $end_time"
    log "📁 Working directory: $PROJECT_DIR"
    log "📝 Full log: $LOG_FILE"
    
    log ""
    log "📁 Generated files:"
    find . -name "startup2vec_*final.pt" -o -name "survival_checkpoints" -type d 2>/dev/null | while read file; do
        log "   ✅ $file"
    done
    
    log ""
    log "🎓 READY FOR ACTION!"
    log "Use your trained models for startup survival prediction!"
    
    return 0
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up background processes..."
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Parse arguments
QUICK_TEST=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --multi-gpu-finetune)
            FINETUNE_GPU="0,1,3"
            log "🔄 Multi-GPU finetuning enabled"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --quick-test: Run quick test mode"
            echo "  --dry-run: Show commands without running"
            echo "  --multi-gpu-finetune: Use all 3 GPUs for finetuning (not recommended)"
            echo "  --help: Show this help"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    log "🔍 DRY RUN MODE - Commands that would be executed:"
    log ""
    log "1. Pretraining:"
    log "   CUDA_VISIBLE_DEVICES=$PRETRAIN_GPUS python step_5_train_startup2vec.py \\"
    log "     --max-epochs 15 --use-wandb --batch-size 280 --num-workers 24"
    log ""
    log "2. Finetuning:"
    log "   CUDA_VISIBLE_DEVICES=$FINETUNE_GPU python step_6_finetune_survival.py \\"
    log "     --batch-size 64 --max-epochs 15 --learning-rate 1e-4 \\"
    log "     --accumulate-grad-batches 1 --num-workers 16 --run"
    log ""
    log "💡 Remove --dry-run to actually execute"
    exit 0
fi

# Quick test mode
if [ "$QUICK_TEST" = true ]; then
    log "🧪 QUICK TEST MODE - 1 epoch, limited batches for testing"
else
    log "💪 FULL TRAINING MODE - 15 epochs"
fi

# Verify environment
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found"
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Run the pipeline
log "🚀 Starting pipeline execution..."
run_pipeline
exit_code=$?

if [ $exit_code -eq 0 ]; then
    success "🎉 MISSION ACCOMPLISHED!"
    log "Check your trained models in: $PROJECT_DIR"
else
    error "💥 Pipeline failed - check log: $LOG_FILE"
fi

exit $exit_code