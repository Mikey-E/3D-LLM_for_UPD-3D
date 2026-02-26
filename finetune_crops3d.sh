#!/bin/bash
# SLURM batch script for 3D-LLM Crops3D Finetuning
# Finetunes BLIP2 model on Crops3D dataset (7,476 training samples)
# Example use: sbatch finetune_crops3d.sh

#SBATCH --account=3dllms
#SBATCH --job-name=3dllm_crops3d_finetune
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=192GB
#SBATCH --partition=mb-vl40s
#SBATCH --nodelist=vl40s-005
#SBATCH --output=./slurm_logs/finetune_crops3d_%j.log
#SBATCH --error=./slurm_logs/finetune_crops3d_%j.log

echo "=========================================="
echo "3D-LLM Crops3D Finetuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# This ensures "conda activate <env>" works in non-interactive shells.
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
    echo "WARNING: CONDA_INSTALL_PATH is not set. Trying $CONDA_SH"
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
fi

# Activate the lavis environment
echo "Activating conda environment: lavis"
conda activate lavis

# Verify conda environment
if [ "$CONDA_DEFAULT_ENV" != "lavis" ]; then
    echo "ERROR: Failed to activate lavis environment"
    exit 1
fi
echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Navigate to working directory
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base
echo "Working directory: $(pwd)"
echo ""

# Verify GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Set config file
CONFIG="lavis/projects/blip2/train/finetune_crops3d.yaml"

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
echo "✓ Config file: $CONFIG"
echo ""

# Verify training data exists
TRAIN_JSON="/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/Crops3D/Crops3D_train.json"
VAL_JSON="/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/Crops3D/Crops3D_val.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: Training data not found: $TRAIN_JSON"
    exit 1
fi
echo "✓ Training data: $TRAIN_JSON"

if [ ! -f "$VAL_JSON" ]; then
    echo "ERROR: Validation data not found: $VAL_JSON"
    exit 1
fi
echo "✓ Validation data: $VAL_JSON"
echo ""

# Verify pretrained checkpoint exists
CHECKPOINT="/project/3dllms/melgin/3D-LLM_for_UPD-3D/checkpoints/pretrain_blip2_sam_flant5xl_v2.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "✓ Pretrained checkpoint: $CHECKPOINT"
echo ""

# Set up distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Distributed Training Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  GPUs: 4"
echo ""

echo "Dataset Information:"
TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('$TRAIN_JSON'))))")
VAL_COUNT=$(python3 -c "import json; print(len(json.load(open('$VAL_JSON'))))")
echo "  Training samples: $TRAIN_COUNT"
echo "  Validation samples: $VAL_COUNT"
echo ""

# Start finetuning with torchrun for proper distributed training
echo "=========================================="
echo "Starting Crops3D Finetuning..."
echo "Command: torchrun --nnodes=1 --nproc_per_node=4 train.py --cfg-path $CONFIG"
echo "=========================================="
echo ""

torchrun --nnodes=1 --nproc_per_node=4 train.py --cfg-path "$CONFIG"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Finetuning completed successfully!"
else
    echo "✗ Finetuning failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
