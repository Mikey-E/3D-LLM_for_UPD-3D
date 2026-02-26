#!/bin/bash
# SLURM batch script for 3D-LLM ScanQA Finetuning
# Finetunes BLIP2 model on ScanQA dataset (24,969 samples)
# Example use: sbatch finetune_scanqa.sh

#SBATCH --account=3dllms
#SBATCH --job-name=3dllm_scanqa_finetune
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=96G
#SBATCH --partition=mb-l40s
#SBATCH --output=./slurm_logs/finetune_scanqa_%j.log
#SBATCH --error=./slurm_logs/finetune_scanqa_%j.log

echo "=========================================="
echo "3D-LLM ScanQA Finetuning"
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
CONFIG="lavis/projects/blip2/train/finetune_scanqa.yaml"

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
echo "✓ Config file: $CONFIG"
echo ""

# Set up distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "Distributed Training Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  World Size: 8 GPUs"
echo ""

# Start finetuning
echo "=========================================="
echo "Starting ScanQA Finetuning..."
echo "Command: python train.py --cfg-path $CONFIG"
echo "=========================================="
echo ""

python train.py --cfg-path "$CONFIG"

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
