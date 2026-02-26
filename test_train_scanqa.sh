#!/bin/bash
#SBATCH --job-name=3dllm_scanqa_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=48G
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --output=./slurm_logs/test_scanqa_%j.log

echo "=========================================="
echo "3D-LLM ScanQA Test Training (Single GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment: lavis"
eval "$(conda shell.bash hook)"
conda activate lavis
echo "✓ Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# Change to project directory
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base || exit 1
echo "Working directory: $(pwd)"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check config file exists
CONFIG_FILE="lavis/projects/blip2/train/finetune_scanqa.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Config file: $CONFIG_FILE"
else
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo ""

# Test data loading first
echo "=========================================="
echo "Testing Data Loading..."
echo "=========================================="
python -c "
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from lavis.datasets.builders import *

cfg = OmegaConf.load('$CONFIG_FILE')
print('Config loaded successfully')
print('Dataset config:', cfg.datasets)
"
echo ""

# Run training with verbose output
echo "=========================================="
echo "Starting Single GPU Test Training..."
echo "Command: python -u train.py --cfg-path $CONFIG_FILE"
echo "=========================================="
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Set distributed training environment variables for single GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

# Run with single GPU (no distributed training)
python -u train.py --cfg-path "$CONFIG_FILE" 2>&1

echo ""
echo "=========================================="
echo "Training completed/terminated"
echo "End time: $(date)"
echo "=========================================="
