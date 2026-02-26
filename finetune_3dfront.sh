#!/bin/bash
#SBATCH --account=3dllms
#SBATCH --job-name=3dllm_3dfront_finetune
#SBATCH --output=./slurm_logs/finetune_3dfront_%j.log
#SBATCH --error=./slurm_logs/finetune_3dfront_%j.log
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=192GB
#SBATCH --time=7-00:00:00

echo "=========================================="
echo "3D-LLM 3D-FRONT Finetuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Activate conda environment
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
fi

if [ ! -e "$CONDA_SH" ]; then
    echo "ERROR: $CONDA_SH does not exist."
    exit 1
fi

source "$CONDA_SH"
conda activate lavis

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Change to working directory
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base

# Set environment variables for distributed training
export MASTER_PORT=29500
export MASTER_ADDR=localhost

echo "Running finetuning with 8 GPUs..."
echo "Config: lavis/projects/blip2/train/finetune_3dfront.yaml"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path lavis/projects/blip2/train/finetune_3dfront.yaml

EXITCODE=$?

echo ""
echo "=========================================="
echo "Job finished"
echo "Exit code: $EXITCODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXITCODE
