#!/bin/bash
#SBATCH --job-name=3dfront_base_openended
#SBATCH --account=3dllms
#SBATCH --output=unified_pipeline/logs/3dfront_base_openended_%j.out
#SBATCH --error=unified_pipeline/logs/3dfront_base_openended_%j.err
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Redo base model inference for 3D-FRONT open_ended category only
# This recreates inf_rslts_3dllm_3D-FRONT_test_open_ended.json

cd /project/3dllms/melgin/3D-LLM_for_UPD-3D

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lavis

echo "=========================================="
echo "Starting 3D-FRONT Base Model Inference (open_ended only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Run inference only for open_ended category
python unified_pipeline/02_run_inference.py \
    --dataset 3D-FRONT \
    --checkpoint /project/3dllms/melgin/3D-LLM_for_UPD-3D/checkpoints/pretrain_blip2_sam_flant5xl_v2.pth \
    --model_name 3dllm_base \
    --category open_ended

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
