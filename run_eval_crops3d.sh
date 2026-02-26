#!/bin/bash
#SBATCH --job-name=eval_crops3d
#SBATCH --output=/project/3dllms/melgin/eval_crops3d_%j.out
#SBATCH --error=/project/3dllms/melgin/eval_crops3d_%j.err
#SBATCH --account=3dllms
#SBATCH --partition=mb-vl40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Activate conda environment
source ~/.bashrc
conda activate lavis

# Change to the working directory
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base

# Run evaluation
echo "Starting Crops3D evaluation on checkpoint_7..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

python -u evaluate.py \
    --cfg-path lavis/projects/blip2/eval/eval_crops3d_checkpoint7.yaml

echo "End time: $(date)"
echo "Evaluation complete!"
