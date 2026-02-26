#!/bin/bash
#SBATCH --job-name=preprocess_giw529_missing
#SBATCH --output=./slurm_logs/preprocess_giw529_missing_%j.log
#SBATCH --error=./slurm_logs/preprocess_giw529_missing_%j.log
#SBATCH --time=00:10:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "GIW529 Missing Scene Preprocessing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source /project/3dllms/melgin/conda/etc/profile.d/conda.sh
conda activate lavis

cd /project/3dllms/melgin/3D-LLM_for_UPD-3D

echo "Processing missing scene: Vinegar@04_19_2024_S2_F_Vinegar_P_3"
python3 unified_pipeline/01_preprocess.py --dataset GIW529 --start_idx 528 --end_idx 529

echo "Job finished at $(date)"
