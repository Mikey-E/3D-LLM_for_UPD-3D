#!/bin/bash
#SBATCH --job-name=inference_3dfront
#SBATCH --output=unified_pipeline/logs/inference_3dfront_%A_%a.out
#SBATCH --error=unified_pipeline/logs/inference_3dfront_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-29

# Inference script for 3D-FRONT dataset
# Splits 2,992 point clouds across 30 array jobs (~100 per job)
# Each point cloud has 12 questions = ~1,200 inferences per job
# Estimated time: ~2-3 seconds per inference = ~40-60 minutes per job

echo "=========================================="
echo "3D-FRONT Inference - Array Job $SLURM_ARRAY_TASK_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Initialize and activate conda environment
source /project/3dllms/melgin/conda/etc/profile.d/conda.sh
conda activate lavis

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Calculate range for this array task
TOTAL_PCLS=2992
TASKS=30
PER_TASK=$((TOTAL_PCLS / TASKS))
START_IDX=$((SLURM_ARRAY_TASK_ID * PER_TASK))

# Last task gets any remaining
if [ $SLURM_ARRAY_TASK_ID -eq $((TASKS - 1)) ]; then
    END_IDX=$TOTAL_PCLS
else
    END_IDX=$((START_IDX + PER_TASK))
fi

echo "Processing point clouds $START_IDX to $((END_IDX - 1))"
echo "Estimated count: $((END_IDX - START_IDX))"
echo "Estimated questions: $((END_IDX - START_IDX)) x 12 = $(((END_IDX - START_IDX) * 12))"
echo ""

# Run inference
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D
python3 unified_pipeline/02_run_inference.py \
    --dataset 3D-FRONT \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --device cuda

echo ""
echo "End time: $(date)"
echo "=========================================="
