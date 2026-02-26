#!/bin/bash
#SBATCH --job-name=preprocess_crops3d_train
#SBATCH --output=./slurm_logs/preprocess_crops3d_train_%A_%a.log
#SBATCH --error=./slurm_logs/preprocess_crops3d_train_%A_%a.log
#SBATCH --time=06:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15%4

# Preprocessing script for Crops3D training+validation sets (821 point clouds)
# Splits across 16 array jobs (~52 per job) with max 4 running simultaneously
# Estimated time: ~10-15 seconds per point cloud = ~10-15 minutes per job

echo "=========================================="
echo "Crops3D Training Data Preprocessing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
fi

conda activate lavis

if [ "$CONDA_DEFAULT_ENV" != "lavis" ]; then
    echo "ERROR: Failed to activate lavis environment"
    exit 1
fi

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Calculate start and end indices for this array task
# 821 point clouds / 16 jobs = ~52 per job
TOTAL_PCLS=821
NUM_JOBS=16
PCLS_PER_JOB=$((TOTAL_PCLS / NUM_JOBS + 1))

START_IDX=$((SLURM_ARRAY_TASK_ID * PCLS_PER_JOB))
END_IDX=$((START_IDX + PCLS_PER_JOB))

# Don't exceed total
if [ $END_IDX -gt $TOTAL_PCLS ]; then
    END_IDX=$TOTAL_PCLS
fi

echo "Processing point clouds $START_IDX to $((END_IDX - 1)) ($(($END_IDX - $START_IDX)) point clouds)"
echo ""

# Run preprocessing using the unified pipeline script
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D

python unified_pipeline/01_preprocess.py \
    --dataset Crops3D_train \
    --start_idx $START_IDX \
    --end_idx $END_IDX

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Array job $SLURM_ARRAY_TASK_ID completed successfully!"
else
    echo "✗ Array job $SLURM_ARRAY_TASK_ID failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
