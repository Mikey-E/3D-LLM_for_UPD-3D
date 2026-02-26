#!/bin/bash
#SBATCH --job-name=preprocess_3dfront_train
#SBATCH --output=./slurm_logs/preprocess_3dfront_train_%A_%a.log
#SBATCH --error=./slurm_logs/preprocess_3dfront_train_%A_%a.log
#SBATCH --time=08:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-31%8

# Preprocessing script for 3D-FRONT training set (8,770 point clouds)
# Splits across 32 array jobs (~274 per job) with max 8 running simultaneously
# Estimated time: ~10-15 seconds per point cloud = ~45-70 minutes per job
# Total estimated time: ~1.5-2.5 hours

echo "=========================================="
echo "3D-FRONT Training Data Preprocessing"
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
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'lavis'"
    exit 1
fi

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# Change to working directory
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D

# Create output directory (will be created by script but good to check)
OUTPUT_DIR="/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run preprocessing for this array task
echo "Running preprocessing for 3D-FRONT_train dataset..."

# Calculate start and end indices for this array task
# Total: 8,770 point clouds, 32 tasks = ~274 per task
TOTAL_PCLS=8770
TOTAL_TASKS=32
PCLS_PER_TASK=$((TOTAL_PCLS / TOTAL_TASKS + 1))
START_IDX=$((SLURM_ARRAY_TASK_ID * PCLS_PER_TASK))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * PCLS_PER_TASK))

# Don't go past the total
if [ $END_IDX -gt $TOTAL_PCLS ]; then
    END_IDX=$TOTAL_PCLS
fi

echo "Array task: $SLURM_ARRAY_TASK_ID / $TOTAL_TASKS"
echo "Processing point clouds: $START_IDX to $END_IDX"
echo "Command: python3 unified_pipeline/01_preprocess.py --dataset 3D-FRONT_train --start_idx $START_IDX --end_idx $END_IDX"
echo ""

python3 unified_pipeline/01_preprocess.py \
    --dataset 3D-FRONT_train \
    --start_idx $START_IDX \
    --end_idx $END_IDX

EXITCODE=$?

echo ""
echo "=========================================="
echo "Job finished"
echo "Exit code: $EXITCODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXITCODE
