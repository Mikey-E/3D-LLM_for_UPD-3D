#!/bin/bash
#SBATCH --job-name=preprocess_3dfront_val
#SBATCH --output=./slurm_logs/preprocess_3dfront_val_%j.log
#SBATCH --error=./slurm_logs/preprocess_3dfront_val_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Preprocessing script for 3D-FRONT validation set (199 point clouds)
# Estimated time: ~20-25 minutes

echo "=========================================="
echo "3D-FRONT Validation Data Preprocessing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Create output directory
OUTPUT_DIR="/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run preprocessing for validation set
echo "Running preprocessing for 3D-FRONT_val dataset (199 point clouds)..."
echo "Command: python3 unified_pipeline/01_preprocess.py --dataset 3D-FRONT_val --start_idx 0 --end_idx 199"
echo ""

python3 unified_pipeline/01_preprocess.py \
    --dataset 3D-FRONT_val \
    --start_idx 0 \
    --end_idx 199

EXITCODE=$?

echo ""
echo "=========================================="
echo "Job finished"
echo "Exit code: $EXITCODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXITCODE
