#!/bin/bash
#SBATCH --job-name=preprocess_missing_crops3d
#SBATCH --output=./slurm_logs/preprocess_missing_%j.log
#SBATCH --error=./slurm_logs/preprocess_missing_%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Process the 4 missing point clouds from Crops3D training set
# These files use different color formats (short RGB) that need special handling

echo "=========================================="
echo "Processing Missing Crops3D Files"
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
source "$CONDA_SH"
conda activate lavis

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

cd /project/3dllms/melgin/3D-LLM_for_UPD-3D

# Run manual processing script
echo "Processing 4 missing files with manual script..."
python process_missing_manual.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
