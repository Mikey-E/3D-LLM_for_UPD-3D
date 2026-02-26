#!/bin/bash
#SBATCH --job-name=giw529_base_inference
#SBATCH --output=unified_pipeline/logs/giw529_base_%j.out
#SBATCH --error=unified_pipeline/logs/giw529_base_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inference script for GIW529 TEST SET with BASE (pretrained) model
# Runs inference on 150 test point clouds
# Each point cloud has 12 questions = 1,800 total inferences
# Estimated time: ~1 hour

echo "=========================================="
echo "GIW529 Test Inference - BASE Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
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

# Base pretrained checkpoint
CHECKPOINT="/project/3dllms/melgin/3D-LLM_for_UPD-3D/checkpoints/pretrain_blip2_sam_flant5xl_v2.pth"
MODEL_NAME="3dllm_base"

echo "Checkpoint: $CHECKPOINT"
echo "Model name: $MODEL_NAME"
echo ""

# Run inference
echo "=========================================="
echo "Starting GIW529 test set inference..."
echo "=========================================="

python3 unified_pipeline/02_run_inference.py \
    --dataset GIW529 \
    --start_idx 0 \
    --end_idx 150 \
    --checkpoint "$CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference completed successfully!"
    echo "Results saved to: results/giw529_inference/$MODEL_NAME/"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
