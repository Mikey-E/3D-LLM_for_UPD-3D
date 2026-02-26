#!/bin/bash
#SBATCH --job-name=3dfront_finetuned_inference
#SBATCH --output=unified_pipeline/logs/3dfront_finetuned_%j.out
#SBATCH --error=unified_pipeline/logs/3dfront_finetuned_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=mb-l40s
#SBATCH --account=3dllms
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inference script for 3D-FRONT dataset with FINETUNED model
# Runs inference on all 2,992 point clouds
# Each point cloud has 12 questions = 35,904 total inferences
# Estimated time: ~1.5-2 hours

echo "=========================================="
echo "3D-FRONT Inference - FINETUNED Model"
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

# Finetuned checkpoint - 3D-FRONT checkpoint_7 (Oct 16, 2025)
CHECKPOINT="/project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_3D-FRONT/20251016100/checkpoint_7.pth"
MODEL_NAME="3dllm_ft-3D-FRONT-ckpt7"

echo "Checkpoint: $CHECKPOINT"
echo "Model name: $MODEL_NAME"
echo ""

# Run inference
echo "=========================================="
echo "Starting 3D-FRONT inference..."
echo "=========================================="

python3 unified_pipeline/02_run_inference.py \
    --dataset 3D-FRONT \
    --start_idx 0 \
    --end_idx 2992 \
    --checkpoint "$CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference completed successfully!"
    echo "Results saved to: results/3dfront_inference/$MODEL_NAME/"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
