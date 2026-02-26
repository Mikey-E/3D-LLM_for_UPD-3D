#!/bin/bash
#SBATCH --account=3dllms
#SBATCH --job-name=giw529_finetuned
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=3-00:00:00
#SBATCH --output=./unified_pipeline/logs/giw529_finetuned_%j.out
#SBATCH --error=./unified_pipeline/logs/giw529_finetuned_%j.err

echo "=========================================="
echo "GIW529 Finetuned Model Inference"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Setup conda
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

if [ "$CONDA_DEFAULT_ENV" != "lavis" ]; then
    echo "ERROR: Failed to activate lavis environment"
    exit 1
fi

echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Run inference
cd unified_pipeline

echo "Running GIW529 finetuned model inference..."
echo "Checkpoint: checkpoint_7.pth (epoch 7)"
echo "Dataset: GIW529 test set (150 scenes)"
echo "Questions: 12 categories per scene"
echo ""

python3 02_run_inference.py \
    --checkpoint /cluster/medbow/project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_GIW529/20251206132/checkpoint_7.pth \
    --model_name 3dllm_ft-GIW529_gpt-5-nano-ckpt7 \
    --dataset GIW529 \
    --start_idx 0 \
    --end_idx 150

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference completed successfully!"
    echo "Results saved to: results/giw529_inference/3dllm_ft-GIW529_gpt-5-nano-ckpt7/"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
