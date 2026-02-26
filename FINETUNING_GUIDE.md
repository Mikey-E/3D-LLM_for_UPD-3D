# 3D-LLM Finetuning Scripts

## Overview
SLURM batch scripts for finetuning 3D-LLM (BLIP2) on 3D Visual Question Answering datasets.

## Available Scripts

### 1. ScanQA Finetuning
```bash
sbatch finetune_scanqa.sh
```
- **Dataset**: ScanQA (24,969 samples)
- **Config**: `3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_scanqa.yaml`
- **Output**: `slurm_logs/finetune_scanqa_<job_id>.log`

### 2. SQA3D Finetuning
```bash
sbatch finetune_sqa3d.sh
```
- **Dataset**: SQA3D (26,182 samples)
- **Config**: `3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_sqa.yaml`
- **Output**: `slurm_logs/finetune_sqa3d_<job_id>.log`

## Job Specifications

- **Account**: 3dllms
- **Partition**: mb-l40s
- **Time Limit**: 7 days (7-00:00:00)
- **Nodes**: 1
- **GPUs**: 8 (L40S)
- **Memory**: 96GB
- **Environment**: conda environment `lavis`

## Training Configuration

Both scripts use distributed training across 8 GPUs:
- **World Size**: 8
- **Batch Size per GPU**: 2
- **Effective Batch Size**: 16 (2 × 8)
- **Max Epochs**: 100
- **Learning Rate**: 1e-4 (with warmup and cosine decay)
- **Pretrained Checkpoint**: `checkpoints/pretrain_blip2_sam_flant5xl_v2.pth`

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View live log output
```bash
tail -f slurm_logs/finetune_scanqa_<job_id>.log
# or
tail -f slurm_logs/finetune_sqa3d_<job_id>.log
```

### Cancel a job
```bash
scancel <job_id>
```

## Output Directory

Training outputs (checkpoints, logs, tensorboard) are saved to:
```
3DLLM_BLIP2-base/output/BLIP2/3DQA/
```

## Verifying Setup Before Training

Before submitting jobs, verify the dataset setup:
```bash
cd 3DLLM_BLIP2-base
conda activate lavis
python test_dataset_loading.py
```

Expected output:
- ✓ ScanQA: PASSED (24,969 samples)
- ✓ SQA3D: PASSED (26,182 samples)

## Troubleshooting

### Job fails immediately
- Check SLURM logs in `slurm_logs/`
- Verify conda environment exists: `conda env list | grep lavis`
- Verify GPU availability on partition: `sinfo -p mb-l40s`

### Out of Memory errors
- Current config uses 5000 points per scene
- Batch size is 2 per GPU
- If OOM occurs, reduce batch size in YAML config or reduce point sampling

### Checkpoint not found
- Verify checkpoint exists: `ls -lh checkpoints/pretrain_blip2_sam_flant5xl_v2.pth`
- Should be ~4.2GB

### Dataset loading errors
- Run test script: `python test_dataset_loading.py`
- Verify features exist: `ls data/scannet_features/voxelized_*_sam_nonzero_preprocess/ | wc -l`
- Should show 1494 files in each directory

## Notes

1. **First Training Run**: The first epoch may be slower as data is loaded and cached
2. **Checkpoint Frequency**: Checkpoints are saved at each epoch
3. **Resume Training**: If job times out, you can resume from the last checkpoint by modifying the YAML config
4. **Multi-Job**: You can submit both ScanQA and SQA3D jobs simultaneously if resources allow

## References

- **Dataset Setup**: See `DATASET_SETUP_SUMMARY.md`
- **Environment**: See `INSTALLATION_GUIDE.md`
- **Original Repository**: [3D-LLM GitHub](https://github.com/UMass-Foundation-Model/3D-LLM)
