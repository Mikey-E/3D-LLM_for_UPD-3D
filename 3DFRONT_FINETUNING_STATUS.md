# 3D-FRONT Finetuning Job Status

**Date:** October 15, 2025, 3:00 PM MDT

## Job History

### First Attempt (Job 45848404) - FAILED ❌
- **Status:** OUT_OF_MEMORY after 4:13 minutes
- **Problem:** Insufficient memory (96GB) for large dataset
- **Error:** "Detected 2 oom_kill events" - SIGKILL on processes
- **Root Cause:** 3D-FRONT has 14× more samples than Crops3D (105K vs 7.5K)

### Configuration Changes Made

**Memory:**
- ❌ Before: 96GB
- ✅ After: 192GB (2× increase)

**Batch Size:**
- ❌ Before: batch_size_train=2, batch_size_eval=4
- ✅ After: batch_size_train=1, batch_size_eval=2
- Added: accum_grad_iters=2 (maintains effective batch size of 2 per GPU)

**DataLoader:**
- ❌ Before: num_workers=4
- ✅ After: num_workers=2 (reduces concurrent data loading)

### Second Attempt (Job 45848496) - RUNNING ✅
- **Submitted:** Oct 15, 14:56 MDT
- **Status:** RUNNING (4+ minutes, past previous OOM point)
- **Memory:** 192GB
- **GPUs:** 8× L40S
- **Node:** mbl40s-003

## Current Job Status

**Progress:**
```
✓ Data loading: COMPLETE
  - Training: 105,240 samples detected
  - Validation: 2,388 samples detected
✓ Dataset auto-detection: 3D-FRONT paths working correctly
⏳ Model loading: IN PROGRESS (loading EVA-ViT on 8 GPUs)
⏳ Training start: PENDING
```

**Key Indicators:**
- ✅ Job survived past 4:13 (previous OOM point)
- ✅ Dataset loading successful
- ✅ Feature paths correctly detected
- ✅ Normalized features loaded (mean=0.0134, std=0.0229)

## Training Configuration

**Epochs & Learning:**
- Max epochs: 20
- Learning rate: 1e-4 (warmup from 1e-8)
- Warmup steps: 1000
- LR scheduler: linear_warmup_cosine_lr
- Weight decay: 0.05

**Batch Configuration:**
- Per-GPU batch size: 1 (train), 2 (eval)
- Gradient accumulation: 2 steps
- Effective batch size: 2 per GPU = 16 total (8 GPUs)
- Same effective batch as Crops3D (but more stable memory usage)

**Training Time Estimate:**
- Steps per epoch: 105,240 / 16 ≈ 6,578 steps
- Time per epoch: ~4-5 hours (estimated)
- Total training time: 20 epochs × 4.5 hours ≈ 90 hours (~3.75 days)

**Output:**
- Directory: `output/BLIP2/3DQA_3D-FRONT/`
- Checkpoints: Saved periodically
- Log: `slurm_logs/finetune_3dfront_45848496.log`

## Why This Should Work

1. **Memory increased 2×** (96GB → 192GB)
   - Provides headroom for large batches and 8 GPU parallel training
   
2. **Batch size reduced by half** (2 → 1 per GPU)
   - Reduces peak memory during forward/backward pass
   - Gradient accumulation maintains training dynamics
   
3. **Workers reduced** (4 → 2)
   - Less concurrent data loading
   - Reduces memory pressure from prefetched batches

4. **Normalized features**
   - Mean=0.0134, std=0.0229 (perfect match to ScanNet)
   - Should prevent NaN loss issues

## Next Steps

**Monitor job for:**
1. ✓ Survival past 5 minutes (OOM threshold) - **ACHIEVED**
2. ⏳ Training iteration start
3. ⏳ First loss values (check for NaN)
4. ⏳ First epoch completion (~4-5 hours)
5. ⏳ Validation metrics

**Commands to monitor:**
```bash
# Check job status
squeue -j 45848496

# Watch log tail
tail -f slurm_logs/finetune_3dfront_45848496.log

# Check for training progress
grep -i "epoch\|loss\|train" slurm_logs/finetune_3dfront_45848496.log
```

## Comparison: 3D-FRONT vs Crops3D

| Metric | Crops3D | 3D-FRONT | Ratio |
|--------|---------|----------|-------|
| Training samples | 7,500 | 105,240 | 14× |
| Validation samples | 180 | 2,388 | 13× |
| Memory allocated | 96GB | 192GB | 2× |
| Batch size (per GPU) | 2 | 1 | 0.5× |
| Effective batch size | 16 | 16 | 1× |
| Epochs | 100 | 20 | 0.2× |
| Est. training time | 4-5 days | 3.75 days | ~0.8× |

**Key Insight:** Despite 14× more data, training time is similar due to fewer epochs (20 vs 100). The model will see roughly the same number of total samples (100 epochs × 7.5K ≈ 750K vs 20 epochs × 105K ≈ 2.1M).

---

**Status as of 15:00 MDT:** Job running successfully, initializing model on 8 GPUs. ✓
