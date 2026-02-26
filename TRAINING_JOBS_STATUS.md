# Training Jobs Status - October 15, 2025

## Active Jobs Overview

### 1. 3D-FRONT Finetuning (Job 45848496) ✅ RUNNING
- **Status:** RUNNING (26+ minutes)
- **Node:** mbl40s-003
- **GPUs:** 8× L40S
- **Memory:** 192GB
- **Started:** Oct 15, 14:56 MDT
- **Progress:** Model initialization in progress
- **Config:** 20 epochs, batch_size=1, accum_grad=2

**Key Metrics:**
- Training samples: 105,240
- Validation samples: 2,388
- Features: Normalized (mean=0.0134, std=0.0229) ✓
- Est. time per epoch: ~4-5 hours
- Est. total time: ~90 hours (~3.75 days)

**History:**
- First attempt (45848404): FAILED with OUT_OF_MEMORY after 4 minutes
- Fixed: Doubled memory (96→192GB), halved batch size (2→1), added gradient accumulation

---

### 2. Crops3D Finetuning (Job 45848689) ⏳ PENDING
- **Status:** PENDING (waiting for GPU resources)
- **Node:** Not yet assigned
- **GPUs:** Requested 8× L40S
- **Memory:** 96GB
- **Submitted:** Oct 15, 15:21 MDT
- **Queue Position:** Waiting for available L40S node

**Key Metrics:**
- Training samples: 7,476
- Validation samples: 180
- Features: Normalized (mean=0.0134, std=0.0229) ✓
- Est. time per epoch: ~30-40 minutes
- Est. total time: ~50-70 hours (~2-3 days) for 100 epochs

**Dataset Info:**
- 1,180 preprocessed files verified
- Training data: Crops3D_train.json (2.8MB)
- Validation data: Crops3D_val.json (896KB)
- Features normalized on Oct 13, 19:00

---

## L40S Partition Status

**Current Usage (3 nodes, all occupied):**

| Node | User | Job | Status | Runtime |
|------|------|-----|--------|---------|
| mbl40s-001 | melgin | interactive (45847620) | RUNNING | 5:05 hrs |
| mbl40s-001 | mzamini | llava_7b (45848507) | RUNNING | 21 min |
| mbl40s-002 | ukapoor | LanM_open (44231397) | RUNNING | 6d 4h |
| mbl40s-003 | melgin | **3D-FRONT** (45848496) | RUNNING | 27 min |

**Queue:**
1. Crops3D (45848689) - waiting for resources

**Expected Availability:**
- Interactive session (45847620): ~3 hours remaining (8hr limit)
- llava_7b (45848507): ~3.5 hours remaining (4hr limit)
- When these complete, Crops3D job should start

---

## Training Configurations Comparison

| Parameter | 3D-FRONT | Crops3D |
|-----------|----------|---------|
| **Data** | | |
| Training samples | 105,240 | 7,476 |
| Validation samples | 2,388 | 180 |
| Dataset size | 14× larger | Baseline |
| **Resources** | | |
| Memory | 192GB | 96GB |
| GPUs | 8× L40S | 8× L40S |
| Batch size (per GPU) | 1 | 2 |
| Gradient accumulation | 2 | 1 |
| Effective batch size | 16 | 16 |
| Data workers | 2 | 4 |
| **Training** | | |
| Epochs | 20 | 100 |
| Learning rate | 1e-4 | 1e-4 |
| Warmup steps | 1000 | 1000 |
| **Estimates** | | |
| Steps per epoch | ~6,578 | ~467 |
| Time per epoch | ~4-5 hrs | ~30-40 min |
| Total training time | ~90 hrs (3.75d) | ~50-70 hrs (2-3d) |
| Total samples seen | ~2.1M | ~750K |

---

## Feature Normalization Status

Both datasets normalized to ScanNet distribution:
- **Target:** mean=0.0134, std=0.0229
- **3D-FRONT:** ✓ Verified (preprocessed Oct 15 with integrated normalization)
- **Crops3D:** ✓ Verified (normalized Oct 13)

**Why Normalization is Critical:**
- Prevents NaN loss during training
- Matches model's expected feature distribution
- 3D-FRONT raw features had 685× larger std than target
- Previous unnormalized attempt showed NaN loss

---

## Monitoring Commands

**Check job status:**
```bash
# 3D-FRONT
squeue -j 45848496
tail -f slurm_logs/finetune_3dfront_45848496.log

# Crops3D (when running)
squeue -j 45848689
tail -f slurm_logs/finetune_crops3d_45848689.log
```

**Check L40S availability:**
```bash
squeue -p mb-l40s --format="%.10i %.20u %.40j %.8T %.10M %.10l"
```

**Monitor training progress:**
```bash
# Check for loss values
grep -i "loss\|epoch" slurm_logs/finetune_3dfront_45848496.log | tail -20
grep -i "loss\|epoch" slurm_logs/finetune_crops3d_45848689.log | tail -20
```

---

## Next Steps

### 3D-FRONT (Currently Running)
1. ⏳ Wait for model loading to complete (~5-10 minutes)
2. ⏳ Verify training starts without errors
3. ⏳ Check first loss values (should not be NaN)
4. ⏳ Monitor first epoch completion (~4-5 hours)

### Crops3D (Pending Resources)
1. ⏳ Wait for L40S node to become available (~3-4 hours estimated)
2. ⏳ Job will auto-start when resources available
3. ⏳ Verify training starts successfully
4. ⏳ Monitor first epoch completion (~30-40 minutes)

---

## Output Directories

**3D-FRONT:**
- Checkpoints: `output/BLIP2/3DQA_3D-FRONT/`
- Log: `slurm_logs/finetune_3dfront_45848496.log`

**Crops3D:**
- Checkpoints: `output/BLIP2/3DQA_Crops3D/`
- Log: `slurm_logs/finetune_crops3d_45848689.log`

---

**Last Updated:** Oct 15, 2025, 15:25 MDT
**Status Summary:** 1 running, 1 pending, both with normalized features ✓
