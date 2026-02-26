# Finetuning Script Consistency Issue - FIXED

**Date:** October 15, 2025, 3:30 PM MDT

## Issue Discovered

The two finetuning scripts had an **important inconsistency** in how they launch the training process:

### Before Fix

**finetune_3dfront.sh:**
```bash
#SBATCH --gres=gpu:8

torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path lavis/projects/blip2/train/finetune_3dfront.yaml
```
✅ **Correct:** Uses `torchrun` to launch 8 separate processes (one per GPU)

**finetune_crops3d.sh:**
```bash
#SBATCH --gres=gpu:4

python train.py --cfg-path "$CONFIG"
```
❌ **Incorrect:** Uses single Python process that probably won't properly utilize all 4 GPUs

## Why This Matters

### Distributed Training Basics

PyTorch distributed training (which the 3D-LLM codebase uses) requires:

1. **Separate processes per GPU:** Each GPU needs its own Python process
2. **Process coordination:** `torchrun` handles:
   - Assigning rank to each process (0, 1, 2, 3 for 4 GPUs)
   - Setting up inter-process communication
   - Managing distributed data loading
   - Synchronizing gradients across GPUs

### What Happens Without torchrun

When you run `python train.py` directly on a multi-GPU setup:
- ❌ **Single process** tries to manage all GPUs
- ❌ **No proper rank assignment** for each GPU
- ❌ **Inefficient communication** between GPUs
- ❌ **May only use 1 GPU** or crash with distributed training errors
- ❌ **Slower training** due to suboptimal parallelization

### What torchrun Provides

When you run `torchrun --nproc_per_node=4 train.py`:
- ✅ **4 separate processes** launched (one per GPU)
- ✅ **Rank 0, 1, 2, 3** assigned automatically
- ✅ **Proper DDP (DistributedDataParallel)** setup
- ✅ **Efficient gradient synchronization**
- ✅ **Optimal training speed** using all GPUs

## The Fix

### Updated finetune_crops3d.sh

**Changed from:**
```bash
export MASTER_ADDR=$(hostname)
python train.py --cfg-path "$CONFIG"
```

**Changed to:**
```bash
export MASTER_ADDR=localhost
torchrun --nnodes=1 --nproc_per_node=4 train.py --cfg-path "$CONFIG"
```

**Changes:**
1. ✅ Added `torchrun` launcher with 4 processes (matching 4 GPUs)
2. ✅ Changed `MASTER_ADDR` from `$(hostname)` to `localhost` (more reliable for single-node)
3. ✅ Consistent with 3D-FRONT script pattern

## Expected Impact

### Training Speed
- **Before:** Likely only using 1 GPU effectively, or very slow multi-GPU
- **After:** All 4 GPUs utilized efficiently in parallel
- **Expected speedup:** ~3-4× faster (near-linear scaling with 4 GPUs)

### Training Correctness
- **Before:** May have encountered distributed training errors
- **After:** Proper gradient synchronization across all GPUs

### Consistency
- **Before:** Two scripts using different launch methods (confusing)
- **After:** Both scripts use same pattern (consistent, maintainable)

## Verification

When the Crops3D job starts, check the log for:

**Expected output with torchrun:**
```
[Rank 0] Starting training...
[Rank 1] Starting training...
[Rank 2] Starting training...
[Rank 3] Starting training...
```

**GPU utilization:**
```bash
nvidia-smi
# Should show all 4 GPUs with similar memory usage and utilization
```

## Why Was This Different?

The scripts were likely created at different times or from different templates:
- **3D-FRONT:** Newer script, properly set up for distributed training
- **Crops3D:** Older script, may have been copied from non-distributed example

## Lesson Learned

**Always use `torchrun` for PyTorch multi-GPU training**, not plain `python`:
```bash
# ❌ Wrong (single process)
python train.py

# ✅ Correct (distributed, multiple processes)
torchrun --nnodes=1 --nproc_per_node=N train.py
```

Where `N` = number of GPUs requested in SLURM script.

## Current Job Status

**Job 45848689 (Crops3D):**
- Status: PENDING (was submitted with old script)
- **Action needed:** Once it starts, monitor if it's using all GPUs properly
- **If issues:** Cancel and resubmit with the fixed script

**Job 45848496 (3D-FRONT):**
- Status: RUNNING ✅
- Already using correct `torchrun` approach
- No changes needed

---

**Summary:** The Crops3D script has been fixed to match the 3D-FRONT script's proper distributed training setup using `torchrun`. This should result in ~3-4× faster training by properly utilizing all 4 GPUs in parallel.
