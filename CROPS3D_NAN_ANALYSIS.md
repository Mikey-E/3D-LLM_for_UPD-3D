# Crops3D Training NaN Loss Analysis

**Date**: October 15, 2024, 8:30 PM MDT  
**Job ID**: 45848700  
**Status**: ⚠️ CRITICAL - Training diverged with NaN losses

## Executive Summary

Crops3D training **diverged at epoch 3, step 850** (out of 934 steps). Loss went from 0.0628 → NaN instantly, indicating gradient explosion. The training has continued with NaN losses through epoch 10, wasting 1.5+ hours of compute time.

**Recommendation**: **STOP JOB IMMEDIATELY** to save compute resources.

---

## Timeline of NaN Onset

### Healthy Training (Epochs 1-3 early)
- **Epoch 1, Step 100**: loss = 0.0156 ✓
- **Epoch 2, Step 100**: loss = 0.4805 ✓
- **Epoch 3, Steps 0-800**: Normal loss progression

### Critical Transition (Epoch 3, Steps 800-850)
```
Step 750: loss = 0.0008 ✓ (valid)
Step 800: loss = 0.0628 ✓ (valid, slightly elevated)
Step 850: loss = nan   ✗ (EXPLOSION!)
Step 900: loss = nan   ✗ (continued)
```

**Exact failure point**: Between step 800 and 850 of epoch 3  
**Time to failure**: ~2.5 hours / 2.5 epochs  
**Total steps before NaN**: ~2,402 steps (2×934 + 850)

### Post-Divergence (Epochs 4-10)
- All subsequent epochs show continuous NaN losses
- One anomaly: Epoch 4 step 0 showed loss=0.0004 (checkpoint reload effect?)
- Training speed unchanged: ~1.1 sec/step
- Memory stable: 22,887 MB (no OOM)

---

## Root Cause Analysis

### 1. No Gradient Clipping
**CRITICAL FINDING**: Codebase has **no gradient clipping** mechanism.
- Searched for: `max_grad_norm`, `clip_grad`, `gradient_clip`
- Result: **Zero matches** in training code
- Impact: Allows unbounded gradient growth → instant NaN

### 2. Config Comparison with 3D-FRONT (Healthy Job)

| Parameter | Crops3D | 3D-FRONT | Impact |
|-----------|---------|----------|--------|
| Learning Rate | 1e-4 | 1e-4 | Same |
| Warmup Steps | 1000 | 1000 | Same |
| Weight Decay | 0.05 | 0.05 | Same |
| Batch Size | 2/GPU × 4 GPUs = 8 | 1/GPU × 8 GPUs × 2 accum = 16 | **Different!** |
| Dataset Size | 7,476 samples | 105,240 samples | 14× smaller |
| Steps/Epoch | 934 | 13,155 | 14× fewer |

**Key Difference**: Crops3D completes epochs 14× faster (17 min vs 4.8 hours).

### 3. Learning Rate Schedule Interaction
Both use `linear_warmup_cosine_lr`:
- Warmup: 1,000 steps = **1.07 epochs** for Crops3D vs **0.076 epochs** for 3D-FRONT
- At step 2,402 (failure point): LR = 0.000100 (full LR, post-warmup)
- Cosine decay slower for Crops3D (smaller dataset, more epochs needed)

### 4. Hypothesis: Dataset-Specific Issue
- **Crops3D**: Small dataset (7,476 samples) → faster epoch completion → more repeat exposure
- **3D-FRONT**: Large dataset (105K samples) → slower convergence → more stable
- Possible: Crops3D memorizing training set → overfitting → gradient explosion on memorized samples

### 5. Why 3D-FRONT is Stable
- Larger dataset provides more diverse gradients
- Longer epochs smooth out gradient variance
- Effective batch size 2× larger (16 vs 8)
- More time for optimizer momentum to stabilize

---

## Why Normalization Didn't Prevent This

Despite both datasets normalized to identical statistics (mean=0.0134, std=0.0229):
- ✓ Input features are normalized
- ✓ Initial losses were reasonable (0.01-0.6 range)
- ✗ But **gradient clipping** was missing to prevent explosion
- ✗ Dataset size differences create different optimization dynamics

**Normalization helps but is NOT sufficient** without gradient clipping.

---

## Comparison: Expected vs Actual

### 3D-FRONT (Job 45848496) - HEALTHY
- **Status**: Running 5+ hours, epoch 1 at 86.3%
- **Loss Range**: 0.0 - 0.4 (stable)
- **NaN Occurrences**: **ZERO**
- **Memory**: 50% utilization, no OOM
- **Progress**: On track, ~4.8 days for 20 epochs

### Crops3D (Job 45848700) - FAILED
- **Status**: Running 4.5 hours, epoch 10 (but useless)
- **Loss Range**: Epochs 1-3: 0.0 - 0.6, Epochs 4-10: **NaN**
- **NaN Onset**: Step 850 of epoch 3
- **Wasted Compute**: 1.5 hours (epochs 4-10) running with NaN
- **Progress**: **ZERO** after epoch 3

---

## Recommended Fixes

### Option A: Add Gradient Clipping (RECOMMENDED)
Add to `train.py` in optimizer step:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Option B: Reduce Learning Rate
Change in `finetune_crops3d.yaml`:
```yaml
init_lr: 5e-5  # Was 1e-4
```

### Option C: Increase Batch Size
Change in `finetune_crops3d.yaml`:
```yaml
batch_size_train: 4  # Was 2 (requires checking memory)
```

### Option D: Extend Warmup
Change in `finetune_crops3d.yaml`:
```yaml
warmup_steps: 2000  # Was 1000 (2.14 epochs instead of 1.07)
```

### Option E (Best): Combination Approach
1. Add gradient clipping (max_norm=1.0)
2. Slightly reduce LR (1e-4 → 8e-5)
3. Extend warmup (1000 → 1500 steps)

---

## Immediate Actions Required

### 1. Stop Current Job ⚠️
```bash
scancel 45848700
```
**Reason**: Training is producing no value after epoch 3. Saving 2+ days of wasted GPU time.

### 2. Implement Fix
**Preferred**: Add gradient clipping to `train.py` before optimizer.step()

### 3. Resubmit with Monitoring
- Check loss values at epoch 3, step 850 specifically
- Add validation logging every 500 steps
- Set up alert if loss > 10.0 (pre-NaN detection)

---

## Technical Details

### Job Information
- **Job ID**: 45848700
- **Partition**: long
- **GPUs**: 4× L40S (46GB each)
- **Memory**: 96GB
- **Time Limit**: 7 days
- **Runtime**: 4:28:05 (4.5 hours)

### Training Configuration
- **Model**: BLIP2-FlanT5-XL (372M params)
- **Optimizer**: AdamW
- **Batch Size**: 2 per GPU × 4 GPUs = 8 effective
- **Gradient Accumulation**: 1 (no accumulation)
- **Mixed Precision**: Likely enabled (inherited from ScanNet)

### Loss Trajectory Sample
```
Epoch 3 progression:
  Step 700: 0.0004
  Step 750: 0.0008
  Step 800: 0.0628  ← Slight increase (warning sign?)
  Step 850: nan     ← EXPLOSION
  Step 900: nan
  ...
Epoch 4:
  Step 0:   0.0004  ← Checkpoint reload shows brief recovery
  Step 50:  nan     ← But divergence continues
```

---

## Lessons Learned

1. **Gradient clipping is essential** for small datasets
2. **Normalization alone is insufficient** for training stability
3. **Different dataset sizes require different hyperparameters**
4. **Monitoring should catch pre-NaN warning signs** (loss > 1.0 spikes)
5. **Checkpoint reload doesn't fix diverged training** (see epoch 4 step 0)

---

## Next Steps

1. ✅ Document issue (this file)
2. ⏳ Stop job 45848700
3. ⏳ Add gradient clipping to codebase
4. ⏳ Test on Crops3D with monitoring
5. ⏳ If successful, add clipping to 3D-FRONT too (preventive)

---

## Conclusion

Crops3D training failed due to **gradient explosion** at epoch 3, step 850, caused by:
1. **Missing gradient clipping** (primary cause)
2. **Small dataset size** creating optimization instability
3. **Fast epoch completion** amplifying divergence issues

The job should be **cancelled immediately** and resubmitted with gradient clipping enabled.

**Status**: 🔴 ACTION REQUIRED - Stop job 45848700 and implement fix
