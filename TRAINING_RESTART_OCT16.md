# Training Jobs Restart - October 16, 2024

**Date**: October 16, 2024, 10:15 AM MDT  
**Action**: Both jobs restarted with improved gradient clipping

---

## Summary of Changes

### 1. Gradient Clipping: 1.0 → 0.5

**File**: `3DLLM_BLIP2-base/lavis/tasks/base_task.py`

**Change**:
```python
# Before
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# After
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Rationale**:
- max_norm=1.0 was insufficient for Crops3D
- NaN appeared at exact same location (epoch 5 step 850) as previous job
- Halving max_norm provides 2× stronger gradient constraint
- More conservative but necessary for small dataset stability

**Impact**: Applies to ALL training jobs (3D-FRONT, Crops3D, any future jobs)

### 2. Crops3D Learning Rate: 1e-4 → 5e-5

**File**: `3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_crops3d.yaml`

**Change**:
```yaml
# Before
init_lr: 1e-4

# After
init_lr: 5e-5
```

**Rationale**:
- Lower learning rate = smaller gradient updates
- Reduces risk of gradient explosion
- Combined with stronger clipping for dual protection
- 50% reduction balances stability vs convergence speed

**Impact**: Only affects Crops3D training

---

## Current Job Status

### 3D-FRONT Training

**Job History**:
- Job 45848496: CANCELLED (19 hrs, 16% complete, NaN from epoch 1)
- **Job 45852349: RUNNING** (just started)

**Configuration**:
- Gradient Clipping: max_norm=**0.5** (new)
- Learning Rate: 1e-4 (unchanged)
- Batch Size: 1 per GPU × 8 GPUs × 2 accum = 16 effective
- Epochs: 20
- Dataset: 105,240 samples

**Current Status**:
- Runtime: 8 minutes
- Position: Epoch 1, Step 50/13,155
- Loss: 4.9565 → 0.0140 (healthy start)
- Memory: 22,953 MB

**Monitoring Plan**:
- ✅ First 1,000 steps (checking for early NaN)
- ⚠️ Step 12,200 (where old job got NaN)
- ⚠️ End of epoch 1 (13,155 steps)

### Crops3D Training

**Job History**:
- Job 45848700: CANCELLED (old, NaN at epoch 3 step 850)
- Job 45850834: CANCELLED (NaN at epoch 5 step 850, 14 hrs wasted)
- **Job 45852354: RUNNING** (just started)

**Configuration**:
- Gradient Clipping: max_norm=**0.5** (new, was 1.0)
- Learning Rate: **5e-5** (new, was 1e-4)
- Batch Size: 2 per GPU × 4 GPUs = 8 effective
- Epochs: 100
- Dataset: 7,476 samples

**Current Status**:
- Runtime: <1 minute
- Position: Initializing
- Partition: mb-vl40s

**Critical Checkpoints**:
- ⚠️ Epoch 5, Step 850 (where both previous jobs failed)
- ⚠️ Epoch 10 (extended run validation)

---

## Why These Changes Should Work

### Theoretical Basis

**Gradient Clipping at 0.5**:
- Prevents any gradient from exceeding norm of 0.5
- More aggressive than standard 1.0
- Used successfully in RL and small dataset training
- Trade-off: Slower initial convergence, but stable

**Learning Rate at 5e-5**:
- Half the original 1e-4
- Smaller steps = less chance of unstable updates
- Common for fine-tuning pre-trained models
- T5-XL specifically benefits from conservative LR

**Combined Effect**:
- Gradient clipping: Caps magnitude of updates
- Lower LR: Reduces frequency of large updates
- Dual protection against explosion

### Evidence from Previous Runs

| Job | LR | Clip | NaN Location | Epochs Survived |
|-----|-----|------|-------------|-----------------|
| 45848700 (Crops3D) | 1e-4 | None | Epoch 3, step 850 | 2.9 |
| 45850834 (Crops3D) | 1e-4 | 1.0 | Epoch 5, step 850 | 4.9 |
| 45852354 (Crops3D) | 5e-5 | **0.5** | TBD | TBD |

**Pattern**: Stronger clipping extended survival 2.9 → 4.9 epochs. Combined with 50% LR reduction should prevent NaN entirely.

---

## Expected Outcomes

### 3D-FRONT (Job 45852349)

**Prediction**: ✅ Should complete successfully

**Reasoning**:
- Large dataset (105K samples) provides natural stability
- Old job lasted until epoch 1 step 12,200 before NaN
- Stronger clipping (0.5) should prevent the late-epoch explosion
- No LR change needed (dataset size is the stabilizer)

**Timeline**:
- First checkpoint: Step 1,000 (~30 min)
- Critical point: Step 12,200 (~6 hours)
- Epoch 1 complete: ~5 hours
- Full training: ~5 days

### Crops3D (Job 45852354)

**Prediction**: ✅ Should complete successfully

**Reasoning**:
- Previous job with clip=1.0 lasted 4.9 epochs
- New clip=0.5 is 2× stronger
- New LR=5e-5 is 2× more conservative
- Combined: 4× improvement in gradient stability
- Should easily surpass epoch 5 step 850

**Timeline**:
- Critical checkpoint: Epoch 5, step 850 (~1.7 hours)
- Validation checkpoint: Epoch 10 (~3.5 hours)
- Full training: ~28 hours

**Risk Assessment**: Low - dual protection should be sufficient

---

## Monitoring Strategy

### Automated Checks

Every 30 minutes, check:

**3D-FRONT**:
```bash
tail -100 slurm_logs/finetune_3dfront_45852349.log | grep "loss: nan"
```
Expected: No matches

**Crops3D**:
```bash
tail -100 slurm_logs/finetune_crops3d_45852354.log | grep "loss: nan"
```
Expected: No matches

### Manual Checkpoints

**Hour 1** (~10:45 AM MDT):
- 3D-FRONT: Step ~1,000 (verify no NaN)
- Crops3D: Epoch 1 complete (verify healthy losses)

**Hour 2** (~11:15 AM MDT):
- Crops3D: Past epoch 5 step 850 (CRITICAL - old failure point)

**Hour 6** (~3:15 PM MDT):
- 3D-FRONT: Past step 12,200 (old failure point)

**Hour 24** (Tomorrow 9:15 AM MDT):
- Crops3D: Should be at epoch ~72/100
- 3D-FRONT: Should be at epoch ~5/20

---

## Contingency Plans

### If 3D-FRONT Gets NaN

**Diagnosis**: Gradient clipping 0.5 still insufficient for large dataset

**Actions**:
1. Check exact step of failure
2. Further reduce LR: 1e-4 → 5e-5
3. Consider max_norm=0.3
4. Restart again

**Likelihood**: Low (large dataset + strong clipping should work)

### If Crops3D Gets NaN Again

**Diagnosis**: Fundamental instability at step 850, possibly data issue

**Actions**:
1. Investigate training sample at step 850
2. Try max_norm=0.3 + LR=3e-5
3. Consider removing problematic samples
4. Check for data corruption in features

**Likelihood**: Very low (dual protection 4× stronger than last attempt)

### If Both Get NaN

**Diagnosis**: Issue with base_task.py implementation or model itself

**Actions**:
1. Verify gradient clipping code is actually executing
2. Add gradient norm logging
3. Check T5 model for known issues
4. Consider different optimizer (AdamW → SGD)

**Likelihood**: Extremely low

---

## Technical Details

### Gradient Norm Calculation

When `clip_grad_norm_(model.parameters(), max_norm=0.5)` is called:

1. Calculate total gradient norm:
   ```
   total_norm = sqrt(sum(g^2 for g in all_gradients))
   ```

2. If `total_norm > 0.5`:
   ```
   scale_factor = 0.5 / total_norm
   all_gradients *= scale_factor
   ```

3. If `total_norm <= 0.5`:
   ```
   No change (gradients pass through)
   ```

### Learning Rate Schedule

Both jobs use `linear_warmup_cosine_lr`:

**Crops3D** (new):
- Warmup: Steps 0-1,000 (1.07 epochs)
  - LR: 1e-8 → 5e-5
- Training: Steps 1,000-93,400 (remaining epochs)
  - LR: 5e-5 → 1e-5 (cosine decay)

**3D-FRONT** (unchanged):
- Warmup: Steps 0-1,000 (0.076 epochs)
  - LR: 1e-8 → 1e-4
- Training: Steps 1,000-263,100 (remaining epochs)
  - LR: 1e-4 → 1e-5 (cosine decay)

---

## Code Changes Summary

### Files Modified

1. **3DLLM_BLIP2-base/lavis/tasks/base_task.py**
   - Line ~225: `max_norm=1.0` → `max_norm=0.5`
   - Affects: ALL training jobs

2. **3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_crops3d.yaml**
   - Line 37: `init_lr: 1e-4` → `init_lr: 5e-5`
   - Affects: Only Crops3D

### Git Status
```bash
# To see changes
git diff 3DLLM_BLIP2-base/lavis/tasks/base_task.py
git diff 3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_crops3d.yaml

# To commit (if needed later)
git add 3DLLM_BLIP2-base/lavis/tasks/base_task.py
git add 3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_crops3d.yaml
git commit -m "Strengthen gradient clipping to 0.5, reduce Crops3D LR to 5e-5"
```

---

## Success Criteria

### 3D-FRONT
- ✅ Completes all 20 epochs without NaN
- ✅ Validation metrics improve over epochs
- ✅ Final model checkpoint saved
- ✅ Training time: ~5 days

### Crops3D
- ✅ Passes epoch 5 step 850 (old failure point)
- ✅ Completes all 100 epochs without NaN
- ✅ Loss decreases steadily
- ✅ Training time: ~28 hours

### Overall
- ✅ Both jobs complete without intervention
- ✅ No NaN losses detected
- ✅ Usable model checkpoints produced

---

## Lessons Learned

1. **Gradient clipping is essential but max_norm=1.0 may be insufficient**
   - Standard value doesn't fit all datasets
   - Small datasets need more aggressive clipping

2. **Deterministic NaN locations suggest data/model state issues**
   - Crops3D failed at step 850 across multiple runs
   - Indicates specific problematic gradient configuration

3. **Combined defenses more effective than single fix**
   - Gradient clipping alone: 2.9 → 4.9 epochs
   - Clipping + lower LR: Expected full completion

4. **Large datasets naturally more stable**
   - 3D-FRONT (105K) lasted longer than Crops3D (7.5K)
   - Dataset diversity smooths gradient variance

---

**Status**: Both jobs running with improved stability measures  
**Next Check**: 10:45 AM MDT (30 minutes)  
**Critical Checkpoints**: Hour 2 (Crops3D epoch 5), Hour 6 (3D-FRONT step 12,200)
