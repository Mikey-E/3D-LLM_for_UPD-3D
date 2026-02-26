# 3D-FRONT Preprocessing with Integrated Normalization

## Date: October 15, 2025

## Background
After disk quota issues required deleting the initial 3D-FRONT processed data, we implemented an improved preprocessing pipeline that includes normalization during feature extraction rather than as a separate post-processing step.

## Changes Made

### Updated: `unified_pipeline/01_preprocess.py`

**Added normalization function:**
```python
def normalize_features(features, target_mean=0.0134, target_std=0.0229):
    """
    Normalize features to match ScanNet feature distribution.
    
    This prevents NaN loss during training by ensuring feature scales
    match the model's training distribution.
    """
    # Standardize to mean=0, std=1
    current_mean = features.mean()
    current_std = features.std()
    
    features_standardized = (features - current_mean) / (current_std + 1e-8)
    
    # Scale to target distribution
    features_normalized = features_standardized * target_std + target_mean
    
    return features_normalized
```

**Integrated into preprocessing pipeline:**
- Features are normalized immediately after extraction, before saving
- Only applied to 3D-FRONT datasets (Crops3D uses its own normalization)
- Target statistics from ScanNet: mean=0.0134, std=0.0229

## Why Normalization is Critical

**Before normalization (measured from initial preprocessing):**
- 3D-FRONT features: mean=-0.1107, std=15.6946
- ScanNet features: mean=0.0134, std=0.0229
- **Scale difference: 685× larger standard deviation!**

**Impact without normalization:**
- Previous job (45540895) showed NaN loss after just a few iterations
- Same issue occurred with Crops3D (669× scale difference)
- Model expects features in narrow ScanNet distribution range

## Current Preprocessing Jobs

**Training Set (Job 45847668):**
- 32 array tasks (8 concurrent)
- Processing 8,770 files from `3D-FRONT_train_minus_val.txt`
- Each task: ~274 files, ~25-30 minutes
- Output: `/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed/`
- Features: Automatically normalized to ScanNet distribution

**Validation Set (Job 45847677):**
- Single job
- Processing 199 files from `3D-FRONT_val_subset_of_train.txt`
- Expected time: ~20 minutes
- Output: Same directory as training set
- Features: Automatically normalized to ScanNet distribution

## Expected Outputs

**Total files:** 8,969 (8,770 train + 199 val)

**Per point cloud:**
- `.pt` file: [8192, 1408] tensor with normalized features
  - mean ≈ 0.0134 (±0.005)
  - std ≈ 0.0229 (±0.005)
- `.npy` file: [8192, 3] array with normalized coordinates ([-1, 1] range)

**Disk usage:** ~549 GB (61 MB average per file pair)

## Verification Steps

After jobs complete:

```bash
# 1. Count files
ls /project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed/*.pt | wc -l
# Expected: 8969

# 2. Check normalization statistics (sample file)
python3 -c "
import torch
pt = torch.load('/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed/[sample_file].pt', map_location='cpu')
print(f'Shape: {pt.shape}')
print(f'Mean: {pt.mean():.4f} (target: 0.0134)')
print(f'Std: {pt.std():.4f} (target: 0.0229)')
print(f'Range: [{pt.min():.4f}, {pt.max():.4f}]')
"
```

## Next Steps

1. **Monitor preprocessing jobs** (~2-3 hours for training set)
2. **Verify normalization** with sample files
3. **Submit finetuning job** (`sbatch finetune_3dfront.sh`)
   - 20 epochs, 8 GPUs
   - Expected time: ~3.7 days
   - Should NOT show NaN loss (unlike job 45540895)

## Advantages of Integrated Normalization

✅ **No disk quota issues** - Don't need to store both unnormalized and normalized versions
✅ **One-step process** - Extract and normalize in single pass
✅ **Consistent** - Every file normalized identically during preprocessing
✅ **Efficient** - No need to reload/resave all files for normalization
✅ **Automatic** - Works for any future 3D-FRONT preprocessing runs

## References

- Feature statistics analysis: Previous session (October 15, 10:15 MDT)
- Previous NaN loss: Job 45540895 log
- Crops3D normalization: `normalize_crops3d_features.py`
- ScanNet target statistics: From original 3D-LLM paper/codebase
