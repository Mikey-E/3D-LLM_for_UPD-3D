# Dataset Setup Summary

## Status: ✅ ScanNet Datasets Ready for Finetuning

Generated: $(date)

---

## Completed Tasks

### 1. ✅ ScanNet Features Setup
- **voxelized_features_sam_nonzero_preprocess**: 1494 .pt files (217GB)
- **voxelized_voxels_sam_nonzero_preprocess**: 1494 .npy files (944MB)
- Both directories properly placed in `data/scannet_features/`
- All required scene files present and verified

### 2. ✅ Question Data Downloaded
- **ScanQA**: `data/questions/ScanQA_v1.0/` (6.3MB)
  - ScanQA_v1.0_train.json: 25,563 questions
  - ScanQA_v1.0_val.json: Validation split
- **SQA3D**: `data/questions/SQA3D/ScanQA_format/` (13MB)
  - SQA_train.json: 26,623 questions
  - SQA_test.json: Test split
  - SQA_val.json: Validation split
- **3DMV-VQA**: `data/questions/3dmv_vqa/questions_only 2/` (9.5MB)
  - train_questions.json: 40,495 questions (Matterport3D scenes)
  - *Note: Requires separate 3DMV-VQA features (not yet downloaded)*

### 3. ✅ Pretrained Checkpoint
- **Location**: `checkpoints/pretrain_blip2_sam_flant5xl_v2.pth` (4.2GB)
- Status: Downloaded and verified

### 4. ✅ Dataset Code Updated
- Fixed `threedvqa_datasets.py`:
  - Training dataset: Uses `../data/scannet_features/`
  - Evaluation dataset: Fixed from `examples/` to `../data/scannet_features/`
- All paths now correctly point to data directories

### 5. ✅ Config Files Updated
- **finetune_scanqa.yaml**: ✅ Ready
  - Annotations: `../data/questions/ScanQA_v1.0/`
  - Checkpoint: `../checkpoints/pretrain_blip2_sam_flant5xl_v2.pth`
  
- **finetune_sqa.yaml**: ✅ Ready
  - Annotations: `../data/questions/SQA3D/ScanQA_format/`
  - Checkpoint: `../checkpoints/pretrain_blip2_sam_flant5xl_v2.pth`
  
- **finetune_3dmvvqa.yaml**: ⚠️ Config updated but features missing
  - Annotations: `../data/questions/3dmv_vqa/questions_only 2/`
  - Checkpoint: `../checkpoints/pretrain_blip2_sam_flant5xl_v2.pth`
  - *Requires: Download 3DMV-VQA features from Google Drive*

### 6. ✅ Dataset Loading Verified
Created `test_dataset_loading.py` and confirmed:
- **ScanQA**: ✅ PASSED
  - 24,969 valid samples loaded
  - Data shapes: pc_feat [5000, 1408], pc [5000, 3]
  - Sample question verified
  
- **SQA3D**: ✅ PASSED
  - 26,182 valid samples loaded
  - Data shapes: pc_feat [5000, 1408], pc [5000, 3]
  - Sample question verified
  
- **3DMV-VQA**: ❌ No matching features (expected - uses Matterport3D scenes)
  - 0 samples (scene IDs don't match ScanNet files)

---

## Ready for Training

### ScanQA Finetuning
```bash
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base
conda activate lavis

# Single GPU training (adjust world_size in config to 1)
python train.py --cfg-path lavis/projects/blip2/train/finetune_scanqa.yaml

# Multi-GPU training (16 GPUs as configured)
# Use SLURM or similar for distributed training
```

### SQA3D Finetuning
```bash
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base
conda activate lavis

python train.py --cfg-path lavis/projects/blip2/train/finetune_sqa.yaml
```

---

## Pending Tasks

### 3DMV-VQA Dataset (Optional)
If you want to finetune on 3DMV-VQA, you need to:

1. **Download 3DMV-VQA features** from [Google Drive](https://drive.google.com/drive/folders/1NdFKKn_IZxGezi6fXA60rF1uxTOmhOet?usp=drive_link)
   - These are features for Matterport3D scenes (different from ScanNet)
   
2. **Extract and organize** the features similar to ScanNet structure

3. **Update paths** in `threedvqa_datasets.py` or create a separate dataset class

Note: The README mentions "3DMV-VQA data will be further updated for a clearer structure"

---

## File Structure
```
3D-LLM_for_UPD-3D/
├── checkpoints/
│   └── pretrain_blip2_sam_flant5xl_v2.pth (4.2GB) ✅
├── data/
│   ├── questions/
│   │   ├── ScanQA_v1.0/ ✅
│   │   ├── SQA3D/ScanQA_format/ ✅
│   │   └── 3dmv_vqa/questions_only 2/ ✅ (questions only)
│   └── scannet_features/
│       ├── voxelized_features_sam_nonzero_preprocess/ (1494 .pt files) ✅
│       ├── voxelized_voxels_sam_nonzero_preprocess/ (1494 .npy files) ✅
│       └── scannet_only/ (compressed archives)
└── 3DLLM_BLIP2-base/
    ├── train.py
    ├── test_dataset_loading.py ✅
    └── lavis/
        ├── datasets/datasets/threedvqa_datasets.py ✅
        └── projects/blip2/train/
            ├── finetune_scanqa.yaml ✅
            ├── finetune_sqa.yaml ✅
            └── finetune_3dmvvqa.yaml ⚠️
```

---

## Verification

Run the test script anytime to verify setup:
```bash
cd /project/3dllms/melgin/3D-LLM_for_UPD-3D/3DLLM_BLIP2-base
conda activate lavis
python test_dataset_loading.py
```

Expected output:
- ✅ ScanQA: PASSED
- ✅ SQA3D: PASSED
- ⚠️ 3DMV-VQA: FAILED (expected without Matterport features)

---

## Notes

1. **Training Configuration**: Current configs are set for distributed training with `world_size: 16`. Adjust this in the YAML files if using fewer GPUs.

2. **Batch Size**: Set to 2 per GPU. With 16 GPUs, effective batch size is 32.

3. **Data Filtering**: The dataset automatically filters out samples where scene files are missing, which is why:
   - ScanQA: 25,563 annotations → 24,969 samples (some scenes not in ScanNet features)
   - SQA3D: 26,623 annotations → 26,182 samples (some scenes not in ScanNet features)

4. **Output Directory**: Training outputs will be saved to `output/BLIP2/3DQA/`

5. **GPU Memory**: With pc_feat=[5000, 1408] and batch_size=2, ensure sufficient GPU memory (likely needs >20GB per GPU)
