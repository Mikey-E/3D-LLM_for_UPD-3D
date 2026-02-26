================================================================================
UNIFIED PIPELINE FOR CROPS3D AND 3D-FRONT DATASETS
================================================================================
Date: October 8, 2025
Model: 3D-LLM BLIP2-FlanT5-XL v2 (pretrained)

================================================================================
OVERVIEW
================================================================================

This unified pipeline handles inference for both Crops3D and 3D-FRONT datasets
using the same codebase. The pipeline automatically detects and handles the
different PLY formats and directory structures.

================================================================================
DATASET DIFFERENCES
================================================================================

Crops3D:
  - Point clouds: 357
  - PLY format: float xyz, ushort/uchar RGB, no normals
  - Path structure: CropType/filename.ply
  - Questions: /UPD-3D/upd_text/Crops3D_gpt-5-nano/
  
3D-FRONT:
  - Point clouds: 2,992
  - PLY format: double xyz, double normals, uchar RGB
  - Path structure: identifier/scene/scene.ply
  - Questions: /UPD-3D/upd_text/3D-FRONT/

Both datasets:
  - 12 question categories each
  - Same model and preprocessing pipeline
  - Same output format (by-category JSON files)

================================================================================
PIPELINE SCRIPTS
================================================================================

00_validate.py
  - Validates dataset paths, PLY formats, and question files
  - Tests both datasets before running full pipeline
  - Usage: python3 unified_pipeline/00_validate.py

01_preprocess.py
  - Converts PLY files to 3D-LLM format (features + coordinates)
  - Handles both Crops3D and 3D-FRONT PLY formats automatically
  - Usage: python3 unified_pipeline/01_preprocess.py --dataset [Crops3D|3D-FRONT] \
           --start_idx 0 --end_idx 100

02_run_inference.py
  - Runs inference on preprocessed point clouds
  - Generates results by category
  - Usage: python3 unified_pipeline/02_run_inference.py --dataset [Crops3D|3D-FRONT] \
           --start_idx 0 --end_idx 100

ply_loader.py
  - Unified PLY loader that auto-detects format
  - Supports:
    - Crops3D: float xyz, ushort/uchar RGB
    - 3D-FRONT: double xyz, double normals, uchar RGB

================================================================================
SLURM SCRIPTS
================================================================================

For Crops3D (from crops3d_pipeline/):
  - run_preprocess_array.sh: 8 array jobs, ~45 PCLs each
  - run_inference_array.sh:  8 array jobs, ~45 PCLs each
  
For 3D-FRONT (from unified_pipeline/):
  - run_preprocess_3dfront.sh: 30 array jobs, ~100 PCLs each
  - run_inference_3dfront.sh:  30 array jobs, ~100 PCLs each

================================================================================
RUNNING THE PIPELINE FOR 3D-FRONT
================================================================================

Step 1: Validate
  cd /project/3dllms/melgin/3D-LLM_for_UPD-3D
  python3 unified_pipeline/00_validate.py

Step 2: Preprocess (SLURM)
  sbatch unified_pipeline/run_preprocess_3dfront.sh
  
  Monitor:
    squeue -u melgin
    ls /cluster/medbow/project/3dllms/melgin/datasets/3D-FRONT_processed/ | wc -l
  
  Expected output:
    - 2,992 .pt files (features)
    - 2,992 .npy files (coordinates)
    - Total: 5,984 files

Step 3: Inference (SLURM)
  sbatch unified_pipeline/run_inference_3dfront.sh
  
  Monitor:
    squeue -u melgin
    ls results/3dfront_inference/*.json | wc -l
  
  Expected output:
    - 30 jobs × 12 categories = 360 JSON files (one per job per category)

Step 4: Merge results by category
  - Results need to be merged from 30 jobs into 12 final files
  - Script TBD (similar to Crops3D post-processing)

================================================================================
OUTPUT FORMAT
================================================================================

Results are saved by category, matching the format from Crops3D:

  inf_rslts_3dllm_3D-FRONT_test_<category>_<start>_<end>_<timestamp>.json

Each file contains:
{
  "identifier@scene": {
    "prompt": "Question text",
    "response": "Model answer",
    "timestamp": "2025-10-08T15:30:00.000000"
  },
  ...
}

Categories:
  - aad_base, aad_additional_instruction, aad_additional_option
  - iasd_base, iasd_additional_instruction, iasd_additional_option
  - ivqd_base, ivqd_additional_instruction, ivqd_additional_option
  - open_ended, open_ended_additional_instruction
  - standard

================================================================================
RERUNNING CROPS3D WITH UNIFIED PIPELINE
================================================================================

The original crops3d_pipeline/ scripts still work, but you can also use the
unified pipeline:

  python3 unified_pipeline/01_preprocess.py --dataset Crops3D --start_idx 0 --end_idx 357
  python3 unified_pipeline/02_run_inference.py --dataset Crops3D --start_idx 0 --end_idx 357

This is useful after fine-tuning to test on both datasets with the same code.

================================================================================
DIRECTORY STRUCTURE
================================================================================

unified_pipeline/
├── 00_validate.py                    Validation script
├── 01_preprocess.py                  Unified preprocessing
├── 02_run_inference.py               Unified inference
├── ply_loader.py                     Adaptive PLY loader
├── run_preprocess_3dfront.sh         SLURM preprocessing script
├── run_inference_3dfront.sh          SLURM inference script
├── logs/                             SLURM output logs
└── README.txt                        This file

Processed data:
├── /cluster/.../Crops3D_processed/   Crops3D features & coords
└── /cluster/.../3D-FRONT_processed/  3D-FRONT features & coords

Results:
├── results/crops3d_inference/        Crops3D inference results
└── results/3dfront_inference/        3D-FRONT inference results

================================================================================
ESTIMATED COMPUTE TIME
================================================================================

Crops3D (357 point clouds):
  - Preprocessing: ~10 min (8 jobs in parallel)
  - Inference:     ~30 min (8 jobs in parallel)
  - Total:         ~40 minutes

3D-FRONT (2,992 point clouds):
  - Preprocessing: ~40 min (30 jobs in parallel)
  - Inference:     ~60 min (30 jobs in parallel)
  - Total:         ~1.7 hours

================================================================================
TROUBLESHOOTING
================================================================================

If preprocessing fails:
  - Check logs in unified_pipeline/logs/
  - Verify conda environment is activated
  - Check GPU availability: nvidia-smi
  - Verify PLY files exist at expected paths

If inference fails:
  - Ensure preprocessing completed successfully
  - Check that .pt and .npy files exist in processed directory
  - Verify question files exist for all categories
  - Check model checkpoint path

Location tokens in output:
  - Normal behavior - model is trained for localization
  - Post-processing script removes <loc> tokens automatically
  - Open-ended questions have ~96% text responses
  - Multiple-choice questions have ~0-5% text responses

================================================================================
