================================================================================
CROPS3D INFERENCE RESULTS - BY CATEGORY
================================================================================
Date: October 8, 2025
Format: Separate JSON file per category, with pcl_id as keys

================================================================================
OUTPUT FILES
================================================================================

Directory: /project/3dllms/melgin/3D-LLM_for_UPD-3D/crops3d_results_by_category/

12 Category Files (357 point clouds each):

  inf_rslts_3dllm_Crops3D_test_aad_additional_instruction.json    (173 KB)
  inf_rslts_3dllm_Crops3D_test_aad_additional_option.json         (150 KB)
  inf_rslts_3dllm_Crops3D_test_aad_base.json                      (156 KB)
  inf_rslts_3dllm_Crops3D_test_iasd_additional_instruction.json   (180 KB)
  inf_rslts_3dllm_Crops3D_test_iasd_additional_option.json        (157 KB)
  inf_rslts_3dllm_Crops3D_test_iasd_base.json                     (142 KB)
  inf_rslts_3dllm_Crops3D_test_ivqd_additional_instruction.json   (167 KB)
  inf_rslts_3dllm_Crops3D_test_ivqd_additional_option.json        (144 KB)
  inf_rslts_3dllm_Crops3D_test_ivqd_base.json                     (129 KB)
  inf_rslts_3dllm_Crops3D_test_open_ended.json                    ( 97 KB)
  inf_rslts_3dllm_Crops3D_test_open_ended_additional_instruction.json (132 KB)
  inf_rslts_3dllm_Crops3D_test_standard.json                      (151 KB)

Total: 1.8 MB across 12 files

================================================================================
FILE STRUCTURE
================================================================================

Each file contains one question per point cloud (357 entries).

Format:
{
  "CropType@identifier": {
    "prompt": "Question text",
    "response": "Model answer",
    "question_type": "open_ended" or "multiple_choice" or "unknown",
    "timestamp": "2025-10-08T14:55:54.464758",
    "options": [...],              // Optional: for multiple choice
    "formatted_question": "..."    // Optional: question with options inline
  }
}

Point Cloud ID Format: CropType@identifier
Examples:
  - Cabbage@sl_1109_14
  - Cotton@mvs_101_05
  - Tomato@sl_1216_12
  - Wheat@mvs_812_03
  - Maize@sl_1109_02
  - Rapeseed@mvs_901_10

================================================================================
CATEGORY DESCRIPTIONS
================================================================================

AAD (Attribute-Aware Detection):
  - aad_base                          Base AAD questions
  - aad_additional_instruction        AAD with additional instructions
  - aad_additional_option             AAD with additional options

IASD (Instance-Aware Scene Description):
  - iasd_base                         Base IASD questions
  - iasd_additional_instruction       IASD with additional instructions
  - iasd_additional_option            IASD with additional options

IVQD (Instance-Aware VQA Detection):
  - ivqd_base                         Base IVQD questions
  - ivqd_additional_instruction       IVQD with additional instructions
  - ivqd_additional_option            IVQD with additional options

Other:
  - open_ended                        Open-ended questions
  - open_ended_additional_instruction Open-ended with additional instructions
  - standard                          Standard questions

================================================================================
STATISTICS BY CATEGORY
================================================================================

Category                                     Entries  Non-empty Responses
----------------------------------------------------------------------------
aad_additional_instruction                   357      0 (0.0%)
aad_additional_option                        357      13 (3.6%)
aad_base                                     357      7 (2.0%)
iasd_additional_instruction                  357      2 (0.6%)
iasd_additional_option                       357      19 (5.3%)
iasd_base                                    357      13 (3.6%)
ivqd_additional_instruction                  357      0 (0.0%)
ivqd_additional_option                       357      8 (2.2%)
ivqd_base                                    357      9 (2.5%)
open_ended                                   357      343 (96.1%)
open_ended_additional_instruction            357      344 (96.4%)
standard                                     357      11 (3.1%)
----------------------------------------------------------------------------
TOTAL                                        4,284    769 (18.0%)

Note: Open-ended questions have the highest response rate (~96%)
      Multiple-choice questions have low response rate (~0-5%)
      This is due to the model being primarily trained for localization

================================================================================
USAGE EXAMPLES
================================================================================

Python - Load specific category:
  import json
  
  # Load open-ended questions
  with open('inf_rslts_3dllm_Crops3D_test_open_ended.json', 'r') as f:
      open_ended = json.load(f)
  
  # Access specific point cloud
  pcl_id = "Cabbage@sl_1109_14"
  if pcl_id in open_ended:
      print(f"Q: {open_ended[pcl_id]['prompt']}")
      print(f"A: {open_ended[pcl_id]['response']}")

Python - Filter by response availability:
  # Get all entries with non-empty responses
  with_responses = {
      k: v for k, v in open_ended.items() 
      if v['response'].strip()
  }
  print(f"Found {len(with_responses)} answers")

Python - Iterate through all point clouds:
  for pcl_id, data in open_ended.items():
      print(f"{pcl_id}: {data['response']}")

Python - Load multiple categories:
  import json
  
  categories = ['open_ended', 'standard', 'aad_base']
  all_data = {}
  
  for cat in categories:
      with open(f'inf_rslts_3dllm_Crops3D_test_{cat}.json', 'r') as f:
          all_data[cat] = json.load(f)
  
  # Access specific point cloud across categories
  pcl_id = "Tomato@sl_1216_12"
  for cat, data in all_data.items():
      if pcl_id in data:
          print(f"{cat}: {data[pcl_id]['response']}")

================================================================================
SAMPLE ENTRIES
================================================================================

1. Open-ended question (with response):
File: inf_rslts_3dllm_Crops3D_test_open_ended.json
{
  "Cabbage@sl_1109_14": {
    "prompt": "What type of rug covers the floor?",
    "response": "area rug",
    "question_type": "open_ended",
    "timestamp": "2025-10-08T14:55:54.464758",
    "formatted_question": "What type of rug covers the floor?"
  }
}

2. Multiple-choice question (typically empty response):
File: inf_rslts_3dllm_Crops3D_test_aad_base.json
{
  "Cotton@mvs_101_05": {
    "prompt": "What type of pot is the succulent plant sitting in?",
    "response": "",
    "question_type": "multiple_choice",
    "timestamp": "2025-10-08T14:55:51.181400",
    "options": [
      "A. White-colored pot",
      "B. Transparent pot",
      "C. Clay pot"
    ],
    "formatted_question": "What type of pot is the succulent plant sitting in? A. White-colored pot B. Transparent pot C. Clay pot"
  }
}

3. Standard question:
File: inf_rslts_3dllm_Crops3D_test_standard.json
{
  "Maize@sl_1109_02": {
    "prompt": "What type of pot is the succulent plant sitting in? A. Dark-colored pot B. Transparent pot C. Clay pot",
    "response": "",
    "question_type": "multiple_choice",
    "timestamp": "2025-10-08T14:56:02.123456",
    "options": [
      "A. Dark-colored pot",
      "B. Transparent pot",
      "C. Clay pot"
    ],
    "formatted_question": "What type of pot is the succulent plant sitting in? A. Dark-colored pot B. Transparent pot C. Clay pot"
  }
}

================================================================================
COMPARISON WITH OTHER MODELS
================================================================================

This format matches other models' output structure:
  - inf_rslts_gplm_3D-FRONT_test_<category>.json  (other model)
  - inf_rslts_3dllm_Crops3D_test_<category>.json  (this model)

Key format:
  - Other models: identifier@scene (e.g., "025c66d0-3114-434a-a7b4-51ea83c2eaf4@SecondBedroom-32499")
  - This model:   CropType@identifier (e.g., "Maize@3-10", "Cabbage@sl_1109_14")

Both use the same internal structure: {"prompt": ..., "response": ..., ...}

================================================================================
