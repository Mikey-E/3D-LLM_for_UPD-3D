================================================================================
REFORMATTED RESULTS - CROPS3D INFERENCE
================================================================================
Date: October 8, 2025
Reformatting: Changed 'question' → 'prompt' and 'model_answer' → 'response'

================================================================================
OUTPUT FILES
================================================================================

Directory: /project/3dllms/melgin/3D-LLM_for_UPD-3D/crops3d_results_formatted/

Individual Files (8 files, by point cloud range):
  - inference_results_0_45_20251008_145939.json         (540 entries)
  - inference_results_45_90_20251008_150135.json        (540 entries)
  - inference_results_90_135_20251008_150121.json       (540 entries)
  - inference_results_135_180_20251008_150141.json      (540 entries)
  - inference_results_180_225_20251008_150207.json      (540 entries)
  - inference_results_225_270_20251008_150020.json      (540 entries)
  - inference_results_270_315_20251008_145933.json      (540 entries)
  - inference_results_315_357_20251008_145836.json      (504 entries)

Merged File (all results in one):
  - all_results_merged.json                             (4,284 entries, 2.1MB)

================================================================================
JSON SCHEMA
================================================================================

Each entry contains:
{
  "pcl_id": "Cabbage@sl_1109_14",          // Point cloud identifier
  "category": "open_ended",                 // Question category
  "prompt": "What type of rug...",          // ← The question (renamed)
  "question_type": "open_ended",            // Type: open_ended or multiple_choice
  "options": [...],                         // Multiple choice options (if applicable)
  "formatted_question": "What type...",     // Formatted question with options
  "response": "area rug",                   // ← Model answer (renamed)
  "timestamp": "2025-10-08T14:55:51.181400" // When inference was run
}

Key Changes:
  ✓ 'question' renamed to 'prompt'
  ✓ 'model_answer' renamed to 'response'
  ✓ All other fields preserved

================================================================================
STATISTICS
================================================================================

Total Entries: 4,284 (357 point clouds × 12 question categories)

Response Statistics:
  - Non-empty responses: 769 (18.0%)
  - Empty responses:     3,515 (82.0%)
  - Average length:      1.2 words (for non-empty)
  - Median length:       1 word

By Question Type:
  - Open-ended questions:         687/714 (96.2%) have responses
  - Multiple-choice questions:    82/3,570 (2.3%) have responses

================================================================================
USAGE EXAMPLES
================================================================================

Python:
  import json
  
  # Load merged file
  with open('all_results_merged.json', 'r') as f:
      results = json.load(f)
  
  # Access data
  for result in results:
      print(f"Prompt: {result['prompt']}")
      print(f"Response: {result['response']}")

Filter for non-empty responses:
  non_empty = [r for r in results if r['response'].strip()]
  print(f"Found {len(non_empty)} non-empty responses")

Filter by category:
  open_ended = [r for r in results if r['category'] == 'open_ended']
  print(f"Open-ended questions: {len(open_ended)}")

Filter by point cloud:
  cabbage = [r for r in results if 'Cabbage' in r['pcl_id']]
  print(f"Cabbage point clouds: {len(cabbage)}")

================================================================================
SAMPLE ENTRIES
================================================================================

1. Open-ended question with response:
{
  "pcl_id": "Cabbage@sl_1109_14",
  "category": "open_ended",
  "prompt": "What type of rug covers the floor?",
  "question_type": "open_ended",
  "options": [],
  "formatted_question": "What type of rug covers the floor?",
  "response": "area rug",
  "timestamp": "2025-10-08T14:55:58.123456"
}

2. Multiple-choice question (typically empty response):
{
  "pcl_id": "Cotton@mvs_101_05",
  "category": "aad_base",
  "prompt": "What type of pot is the succulent plant sitting in?",
  "question_type": "multiple_choice",
  "options": [
    "A. White-colored pot",
    "B. Transparent pot",
    "C. Clay pot"
  ],
  "formatted_question": "What type of pot is the succulent plant sitting in? A. White-colored pot B. Transparent pot C. Clay pot",
  "response": "",
  "timestamp": "2025-10-08T14:55:51.181400"
}

================================================================================
FUTURE USE
================================================================================

The inference script (03_run_inference.py) has been updated to use 'prompt' 
and 'response' keys for all future runs, ensuring consistency.

================================================================================
