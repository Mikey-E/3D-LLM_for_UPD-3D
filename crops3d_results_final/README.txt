================================================================================
FINAL CROPS3D INFERENCE RESULTS
================================================================================
Date: October 8, 2025
Format: identifier@scene as dictionary keys

================================================================================
OUTPUT FILES
================================================================================

Directory: /project/3dllms/melgin/3D-LLM_for_UPD-3D/crops3d_results_final/

Available Formats:

1. crops3d_results_nested.json (1.9 MB)
   Structure: pcl_id -> category -> {prompt, response, ...}
   Top-level keys: 357 (one per point cloud)
   
   Example:
   {
     "Cabbage@sl_1109_14": {
       "open_ended": {
         "prompt": "What type of rug covers the floor?",
         "response": "area rug",
         "question_type": "open_ended",
         "timestamp": "2025-10-08T14:55:58.123456"
       },
       "aad_base": {
         "prompt": "What type of pot is the succulent plant sitting in?",
         "response": "",
         "question_type": "multiple_choice",
         "options": ["A. White-colored pot", "B. Transparent pot", "C. Clay pot"],
         "timestamp": "2025-10-08T14:55:51.181400"
       },
       ... (10 more categories)
     }
   }

2. crops3d_results_flat.json (1.9 MB)
   Structure: pcl_id@category -> {prompt, response, ...}
   Top-level keys: 4,284 (one per question)
   
   Example:
   {
     "Cabbage@sl_1109_14@open_ended": {
       "prompt": "What type of rug covers the floor?",
       "response": "area rug",
       "question_type": "open_ended",
       "timestamp": "2025-10-08T14:55:58.123456"
     },
     "Cabbage@sl_1109_14@aad_base": {
       "prompt": "What type of pot is the succulent plant sitting in?",
       "response": "",
       "question_type": "multiple_choice",
       "options": ["A. White-colored pot", "B. Transparent pot", "C. Clay pot"],
       "timestamp": "2025-10-08T14:55:51.181400"
     }
   }

================================================================================
KEY STRUCTURE
================================================================================

Point Cloud IDs follow the pattern: CropType@identifier
Examples:
  - Cabbage@sl_1109_14
  - Cotton@mvs_101_05
  - Tomato@sl_1216_12
  - Wheat@mvs_812_03
  - Maize@sl_1109_02
  - Rapeseed@mvs_901_10

Question Categories (12 per point cloud):
  - aad_base                          (Attribute-aware Detection - base)
  - aad_additional_instruction        (AAD with additional instruction)
  - aad_additional_option             (AAD with additional options)
  - iasd_base                         (Instance-aware Scene Description - base)
  - iasd_additional_instruction       (IASD with additional instruction)
  - iasd_additional_option            (IASD with additional options)
  - ivqd_base                         (Instance-aware VQA Detection - base)
  - ivqd_additional_instruction       (IVQD with additional instruction)
  - ivqd_additional_option            (IVQD with additional options)
  - open_ended                        (Open-ended question)
  - open_ended_additional_instruction (Open-ended with additional instruction)
  - standard                          (Standard question)

================================================================================
ENTRY SCHEMA
================================================================================

Each entry contains:
{
  "prompt": str,              // The question text
  "response": str,            // Model's answer (may be empty)
  "question_type": str,       // "multiple_choice", "open_ended", or "unknown"
  "timestamp": str,           // ISO format: "2025-10-08T14:55:51.181400"
  "options": list[str],       // (Optional) Multiple choice options
  "formatted_question": str   // (Optional) Question with options inline
}

================================================================================
STATISTICS
================================================================================

Total Point Clouds: 357
Total Questions:    4,284 (357 × 12 categories)

Response Statistics:
  - Non-empty responses: 769 (18.0%)
  - Empty responses:     3,515 (82.0%)
  
By Question Type:
  - Open-ended:              687/714 (96.2%) have responses
  - Multiple-choice:         82/3,570 (2.3%) have responses

Answer Characteristics:
  - Median length:           1 word
  - Average length:          1.2 words
  - Max length:              8 words
  - Most answers are very brief (e.g., "area rug", "copper", "none")

================================================================================
USAGE EXAMPLES
================================================================================

Python - Nested Format:
  import json
  
  with open('crops3d_results_nested.json', 'r') as f:
      data = json.load(f)
  
  # Access all questions for a specific point cloud
  pcl_id = "Cabbage@sl_1109_14"
  all_questions = data[pcl_id]
  
  # Access a specific category
  open_ended = data[pcl_id]["open_ended"]
  print(f"Q: {open_ended['prompt']}")
  print(f"A: {open_ended['response']}")
  
  # Iterate through all point clouds
  for pcl_id, categories in data.items():
      for category, qa in categories.items():
          if qa['response'].strip():  # Only non-empty
              print(f"{pcl_id} - {category}: {qa['response']}")

Python - Flat Format:
  import json
  
  with open('crops3d_results_flat.json', 'r') as f:
      data = json.load(f)
  
  # Direct access with composite key
  key = "Cabbage@sl_1109_14@open_ended"
  qa = data[key]
  print(f"Q: {qa['prompt']}")
  print(f"A: {qa['response']}")
  
  # Filter by point cloud
  pcl_id = "Cabbage@sl_1109_14"
  pcl_questions = {k: v for k, v in data.items() if k.startswith(pcl_id + "@")}
  
  # Filter by category
  category = "open_ended"
  category_questions = {k: v for k, v in data.items() if k.endswith("@" + category)}
  
  # Get all non-empty responses
  with_responses = {k: v for k, v in data.items() if v['response'].strip()}
  print(f"Found {len(with_responses)} questions with answers")

================================================================================
WHICH FORMAT TO USE?
================================================================================

Use NESTED format if:
  - You want to work with all questions for a specific point cloud
  - You're analyzing per-scene behavior
  - You prefer hierarchical organization
  - File size: 357 top-level keys (one per point cloud)

Use FLAT format if:
  - You want quick lookup by specific question
  - You're filtering/searching across all questions
  - You prefer simpler dictionary access
  - File size: 4,284 top-level keys (one per question)

Both formats contain identical data, just organized differently.

================================================================================
SAMPLE COMPLETE ENTRIES
================================================================================

Nested Format - Point Cloud with Multiple Questions:
{
  "Tomato@sl_1216_12": {
    "open_ended": {
      "prompt": "What type of metal is the lamp made of?",
      "response": "copper",
      "question_type": "open_ended",
      "timestamp": "2025-10-08T14:56:15.123456"
    },
    "open_ended_additional_instruction": {
      "prompt": "What type of metal is the lamp made of? If the question is unanswerable, answer: 'F'",
      "response": "Copper",
      "question_type": "open_ended",
      "timestamp": "2025-10-08T14:56:15.789012"
    },
    "aad_base": {
      "prompt": "What type of pot is the succulent plant sitting in?",
      "response": "",
      "question_type": "multiple_choice",
      "options": ["A. White-colored pot", "B. Transparent pot", "C. Clay pot"],
      "formatted_question": "What type of pot is the succulent plant sitting in? A. White-colored pot B. Transparent pot C. Clay pot",
      "timestamp": "2025-10-08T14:56:16.345678"
    }
  }
}

Flat Format - Individual Question:
{
  "Tomato@sl_1216_12@open_ended": {
    "prompt": "What type of metal is the lamp made of?",
    "response": "copper",
    "question_type": "open_ended",
    "timestamp": "2025-10-08T14:56:15.123456"
  }
}

================================================================================
