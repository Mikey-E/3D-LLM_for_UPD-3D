# Crops3D Finetuning Data Investigation

## Data Available

### 1. Point Cloud Lists
- **Training:** `/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_train.txt`
  - 822 training samples
  - Format: `CropType@filename` (e.g., `Cabbage@mvs_1123_06`)

- **Validation:** `/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_val_subset_of_train.txt`
- **Test:** `/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_test.txt` (357 samples)

### 2. Text Samples (Questions)
- **Location:** `/project/3dllms/melgin/UPD-3D/upd_text/Crops3D_gpt-5-nano/`
- **Categories:** 12 question types (excluding `standard_answer`)
  - aad_base, aad_additional_instruction, aad_additional_option
  - iasd_base, iasd_additional_instruction, iasd_additional_option
  - ivqd_base, ivqd_additional_instruction, ivqd_additional_option
  - open_ended, open_ended_additional_instruction
  - standard
- **Format:** One `.txt` file per point cloud per category
- **Example:** `Cabbage@mvs_1005_01.txt` contains a question (no answer)
- **Answers:** Stored in `standard_answer/` folder (has correct answer option)

### 3. Preprocessed Point Clouds
- **Location:** `/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed/`
- **Files:** 714 total (357 × 2 = .pt features + .npy coordinates)
- **Status:** ✅ Already preprocessed for inference!

### 4. Original Point Clouds
- **Location:** `/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D/`
- **Structure:** Organized by crop type folders
- **Format:** PLY files

---

## ScanQA Training Format (Target)

### JSON Structure Required:
```json
[
    {
        "answers": ["answer text"],
        "object_ids": [8],
        "object_names": ["object"],
        "question": "Question text?",
        "question_id": "train-scene0000-0",
        "scene_id": "scene0000_00"
    },
    ...
]
```

### Key Fields:
1. **`scene_id`**: Identifier matching feature files (e.g., `scene0000_00`)
2. **`question`**: The question text
3. **`answers`**: List of answer strings
4. **`question_id`**: Unique identifier
5. **`object_ids`**: Optional (can use dummy values)
6. **`object_names`**: Optional (can use dummy values)

### Feature Files Expected:
- **Features:** `{scene_id}.pt` - torch tensor [N, 1408]
- **Coordinates:** `{scene_id}.npy` - numpy array [N, 3]

---

## Key Questions & Issues

### ❓ Question 1: Answer Extraction
**Problem:** Crops3D text files contain multiple-choice questions but don't explicitly state the correct answer.

**Example:**
- `aad_base/Cabbage@mvs_1005_01.txt`: "What color are the leaves...? A. Bright pink B. Red with yellow stripes"
- `standard_answer/Cabbage@mvs_1005_01.txt`: Same question with options "A, B, C, D"

**Question:** How do we determine the correct answer?
- Option A: Use the `standard_answer` version (has all options including correct one)
- Option B: Parse both files and find the difference?
- Option C: Is there a separate answer key file?
- Option D: Should we train on the question only (open-ended)?

### ❓ Question 2: Training Sample Count
**Data:**
- 822 training point clouds
- 12 question categories per cloud
- = 9,864 total training questions

**Question:** Should we:
- Use ALL 12 categories mixed together?
- Focus on specific categories (e.g., just `standard`)?
- Sample a subset to match ScanQA's 25k samples?

### ❓ Question 3: Validation Split
**Available:**
- `Crops3D_val_subset_of_train.txt` exists
- Could also use a portion of training

**Question:** What validation strategy do you prefer?

### ❓ Question 4: Feature File Naming
**Current:** `Cabbage_mvs_1005_01.pt` (underscore instead of @)
**ScanQA:** `scene0000_00.pt`

**Question:** Do we need to:
- Update the dataset loader to handle `_` in filenames?
- Or modify how we reference scenes?

### ❓ Question 5: Multiple-Choice vs Open-Ended
**Current format:** Multiple choice with options
**ScanQA format:** Open-ended answers

**Question:** Should we:
- Convert to open-ended (just the answer text)?
- Keep multiple-choice format?
- Train on both?

---

## Proposed Approach (Pending Your Answers)

### Option A: Simple Loader (No Persistent Files)
```python
class Crops3DVQADataset(VQADataset):
    def __init__(self, ...):
        # Load from text files on-the-fly
        # Parse questions and answers dynamically
        # Use existing preprocessed features
```

### Option B: Generate JSON Once
```python
# Create: data/questions/Crops3D/Crops3D_train.json
# Format matches ScanQA exactly
# Run once, then train normally
```

---

## What I Need From You

1. **Answer format:** How to extract correct answers from the text files?
2. **Categories:** Which question categories to include in training?
3. **Sample strategy:** All 9,864 samples or subset?
4. **Output format:** Multiple-choice or open-ended answers?
5. **Validation split:** Use existing val list or create from train?

Once I have these answers, I can create the appropriate data loader or conversion script!
