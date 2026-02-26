#!/usr/bin/env python3
"""
Generate Crops3D training data in ScanQA format for finetuning.
Creates JSON files with questions and answers from Crops3D text samples.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Paths
PCL_TRAIN = "/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_train_minus_val.txt"
PCL_VAL = "/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_val_subset_of_train.txt"
QUESTIONS_BASE = "/project/3dllms/melgin/UPD-3D/upd_text/Crops3D_gpt-5-nano"
ANSWER_KEY = "/cluster/medbow/project/3dllms/melgin/UPD-3D/answer_keys/Crops3D_gpt-5-nano.json"
OUTPUT_DIR = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/Crops3D"

# Question categories (excluding standard_answer)
CATEGORIES = [
    'aad_base', 'aad_additional_instruction', 'aad_additional_option',
    'iasd_base', 'iasd_additional_instruction', 'iasd_additional_option',
    'ivqd_base', 'ivqd_additional_instruction', 'ivqd_additional_option',
    'open_ended', 'open_ended_additional_instruction',
    'standard'
]


def load_answer_key(answer_key_path):
    """Load the answer key JSON file."""
    with open(answer_key_path, 'r') as f:
        return json.load(f)


def load_pcl_list(pcl_list_path):
    """Load list of point cloud IDs."""
    with open(pcl_list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_question_text(pcl_id, category, questions_base):
    """Load question text from file."""
    question_path = os.path.join(questions_base, category, f"{pcl_id}.txt")
    
    if not os.path.exists(question_path):
        return None
    
    with open(question_path, 'r') as f:
        return f.read().strip()


def parse_answer_from_text(question_text, answer_letter):
    """
    Extract the full answer text given the letter.
    
    Args:
        question_text: Full question text with options
        answer_letter: Letter (A, B, C, or D)
    
    Returns:
        Full answer like "C. Green with purple-tinged veins"
    """
    lines = question_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith(f"{answer_letter}."):
            return line
    
    # Fallback: just return the letter
    return answer_letter


def generate_dataset(pcl_list, answer_key, questions_base, split_name):
    """
    Generate dataset in ScanQA format.
    
    Args:
        pcl_list: List of point cloud IDs
        answer_key: Dictionary mapping pcl_id to answer letter
        questions_base: Base path to question text files
        split_name: 'train' or 'val'
    
    Returns:
        List of samples in ScanQA format
    """
    samples = []
    question_counter = 0
    
    for pcl_id in tqdm(pcl_list, desc=f"Processing {split_name}"):
        # Convert @ to _ for scene_id (to match preprocessed filenames)
        scene_id = pcl_id.replace('@', '_')
        
        for category in CATEGORIES:
            # Load question text
            question_text = load_question_text(pcl_id, category, questions_base)
            
            if question_text is None:
                print(f"Warning: Missing question for {pcl_id} in {category}")
                continue
            
            # Determine answer based on category
            if category == 'standard':
                # Use answer key
                if pcl_id not in answer_key:
                    print(f"Warning: No answer key for {pcl_id}")
                    continue
                
                answer_letter = answer_key[pcl_id]
                answer = parse_answer_from_text(question_text, answer_letter)
            else:
                # All other categories: no answer
                answer = "there is no answer"
            
            # Create sample in ScanQA format
            sample = {
                "scene_id": scene_id,
                "question": question_text,
                "answers": [answer],
                "question_id": f"{split_name}-{pcl_id}-{category}-{question_counter}",
                "object_ids": [0],  # Dummy value (not used for Crops3D)
                "object_names": ["plant"]  # Generic name
            }
            
            samples.append(sample)
            question_counter += 1
    
    return samples


def main():
    print("=" * 70)
    print("CROPS3D FINETUNING DATA GENERATION")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load answer key
    print("Loading answer key...")
    answer_key = load_answer_key(ANSWER_KEY)
    print(f"  ✓ Loaded {len(answer_key)} answers")
    print()
    
    # Check if split files exist
    if os.path.exists(PCL_TRAIN) and os.path.exists(PCL_VAL):
        print("Using train/val split:")
        print(f"  Train: {PCL_TRAIN}")
        print(f"  Val: {PCL_VAL}")
        use_split = True
    else:
        print("Split files not found, using single train file:")
        PCL_TRAIN_SINGLE = "/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_train.txt"
        print(f"  Train: {PCL_TRAIN_SINGLE}")
        use_split = False
    print()
    
    # Generate training data
    if use_split:
        print("Generating TRAIN data...")
        train_pcls = load_pcl_list(PCL_TRAIN)
        print(f"  Point clouds: {len(train_pcls)}")
        print(f"  Categories: {len(CATEGORIES)}")
        print(f"  Total samples: {len(train_pcls) * len(CATEGORIES)}")
        train_samples = generate_dataset(train_pcls, answer_key, QUESTIONS_BASE, "train")
        print(f"  ✓ Generated {len(train_samples)} training samples")
        print()
        
        # Generate validation data
        print("Generating VAL data...")
        val_pcls = load_pcl_list(PCL_VAL)
        print(f"  Point clouds: {len(val_pcls)}")
        print(f"  Categories: {len(CATEGORIES)}")
        print(f"  Total samples: {len(val_pcls) * len(CATEGORIES)}")
        val_samples = generate_dataset(val_pcls, answer_key, QUESTIONS_BASE, "val")
        print(f"  ✓ Generated {len(val_samples)} validation samples")
        print()
    else:
        print("Generating TRAIN data (no validation split)...")
        train_pcls = load_pcl_list(PCL_TRAIN_SINGLE)
        print(f"  Point clouds: {len(train_pcls)}")
        print(f"  Categories: {len(CATEGORIES)}")
        print(f"  Total samples: {len(train_pcls) * len(CATEGORIES)}")
        train_samples = generate_dataset(train_pcls, answer_key, QUESTIONS_BASE, "train")
        print(f"  ✓ Generated {len(train_samples)} training samples")
        print()
        val_samples = []
    
    # Save to JSON files
    print("Saving JSON files...")
    train_path = os.path.join(OUTPUT_DIR, "Crops3D_train.json")
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"  ✓ Saved: {train_path}")
    print(f"    Size: {len(train_samples)} samples")
    
    if val_samples:
        val_path = os.path.join(OUTPUT_DIR, "Crops3D_val.json")
        with open(val_path, 'w') as f:
            json.dump(val_samples, f, indent=2)
        print(f"  ✓ Saved: {val_path}")
        print(f"    Size: {len(val_samples)} samples")
    
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the generated JSON files")
    print("2. Create a finetuning config YAML (like finetune_scanqa.yaml)")
    print("3. Update dataset loader to handle Crops3D scene_ids")
    print("4. Start finetuning!")
    print()


if __name__ == "__main__":
    main()
