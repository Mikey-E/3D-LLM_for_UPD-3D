#!/usr/bin/env python3
"""
Generate GIW529 training data in ScanQA format for finetuning.
Creates JSON files with questions and answers from GIW529 text samples.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Paths
PCL_TRAIN = "/project/3dllms/melgin/UPD-3D/pcl_lists/GIW529_train_minus_val.txt"
PCL_VAL = "/project/3dllms/melgin/UPD-3D/pcl_lists/GIW529_val_subset_of_train.txt"
QUESTIONS_BASE = "/project/3dllms/melgin/UPD-3D/upd_text/GIW529_gpt-5-nano"
ANSWER_KEY = "/project/3dllms/melgin/UPD-3D/answer_keys/GIW529_gpt-5-nano.json"
OUTPUT_DIR = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/GIW529"

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
                "object_ids": [0],  # Dummy value (not used for GIW529)
                "object_names": ["item"]  # Generic name for grocery items
            }
            
            samples.append(sample)
            question_counter += 1
    
    return samples


def main():
    print("=" * 70)
    print("GIW529 FINETUNING DATA GENERATION")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load answer key
    print("Loading answer key...")
    answer_key = load_answer_key(ANSWER_KEY)
    print(f"  ✓ Loaded {len(answer_key)} answers")
    print()
    
    # Verify split files exist
    if not os.path.exists(PCL_TRAIN):
        print(f"ERROR: Train list not found: {PCL_TRAIN}")
        return
    if not os.path.exists(PCL_VAL):
        print(f"ERROR: Val list not found: {PCL_VAL}")
        return
    
    print("Using train/val split:")
    print(f"  Train: {PCL_TRAIN}")
    print(f"  Val: {PCL_VAL}")
    print()
    
    # Generate training data
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
    
    # Save to JSON files
    print("Saving JSON files...")
    train_path = os.path.join(OUTPUT_DIR, "GIW529_train.json")
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"  ✓ Saved: {train_path}")
    print(f"    Size: {len(train_samples)} samples")
    
    val_path = os.path.join(OUTPUT_DIR, "GIW529_val.json")
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2)
    print(f"  ✓ Saved: {val_path}")
    print(f"    Size: {len(val_samples)} samples")
    
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  - GIW529_train.json: {len(train_samples)} samples ({len(train_pcls)} scenes × {len(CATEGORIES)} categories)")
    print(f"  - GIW529_val.json: {len(val_samples)} samples ({len(val_pcls)} scenes × {len(CATEGORIES)} categories)")
    print()
    print("Next steps:")
    print("1. Review the generated JSON files")
    print("2. Create a finetuning config YAML (finetune_giw529.yaml)")
    print("3. Create a finetuning shell script (finetune_giw529.sh)")
    print("4. Start finetuning!")
    print()


if __name__ == "__main__":
    main()
