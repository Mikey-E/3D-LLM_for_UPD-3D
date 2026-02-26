#!/usr/bin/env python3
"""
Test Crops3D dataset loading for finetuning.
Verifies JSON format, feature file existence, and data loading.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path

# Paths
TRAIN_JSON = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/Crops3D/Crops3D_train.json"
VAL_JSON = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/data/questions/Crops3D/Crops3D_val.json"
FEATURE_DIR = "/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed"

def test_json_format():
    """Test JSON file format."""
    print("=" * 70)
    print("TEST 1: JSON Format")
    print("=" * 70)
    
    # Load training JSON
    with open(TRAIN_JSON, 'r') as f:
        train_data = json.load(f)
    
    # Load validation JSON
    with open(VAL_JSON, 'r') as f:
        val_data = json.load(f)
    
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Validation samples: {len(val_data)}")
    print()
    
    # Check required fields
    required_fields = ["scene_id", "question", "answers", "question_id", "object_ids", "object_names"]
    sample = train_data[0]
    
    print("Checking required fields:")
    for field in required_fields:
        if field in sample:
            print(f"  ✓ {field}")
        else:
            print(f"  ✗ {field} MISSING")
    print()
    
    # Print sample
    print("Sample entry:")
    print(f"  scene_id: {sample['scene_id']}")
    print(f"  question: {sample['question'][:80]}...")
    print(f"  answers: {sample['answers']}")
    print(f"  question_id: {sample['question_id']}")
    print()
    
    return train_data, val_data


def test_feature_files(train_data, val_data):
    """Test feature file existence."""
    print("=" * 70)
    print("TEST 2: Feature Files")
    print("=" * 70)
    
    # Get unique scene IDs
    all_scene_ids = set()
    for sample in train_data + val_data:
        all_scene_ids.add(sample['scene_id'])
    
    print(f"Unique scene IDs: {len(all_scene_ids)}")
    print()
    
    # Check feature file existence
    missing_pt = []
    missing_npy = []
    found_pt = 0
    found_npy = 0
    
    for scene_id in sorted(all_scene_ids):
        pt_file = os.path.join(FEATURE_DIR, f"{scene_id}.pt")
        npy_file = os.path.join(FEATURE_DIR, f"{scene_id}.npy")
        
        if os.path.exists(pt_file):
            found_pt += 1
        else:
            missing_pt.append(scene_id)
        
        if os.path.exists(npy_file):
            found_npy += 1
        else:
            missing_npy.append(scene_id)
    
    print(f"Feature files (.pt): {found_pt}/{len(all_scene_ids)}")
    print(f"Coordinate files (.npy): {found_npy}/{len(all_scene_ids)}")
    print()
    
    if missing_pt:
        print(f"⚠ Missing .pt files: {len(missing_pt)}")
        print(f"  Examples: {missing_pt[:5]}")
        print()
    
    if missing_npy:
        print(f"⚠ Missing .npy files: {len(missing_npy)}")
        print(f"  Examples: {missing_npy[:5]}")
        print()
    
    if not missing_pt and not missing_npy:
        print("✓ All feature files exist!")
    print()
    
    return list(all_scene_ids), missing_pt, missing_npy


def test_data_loading(scene_ids):
    """Test loading actual data."""
    print("=" * 70)
    print("TEST 3: Data Loading")
    print("=" * 70)
    
    # Test loading a sample
    scene_id = scene_ids[0]
    print(f"Testing with scene_id: {scene_id}")
    print()
    
    pt_file = os.path.join(FEATURE_DIR, f"{scene_id}.pt")
    npy_file = os.path.join(FEATURE_DIR, f"{scene_id}.npy")
    
    try:
        # Load features
        pc_feat = torch.load(pt_file, map_location="cpu")
        print(f"✓ Loaded features: {pt_file}")
        print(f"  Shape: {pc_feat.shape}")
        print(f"  Dtype: {pc_feat.dtype}")
        print()
        
        # Load coordinates
        pc = np.load(npy_file)
        print(f"✓ Loaded coordinates: {npy_file}")
        print(f"  Shape: {pc.shape}")
        print(f"  Dtype: {pc.dtype}")
        print()
        
        # Check dimensions
        if pc_feat.shape[0] != pc.shape[0]:
            print(f"⚠ WARNING: Feature and coordinate shapes don't match!")
            print(f"  Features: {pc_feat.shape[0]} points")
            print(f"  Coordinates: {pc.shape[0]} points")
        else:
            print(f"✓ Matching point counts: {pc_feat.shape[0]}")
        print()
        
        # Check feature dimension
        if pc_feat.shape[1] == 1408:
            print(f"✓ Feature dimension correct: 1408")
        else:
            print(f"⚠ WARNING: Unexpected feature dimension: {pc_feat.shape[1]} (expected 1408)")
        print()
        
        # Check coordinate dimension
        if pc.shape[1] == 3:
            print(f"✓ Coordinate dimension correct: 3 (x, y, z)")
        else:
            print(f"⚠ WARNING: Unexpected coordinate dimension: {pc.shape[1]} (expected 3)")
        print()
        
        print("✓ Data loading test passed!")
        
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
    
    print()


def test_category_distribution(train_data):
    """Test distribution of question categories."""
    print("=" * 70)
    print("TEST 4: Category Distribution")
    print("=" * 70)
    
    # Count categories
    category_counts = {}
    for sample in train_data:
        if 'category' in sample:
            category = sample['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    if category_counts:
        print("Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category:40s}: {count:5d} samples")
        print()
        
        # Check if balanced
        counts = list(category_counts.values())
        if len(set(counts)) == 1:
            print(f"✓ All categories have equal samples: {counts[0]}")
        else:
            print(f"⚠ Unbalanced categories:")
            print(f"  Min: {min(counts)} samples")
            print(f"  Max: {max(counts)} samples")
    else:
        print("⚠ No 'category' field found in samples")
    
    print()


def main():
    print()
    print("=" * 70)
    print("CROPS3D FINETUNING DATA TEST")
    print("=" * 70)
    print()
    
    # Test 1: JSON format
    train_data, val_data = test_json_format()
    
    # Test 2: Feature files
    scene_ids, missing_pt, missing_npy = test_feature_files(train_data, val_data)
    
    # Test 3: Data loading (if files exist)
    if not missing_pt and not missing_npy:
        test_data_loading(scene_ids)
    else:
        print("=" * 70)
        print("TEST 3: Data Loading")
        print("=" * 70)
        print("⚠ Skipping data loading test due to missing feature files")
        print()
    
    # Test 4: Category distribution
    test_category_distribution(train_data)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Unique scenes: {len(scene_ids)}")
    print(f"Missing .pt files: {len(missing_pt)}")
    print(f"Missing .npy files: {len(missing_npy)}")
    print()
    
    if not missing_pt and not missing_npy:
        print("✓ All tests passed! Ready for finetuning.")
    else:
        print("⚠ Some feature files are missing. Check the preprocessing step.")
    print()


if __name__ == "__main__":
    main()
