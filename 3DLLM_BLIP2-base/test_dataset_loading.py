#!/usr/bin/env python3
"""
Test script to verify dataset loading for 3D-LLM finetuning.
This script loads a small batch from each dataset to verify paths and data shapes.
"""

import os
import sys
import torch
import json

# Add lavis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lavis"))

from lavis.datasets.datasets.threedvqa_datasets import ThreeDVQADataset, ThreeDVQAEvalDataset
from lavis.processors.blip_processors import BlipQuestionProcessor, Blip2ImageTrainProcessor

def test_dataset(dataset_name, ann_path, dataset_class=ThreeDVQADataset):
    """Test loading a dataset and print statistics."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name}")
    print(f"{'='*60}")
    
    # Check if annotation file exists
    if not os.path.exists(ann_path):
        print(f"❌ ERROR: Annotation file not found: {ann_path}")
        return False
    
    print(f"✓ Annotation file exists: {ann_path}")
    
    # Load annotation to check format
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} annotations")
    
    # Initialize processors (minimal setup)
    text_processor = BlipQuestionProcessor()
    vis_processor = Blip2ImageTrainProcessor(image_size=364)
    
    # Initialize dataset
    try:
        dataset = dataset_class(
            vis_processor=vis_processor,
            text_processor=text_processor,
            vis_root="",  # Not used for 3D datasets
            ann_paths=[ann_path]
        )
        print(f"✓ Dataset initialized with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ ERROR initializing dataset: {e}")
        return False
    
    # Try to load first sample
    try:
        sample = dataset[0]
        print(f"✓ Successfully loaded first sample")
        print(f"  - pc_feat shape: {sample['pc_feat'].shape}")
        print(f"  - pc shape: {sample['pc'].shape}")
        print(f"  - text_input: {sample['text_input'][:100]}...")
        print(f"  - answer: {sample.get('answer', 'N/A')}")
        print(f"  - scene_id: {sample.get('scene_id', 'N/A')}")
        print(f"  - question_id: {sample.get('question_id', 'N/A')}")
        return True
    except FileNotFoundError as e:
        print(f"❌ ERROR loading sample (missing file): {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_directories():
    """Check if required data directories exist."""
    print(f"\n{'='*60}")
    print("Checking Data Directories")
    print(f"{'='*60}")
    
    base_path = "../data/scannet_features"
    feat_path = os.path.join(base_path, "voxelized_features_sam_nonzero_preprocess")
    voxel_path = os.path.join(base_path, "voxelized_voxels_sam_nonzero_preprocess")
    
    feat_exists = os.path.exists(feat_path)
    voxel_exists = os.path.exists(voxel_path)
    
    if feat_exists:
        feat_count = len([f for f in os.listdir(feat_path) if f.endswith('.pt')])
        print(f"✓ Features directory exists: {feat_path}")
        print(f"  - Contains {feat_count} .pt files")
    else:
        print(f"❌ Features directory missing: {feat_path}")
    
    if voxel_exists:
        voxel_count = len([f for f in os.listdir(voxel_path) if f.endswith('.npy')])
        print(f"✓ Voxels directory exists: {voxel_path}")
        print(f"  - Contains {voxel_count} .npy files")
    else:
        print(f"❌ Voxels directory missing: {voxel_path}")
    
    return feat_exists and voxel_exists

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("3D-LLM Dataset Loading Test")
    print("="*60)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check data directories
    if not check_data_directories():
        print("\n❌ Data directories are not properly set up. Please check the paths.")
        return
    
    # Test datasets
    results = {}
    
    # Test ScanQA
    results['ScanQA'] = test_dataset(
        "ScanQA",
        "../data/questions/ScanQA_v1.0/ScanQA_v1.0_train.json",
        ThreeDVQADataset
    )
    
    # Test SQA3D
    results['SQA3D'] = test_dataset(
        "SQA3D",
        "../data/questions/SQA3D/ScanQA_format/SQA_train.json",
        ThreeDVQADataset
    )
    
    # Test 3DMV-VQA
    results['3DMV-VQA'] = test_dataset(
        "3DMV-VQA",
        "../data/questions/3dmv_vqa/questions_only 2/train_questions.json",
        ThreeDVQADataset
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in results.items():
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All tests passed! Dataset setup is complete.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
