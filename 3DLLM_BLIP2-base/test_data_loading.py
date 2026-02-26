#!/usr/bin/env python3
"""
Test data loading for 3D-LLM finetuning
This script tests if the dataset can be loaded correctly before starting training
"""
import os
import sys
import json
import torch
from omegaconf import OmegaConf

# Add lavis to path
sys.path.insert(0, os.path.dirname(__file__))

from lavis.common.registry import registry
from lavis.datasets.builders import load_dataset

def test_scanqa_loading():
    """Test ScanQA dataset loading"""
    print("=" * 70)
    print("Testing ScanQA Dataset Loading")
    print("=" * 70)
    
    # Load config
    config_path = "lavis/projects/blip2/train/finetune_scanqa.yaml"
    cfg = OmegaConf.load(config_path)
    
    print(f"\n✓ Config loaded from: {config_path}")
    
    # Check annotation files
    annotations = cfg.datasets['3d_vqa']['build_info']['annotations']
    for split, info in annotations.items():
        ann_path = info['storage']
        exists = os.path.exists(ann_path)
        print(f"  {split:6s}: {ann_path}")
        print(f"           {'✓ EXISTS' if exists else '✗ MISSING'}")
        if exists:
            with open(ann_path, 'r') as f:
                data = json.load(f)
                print(f"           {len(data)} samples")
    
    # Test dataset creation
    print("\nAttempting to create dataset...")
    try:
        dataset = load_dataset("3d_vqa", cfg=cfg)
        print(f"✓ Dataset created successfully!")
        
        # Test train split
        if 'train' in dataset:
            train_dataset = dataset['train']
            print(f"\nTrain dataset info:")
            print(f"  Total samples: {len(train_dataset)}")
            
            # Try loading first sample
            print("\n  Loading first sample...")
            sample = train_dataset[0]
            print(f"  ✓ Sample loaded successfully!")
            print(f"    Keys: {list(sample.keys())}")
            print(f"    Text input: {sample.get('text_input', 'N/A')[:100]}...")
            print(f"    PC feat shape: {sample.get('pc_feat', torch.tensor([])).shape}")
            print(f"    PC shape: {sample.get('pc', torch.tensor([])).shape}")
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("ScanQA Dataset Test: PASSED")
    print("=" * 70)
    return True

def test_sqa3d_loading():
    """Test SQA3D dataset loading"""
    print("\n" + "=" * 70)
    print("Testing SQA3D Dataset Loading")
    print("=" * 70)
    
    # Load config
    config_path = "lavis/projects/blip2/train/finetune_sqa.yaml"
    cfg = OmegaConf.load(config_path)
    
    print(f"\n✓ Config loaded from: {config_path}")
    
    # Check annotation files
    annotations = cfg.datasets['3d_vqa']['build_info']['annotations']
    for split, info in annotations.items():
        ann_path = info['storage']
        exists = os.path.exists(ann_path)
        print(f"  {split:6s}: {ann_path}")
        print(f"           {'✓ EXISTS' if exists else '✗ MISSING'}")
        if exists:
            with open(ann_path, 'r') as f:
                data = json.load(f)
                print(f"           {len(data)} samples")
    
    # Test dataset creation
    print("\nAttempting to create dataset...")
    try:
        dataset = load_dataset("3d_vqa", cfg=cfg)
        print(f"✓ Dataset created successfully!")
        
        # Test train split
        if 'train' in dataset:
            train_dataset = dataset['train']
            print(f"\nTrain dataset info:")
            print(f"  Total samples: {len(train_dataset)}")
            
            # Try loading first sample
            print("\n  Loading first sample...")
            sample = train_dataset[0]
            print(f"  ✓ Sample loaded successfully!")
            print(f"    Keys: {list(sample.keys())}")
            print(f"    Text input: {sample.get('text_input', 'N/A')[:100]}...")
            print(f"    PC feat shape: {sample.get('pc_feat', torch.tensor([])).shape}")
            print(f"    PC shape: {sample.get('pc', torch.tensor([])).shape}")
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("SQA3D Dataset Test: PASSED")
    print("=" * 70)
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("3D-LLM Data Loading Test Suite")
    print("=" * 70)
    
    results = []
    
    # Test ScanQA
    try:
        results.append(("ScanQA", test_scanqa_loading()))
    except Exception as e:
        print(f"\n✗ ScanQA test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ScanQA", False))
    
    # Test SQA3D
    try:
        results.append(("SQA3D", test_sqa3d_loading()))
    except Exception as e:
        print(f"\n✗ SQA3D test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SQA3D", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:10s}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed! Ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
