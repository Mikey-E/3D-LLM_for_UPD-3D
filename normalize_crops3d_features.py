#!/usr/bin/env python3
"""
Normalize Crops3D features to match ScanNet feature distribution.

ScanNet features have mean≈0.013, std≈0.023
Our Crops3D features currently have much larger scale (std≈15.24)

This script will normalize all .pt files in Crops3D_processed directory.
"""

import os
import torch
import glob
from tqdm import tqdm
import shutil

# Paths
CROPS3D_DIR = "/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed"
BACKUP_DIR = "/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed_unnormalized"

# Target statistics (from ScanNet)
TARGET_MEAN = 0.0134
TARGET_STD = 0.0229


def normalize_features(features, target_mean=TARGET_MEAN, target_std=TARGET_STD):
    """
    Normalize features to have target mean and std.
    
    Args:
        features: torch.Tensor of shape [N, 1408]
        target_mean: desired mean
        target_std: desired std
    
    Returns:
        Normalized features
    """
    # Standardize to mean=0, std=1
    current_mean = features.mean()
    current_std = features.std()
    
    features_standardized = (features - current_mean) / (current_std + 1e-8)
    
    # Scale to target distribution
    features_normalized = features_standardized * target_std + target_mean
    
    return features_normalized


def main():
    print("=" * 70)
    print("CROPS3D FEATURE NORMALIZATION")
    print("=" * 70)
    print()
    
    # Find all .pt files
    pt_files = sorted(glob.glob(os.path.join(CROPS3D_DIR, "*.pt")))
    print(f"Found {len(pt_files)} .pt files to normalize")
    print()
    
    # Create backup directory
    if not os.path.exists(BACKUP_DIR):
        print(f"Creating backup directory: {BACKUP_DIR}")
        os.makedirs(BACKUP_DIR, exist_ok=True)
        print()
    
    # Sample a few files first to check current statistics
    print("Checking current statistics (5 samples)...")
    sample_files = pt_files[:5]
    current_stats = []
    for f in sample_files:
        pt = torch.load(f, map_location='cpu')
        current_stats.append({
            'mean': pt.mean().item(),
            'std': pt.std().item(),
            'shape': pt.shape
        })
        print(f"  {os.path.basename(f)}: mean={pt.mean():.4f}, std={pt.std():.4f}, shape={pt.shape}")
    
    avg_mean = sum(s['mean'] for s in current_stats) / len(current_stats)
    avg_std = sum(s['std'] for s in current_stats) / len(current_stats)
    print(f"Current average: mean={avg_mean:.4f}, std={avg_std:.4f}")
    print(f"Target: mean={TARGET_MEAN:.4f}, std={TARGET_STD:.4f}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with normalization? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    print()
    
    # Normalize all files
    print("Normalizing features...")
    failed = []
    
    for pt_file in tqdm(pt_files, desc="Processing"):
        try:
            # Load features
            features = torch.load(pt_file, map_location='cpu')
            
            # Backup original (only if not already backed up)
            backup_path = os.path.join(BACKUP_DIR, os.path.basename(pt_file))
            if not os.path.exists(backup_path):
                shutil.copy2(pt_file, backup_path)
            
            # Normalize
            features_normalized = normalize_features(features)
            
            # Save normalized features
            torch.save(features_normalized, pt_file)
            
        except Exception as e:
            failed.append((pt_file, str(e)))
            print(f"\nERROR processing {os.path.basename(pt_file)}: {e}")
    
    print()
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)
    print(f"Processed: {len(pt_files) - len(failed)} files")
    print(f"Failed: {len(failed)} files")
    if failed:
        print("\nFailed files:")
        for f, err in failed:
            print(f"  {os.path.basename(f)}: {err}")
    print()
    
    # Verify a few normalized files
    print("Verifying normalized statistics (5 samples)...")
    sample_files = pt_files[:5]
    for f in sample_files:
        pt = torch.load(f, map_location='cpu')
        print(f"  {os.path.basename(f)}: mean={pt.mean():.4f}, std={pt.std():.4f}, shape={pt.shape}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
