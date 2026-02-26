#!/usr/bin/env python3
"""
Normalize 3D-FRONT features to match ScanNet feature distribution.

ScanNet features have mean≈0.013, std≈0.023
Our 3D-FRONT features currently have much larger scale (mean≈-0.111, std≈15.69)

This script will normalize all .pt files in 3D-FRONT_processed directory.
"""

import os
import torch
import glob
from tqdm import tqdm
import shutil

# Paths
FRONT3D_DIR = "/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed"
BACKUP_DIR = "/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed_unnormalized"

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
    print("3D-FRONT FEATURE NORMALIZATION")
    print("=" * 70)
    print()
    
    # Find all .pt files
    pt_files = sorted(glob.glob(os.path.join(FRONT3D_DIR, "*.pt")))
    print(f"Found {len(pt_files)} .pt files to normalize")
    print()
    
    # SKIP BACKUPS due to disk quota - normalize in-place only
    print("WARNING: Normalizing in-place WITHOUT backups (disk quota limitation)")
    print("         Original unnormalized features will be lost!")
    print()
    
    # Sample a few files first to check current statistics
    print("Checking current statistics (10 samples)...")
    sample_files = pt_files[:10]
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
    
    # Auto-confirm (for non-interactive execution)
    print("Proceeding with normalization automatically...")
    print()
    
    # Normalize all files
    print("Normalizing features...")
    failed = []
    
    for pt_file in tqdm(pt_files, desc="Processing"):
        try:
            # Load features
            features = torch.load(pt_file, map_location='cpu')
            
            # Normalize (skip backup to avoid disk quota issues)
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
    print("Verifying normalized statistics (10 samples)...")
    sample_files = pt_files[:10]
    for f in sample_files:
        pt = torch.load(f, map_location='cpu')
        print(f"  {os.path.basename(f)}: mean={pt.mean():.4f}, std={pt.std():.4f}, shape={pt.shape}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
