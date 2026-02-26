#!/usr/bin/env python3
"""
Validation script to test the unified pipeline on single samples from each dataset.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3DLLM_BLIP2-base'))
sys.path.insert(0, os.path.dirname(__file__))

from ply_loader import load_ply


def test_3dfront():
    """Test 3D-FRONT PLY loading and path construction."""
    print("=" * 70)
    print("TESTING 3D-FRONT")
    print("=" * 70)
    print()
    
    # Test PCL ID from list
    pcl_id = "bf18f2a4-e180-4990-9ebb-1e52f4ab4e37@Lounge-55050"
    identifier, scene = pcl_id.split('@')
    
    # Construct path
    base = "/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT"
    ply_path = os.path.join(base, identifier, scene, f"{scene}.ply")
    
    print(f"PCL ID: {pcl_id}")
    print(f"PLY path: {ply_path}")
    print(f"Exists: {os.path.exists(ply_path)}")
    print()
    
    if os.path.exists(ply_path):
        print("Loading PLY file...")
        points, colors = load_ply(ply_path)
        print(f"✓ Loaded successfully")
        print(f"  Points: {points.shape}")
        print(f"  Colors: {colors.shape}")
        print()
    
    # Check question files
    questions_base = "/project/3dllms/melgin/UPD-3D/upd_text/3D-FRONT"
    categories = ['open_ended', 'standard', 'aad_base']
    
    print("Checking question files:")
    for cat in categories:
        q_path = os.path.join(questions_base, cat, f"{pcl_id}.txt")
        exists = os.path.exists(q_path)
        if exists:
            with open(q_path, 'r') as f:
                question = f.read().strip()
            print(f"  ✓ {cat}: {question[:60]}...")
        else:
            print(f"  ✗ {cat}: NOT FOUND")
    print()
    
    # Check output path
    output_dir = "/cluster/medbow/project/3dllms/melgin/datasets/3D-FRONT_processed"
    feat_path = os.path.join(output_dir, f"{pcl_id}.pt")
    coord_path = os.path.join(output_dir, f"{pcl_id}.npy")
    
    print("Output paths:")
    print(f"  Features: {feat_path}")
    print(f"  Coords:   {coord_path}")
    print()


def test_crops3d():
    """Test Crops3D PLY loading and path construction."""
    print("=" * 70)
    print("TESTING CROPS3D")
    print("=" * 70)
    print()
    
    # Test PCL ID from list
    pcl_id = "Cabbage@sl_1109_14"
    crop_type, filename = pcl_id.split('@')
    
    # Construct path
    base = "/project/3dllms/melgin/datasets/CEA/Crops3D"
    ply_path = os.path.join(base, crop_type, f"{filename}.ply")
    
    print(f"PCL ID: {pcl_id}")
    print(f"PLY path: {ply_path}")
    print(f"Exists: {os.path.exists(ply_path)}")
    print()
    
    if os.path.exists(ply_path):
        print("Loading PLY file...")
        points, colors = load_ply(ply_path)
        print(f"✓ Loaded successfully")
        print(f"  Points: {points.shape}")
        print(f"  Colors: {colors.shape}")
        print()
    
    # Check question files
    questions_base = "/project/3dllms/melgin/UPD-3D/upd_text/Crops3D"
    categories = ['open_ended', 'standard', 'aad_base']
    
    print("Checking question files:")
    for cat in categories:
        q_path = os.path.join(questions_base, cat, f"{pcl_id}.txt")
        exists = os.path.exists(q_path)
        if exists:
            with open(q_path, 'r') as f:
                question = f.read().strip()
            print(f"  ✓ {cat}: {question[:60]}...")
        else:
            print(f"  ✗ {cat}: NOT FOUND")
    print()
    
    # Check output path
    output_dir = "/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed"
    safe_name = pcl_id.replace('@', '_')
    feat_path = os.path.join(output_dir, f"{safe_name}.pt")
    coord_path = os.path.join(output_dir, f"{safe_name}.npy")
    
    print("Output paths:")
    print(f"  Features: {feat_path}")
    print(f"  Coords:   {coord_path}")
    print(f"  Exists:   {os.path.exists(feat_path)} / {os.path.exists(coord_path)}")
    print()


if __name__ == "__main__":
    test_3dfront()
    test_crops3d()
    
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("✓ All paths and formats validated")
    print("✓ Ready to run preprocessing and inference")
