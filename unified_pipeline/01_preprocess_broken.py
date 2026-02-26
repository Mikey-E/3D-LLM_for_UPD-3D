#!/usr/bin/env python3
"""
Unified preprocessing script for both Crops3D and 3D-FRONT datasets.
Converts PLY files to 3D-LLM format (features + coordinates).
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3DLLM_BLIP2-base'))

from ply_loader import load_ply


# Dataset configurations
DATASET_CONFIGS = {
    'Crops3D': {
        'pcl_base': '/project/3dllms/melgin/datasets/CEA/Crops3D',
        'output_base': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_test.txt',
        'path_format': 'croptype',  # CropType/filename.ply
    },
    '3D-FRONT': {
        'pcl_base': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT',
        'output_base': '/cluster/medbow/project/3dllms/melgin/datasets/3D-FRONT_processed',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_test.txt',
        'path_format': 'nested',  # identifier/scene/scene.ply
    }
}


def get_ply_path(pcl_id, dataset_name):
    """
    Construct PLY file path based on dataset structure.
    
    Args:
        pcl_id: Point cloud identifier
        dataset_name: 'Crops3D' or '3D-FRONT'
        
    Returns:
        Full path to PLY file
    """
    config = DATASET_CONFIGS[dataset_name]
    base = config['pcl_base']
    
    if config['path_format'] == 'croptype':
        # Crops3D: CropType@filename -> CropType/filename.ply
        crop_type, filename = pcl_id.split('@')
        return os.path.join(base, crop_type, f"{filename}.ply")
    
    elif config['path_format'] == 'nested':
        # 3D-FRONT: identifier@scene -> identifier/scene/scene.ply
        identifier, scene = pcl_id.split('@')
        return os.path.join(base, identifier, scene, f"{scene}.ply")
    
    else:
        raise ValueError(f"Unknown path format: {config['path_format']}")


def get_output_path(pcl_id, dataset_name, extension):
    """
    Construct output file path.
    
    Args:
        pcl_id: Point cloud identifier (e.g., "Cabbage@sl_1109_14" or "abc123@Room-456")
        dataset_name: 'Crops3D' or '3D-FRONT'
        extension: '.pt' or '.npy'
        
    Returns:
        Full path to output file
    """
    config = DATASET_CONFIGS[dataset_name]
    output_dir = config['output_base']
    
    # For Crops3D, use CropType_filename format to avoid collisions
    # For 3D-FRONT, use identifier@scene format directly
    if dataset_name == 'Crops3D':
        # CropType@filename -> CropType_filename
        safe_name = pcl_id.replace('@', '_')
    else:
        # identifier@scene -> identifier@scene (keep @ separator)
        safe_name = pcl_id
    
    return os.path.join(output_dir, f"{safe_name}{extension}")


def preprocess_point_cloud(pcl_id, dataset_name, model, vis_processor, device):
    """
    Preprocess a single point cloud file.
    
    Args:
        pcl_id: Point cloud identifier
        dataset_name: 'Crops3D' or '3D-FRONT'
        model: 3D-LLM model for feature extraction
        vis_processor: Processor for point cloud
        device: Device to run on
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get PLY path
        ply_path = get_ply_path(pcl_id, dataset_name)
        
        if not os.path.exists(ply_path):
            print(f"  ✗ PLY file not found: {ply_path}")
            return False
        
        # Load PLY file
        points, colors = load_ply(ply_path)
        
        # Prepare inputs
        pc_data = {'pc': points, 'color': colors}
        pc_processed = vis_processor(pc_data)
        
        pc_feat = pc_processed['pc_feat'].to(device)  # [N, 1408]
        pc_coord = pc_processed['pc'].to(device)  # [N, 3]
        
        # Extract features using model
        with torch.no_grad():
            pc_embeds = model.encode_pc(pc_feat.unsqueeze(0), pc_coord.unsqueeze(0))
            pc_feat_extracted = pc_embeds.squeeze(0).cpu()  # [8192, 1408]
        
        # Save features and coordinates
        feat_path = get_output_path(pcl_id, dataset_name, '.pt')
        coord_path = get_output_path(pcl_id, dataset_name, '.npy')
        
        torch.save(pc_feat_extracted, feat_path)
        np.save(coord_path, pc_coord.cpu().numpy())
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pcl_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess point clouds for 3D-LLM")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['Crops3D', '3D-FRONT'],
        help="Dataset to preprocess"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index in point cloud list"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index in point cloud list (exclusive)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()
    
    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    
    print("=" * 70)
    print(f"PREPROCESSING {args.dataset.upper()} POINT CLOUDS")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Point cloud base: {config['pcl_base']}")
    print(f"Output directory: {config['output_base']}")
    print(f"Point cloud list: {config['pcl_list']}")
    print()
    
    # Create output directory
    os.makedirs(config['output_base'], exist_ok=True)
    
    # Load point cloud list
    print("Loading point cloud list...")
    with open(config['pcl_list'], 'r') as f:
        all_pcls = [line.strip() for line in f if line.strip()]
    
    # Select subset
    if args.end_idx is None:
        args.end_idx = len(all_pcls)
    
    pcl_list = all_pcls[args.start_idx:args.end_idx]
    
    print(f"  Total point clouds: {len(all_pcls)}")
    print(f"  Processing range: {args.start_idx} to {args.end_idx}")
    print(f"  Processing count: {len(pcl_list)}")
    print()
    
    # Load model
    print("Loading 3D-LLM model...")
    from lavis.models import load_model_and_preprocess
    
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=args.device,
    )
    vis_processor = vis_processors["eval"]
    
    print("  ✓ Model loaded")
    print()
    
    # Process each point cloud
    print("=" * 70)
    print("PROCESSING POINT CLOUDS")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    
    for pcl_id in tqdm(pcl_list, desc="Processing"):
        if preprocess_point_cloud(pcl_id, args.dataset, model, vis_processor, args.device):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(pcl_list)}")
    print(f"Failed:     {fail_count}/{len(pcl_list)}")
    print(f"Output dir: {config['output_base']}")


if __name__ == "__main__":
    main()
