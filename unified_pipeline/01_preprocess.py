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
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3DLLM_BLIP2-base'))

from ply_loader import load_ply
from lavis.models import load_model_and_preprocess


# Dataset configurations
DATASET_CONFIGS = {
    'Crops3D': {
        'pcl_base': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_test.txt',
        'output_dir': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed',
        'questions_base': '/cluster/medbow/project/3dllms/melgin/UPD-3D/upd_text/Crops3D',
    },
    'Crops3D_train': {
        'pcl_base': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_train.txt',
        'output_dir': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed',
        'questions_base': '/cluster/medbow/project/3dllms/melgin/UPD-3D/upd_text/Crops3D',
    },
    '3D-FRONT': {
        'pcl_base': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT',
        'pcl_list': '/project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_test.txt',
        'output_dir': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed',
        'questions_base': '/project/3dllms/melgin/UPD-3D/upd_text/3D-FRONT',
    },
    '3D-FRONT_train': {
        'pcl_base': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT',
        'pcl_list': '/project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_train_minus_val.txt',
        'output_dir': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed',
        'questions_base': '/project/3dllms/melgin/UPD-3D/upd_text/3D-FRONT',
    },
    '3D-FRONT_val': {
        'pcl_base': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT',
        'pcl_list': '/project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_val_subset_of_train.txt',
        'output_dir': '/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed',
        'questions_base': '/project/3dllms/melgin/UPD-3D/upd_text/3D-FRONT',
    },
    'GIW529': {
        'pcl_base': '/project/3dllms/melgin/datasets/GIW/giw529subcat',
        'pcl_list': '/project/3dllms/melgin/UPD-3D/pcl_lists/GIW529.txt',
        'output_dir': '/project/3dllms/melgin/datasets/GIW/giw529_processed_for_3dllm',
        'questions_base': '/project/3dllms/melgin/UPD-3D/upd_text/GIW529_gpt-5-nano'
    },
}


def get_ply_path(pcl_id, dataset_name):
    """Construct path to PLY file based on dataset and identifier."""
    # Handle dataset variants (e.g., Crops3D_train -> Crops3D, 3D-FRONT_train -> 3D-FRONT)
    if 'Crops3D' in dataset_name:
        base_dataset = 'Crops3D'
    elif '3D-FRONT' in dataset_name:
        base_dataset = '3D-FRONT'
    elif 'GIW529' in dataset_name:
        base_dataset = 'GIW529'
    else:
        base_dataset = dataset_name
    
    config = DATASET_CONFIGS[dataset_name]
    base_dir = config['pcl_base']
    
    if base_dataset == 'Crops3D':
        # Format: CropType@filename -> CropType/filename.ply
        crop_type, filename = pcl_id.split('@')
        return os.path.join(base_dir, crop_type, f"{filename}.ply")
    
    elif base_dataset == '3D-FRONT':
        # Format: identifier@scene -> identifier/scene/scene.ply
        identifier, scene = pcl_id.split('@')
        return os.path.join(base_dir, identifier, scene, f"{scene}.ply")
    
    elif base_dataset == 'GIW529':
        # Format: Category@filename -> Category/filename.ply
        category, filename = pcl_id.split('@')
        return os.path.join(base_dir, category, f"{filename}.ply")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_output_path(pcl_id, dataset_name, extension):
    """Create safe output filename based on dataset."""
    config = DATASET_CONFIGS[dataset_name]
    output_dir = config['output_dir']
    
    # Replace @ with _ for filename safety
    safe_name = pcl_id.replace('@', '_')
    
    return os.path.join(output_dir, f"{safe_name}{extension}")


def render_point_cloud(points, colors, view_angle='front'):
    """
    Render point cloud to image from specified viewpoint.
    
    Args:
        points: [N, 3] point coordinates
        colors: [N, 3] RGB colors (0-1 range)
        view_angle: 'front', 'back', 'left', 'right', 'top', 'bottom'
    
    Returns:
        PIL Image (224x224)
    """
    # Create figure (NaN/Inf filtering done before calling this function)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=1, marker='.')
    
    # Set viewpoint based on angle
    view_angles = {
        'front': (0, 0),
        'back': (0, 180),
        'left': (0, 90),
        'right': (0, 270),
        'top': (90, 0),
        'bottom': (-90, 0)
    }
    elev, azim = view_angles.get(view_angle, (0, 0))
    ax.view_init(elev=elev, azim=azim)
    
    # Remove axes
    ax.set_axis_off()
    ax.set_xlim([points[:, 0].min(), points[:, 0].max()])
    ax.set_ylim([points[:, 1].min(), points[:, 1].max()])
    ax.set_zlim([points[:, 2].min(), points[:, 2].max()])
    
    # Render to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert to PIL and resize to 224x224
    img = Image.fromarray(img_array)
    img = img.resize((224, 224), Image.BILINEAR)
    
    return img


def extract_blip2_features(img, model, vis_processors):
    """Extract BLIP2 features from rendered image."""
    # Preprocess image
    image_tensor = vis_processors["eval"](img).unsqueeze(0)  # [1, 3, 224, 224]
    
    # Move to GPU if available
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Extract features
    with torch.no_grad():
        # Get image embeddings from BLIP2 vision encoder
        samples = {"image": image_tensor}
        image_embeds = model.ln_vision(model.visual_encoder(samples["image"]))
        
        # Use Q-Former to extract query tokens
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        # Project to T5 input space
        inputs_t5 = model.t5_proj(query_output.last_hidden_state)  # [1, 32, 2560]
        
        # Average over queries to get single feature vector
        features = inputs_t5.mean(dim=1).cpu()  # [1, 2560]
        
    return features


def normalize_features(features, target_mean=0.0134, target_std=0.0229):
    """
    Normalize features to match ScanNet feature distribution.
    
    Args:
        features: torch.Tensor of shape [N, 1408]
        target_mean: desired mean (default from ScanNet)
        target_std: desired std (default from ScanNet)
    
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


def sample_points(points, colors, num_samples=8192):
    """Sample specified number of points."""
    num_points = len(points)
    
    if num_points >= num_samples:
        # Random sampling
        indices = np.random.choice(num_points, num_samples, replace=False)
    else:
        # Upsample with replacement
        indices = np.random.choice(num_points, num_samples, replace=True)
    
    return points[indices], colors[indices]


def preprocess_point_cloud(pcl_id, dataset_name, model, vis_processors, num_points=8192):
    """
    Preprocess a single point cloud file.
    
    Args:
        pcl_id: Point cloud identifier
        dataset_name: 'Crops3D' or '3D-FRONT'
        model: BLIP2 model for feature extraction
        vis_processors: BLIP2 image preprocessors
        num_points: Number of points to sample
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get PLY path
        ply_path = get_ply_path(pcl_id, dataset_name)
        
        if not os.path.exists(ply_path):
            print(f"  ✗ PLY file not found: {ply_path}")
            return False
        
        # Load point cloud
        points, colors = load_ply(ply_path)
        
        # Filter out NaN and Inf values BEFORE processing
        valid_mask = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(colors), axis=1)
        if not np.any(valid_mask):
            raise ValueError("All points contain NaN or Inf values")
        if np.sum(valid_mask) < len(points):
            print(f"  Warning: Filtered out {len(points) - np.sum(valid_mask)} points with NaN/Inf values")
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Normalize colors to [0, 1] for rendering
        colors_normalized = colors.astype(np.float32) / 255.0
        
        # Render to image (front view)
        img = render_point_cloud(points, colors_normalized, view_angle='front')
        
        # Extract BLIP2 features from image
        features = extract_blip2_features(img, model, vis_processors)  # [1, 2560]
        
        # Sample points for coordinates
        sampled_points, _ = sample_points(points, colors, num_samples=num_points)
        
        # Check for NaN/Inf in sampled points
        if not np.all(np.isfinite(sampled_points)):
            raise ValueError(f"Sampled points contain NaN or Inf values")
        
        # Normalize coordinates to [-1, 1]
        centroid = sampled_points.mean(axis=0)
        sampled_points_centered = sampled_points - centroid
        max_dist = np.abs(sampled_points_centered).max()
        
        # Handle edge case where all points are at the same location or max_dist is very small
        if max_dist < 1e-6 or not np.isfinite(max_dist):
            print(f"  Warning: max_dist={max_dist}, using zero-centered coordinates")
            sampled_points_norm = np.zeros_like(sampled_points_centered)
        else:
            sampled_points_norm = sampled_points_centered / max_dist
            # Replace any remaining NaN/Inf with zeros (shouldn't happen but safety check)
            if not np.all(np.isfinite(sampled_points_norm)):
                print(f"  Warning: NaN/Inf in normalized coords, replacing with zeros")
                sampled_points_norm = np.where(np.isfinite(sampled_points_norm), sampled_points_norm, 0.0)
        
        # Expand features to match coordinate dimensions
        # Model expects [N, 1408], but BLIP2-T5 gives [1, 2560]
        # Replicate for each point
        features_expanded = features.repeat(num_points, 1)  # [num_points, 2560]
        
        # Slice to 1408 dimensions (model expectation)
        if features_expanded.shape[1] > 1408:
            features_expanded = features_expanded[:, :1408]
        elif features_expanded.shape[1] < 1408:
            # Pad with zeros if needed
            pad_size = 1408 - features_expanded.shape[1]
            features_expanded = torch.nn.functional.pad(
                features_expanded, (0, pad_size), value=0
            )
        
        # Normalize features to match ScanNet distribution
        if '3D-FRONT' in dataset_name or 'Crops3D' in dataset_name or 'GIW529' in dataset_name:
            features_expanded = normalize_features(features_expanded)
        
        # Save features and coordinates
        feat_path = get_output_path(pcl_id, dataset_name, '.pt')
        coord_path = get_output_path(pcl_id, dataset_name, '.npy')
        
        torch.save(features_expanded, feat_path)  # [num_points, 1408]
        np.save(coord_path, sampled_points_norm.astype(np.float32))  # [num_points, 3]
        
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
        choices=['Crops3D', 'Crops3D_train', '3D-FRONT', '3D-FRONT_train', '3D-FRONT_val', 'GIW529'],
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
    print(f"{args.dataset.upper()} Preprocessing - Array Job {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print("=" * 70)
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")
    print(f"Start time: {os.popen('date').read().strip()}")
    print()
    
    # Check Python environment
    print(f"Python: {sys.executable}")
    print(f"Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        free_memory = torch.cuda.mem_get_info()[0] // (1024**2)
        print(f"{gpu_name}, {total_memory} MiB, {free_memory} MiB")
        print()
    
    print(f"Processing point clouds {args.start_idx} to {args.end_idx - 1}")
    print(f"Estimated count: {args.end_idx - args.start_idx}")
    print()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load BLIP2 model
    print("Loading BLIP2 model...")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=args.device
    )
    model.eval()
    print("Model loaded successfully!")
    print()
    
    # Load point cloud list
    with open(config['pcl_list'], 'r') as f:
        pcl_list = [line.strip() for line in f if line.strip()]
    
    # Process subset
    pcl_subset = pcl_list[args.start_idx:args.end_idx]
    
    # Process each point cloud
    successful = 0
    failed = 0
    
    for pcl_id in tqdm(pcl_subset, desc="Processing"):
        # Show PLY info
        ply_path = get_ply_path(pcl_id, args.dataset)
        try:
            points, colors = load_ply(ply_path)
            print(f"\nProcessing: {pcl_id}")
            print(f"  PLY: {ply_path}")
            print(f"  Vertices: {len(points):,}")
            
            if preprocess_point_cloud(pcl_id, args.dataset, model, vis_processors):
                successful += 1
                print(f"  ✓ Success")
            else:
                failed += 1
        except Exception as e:
            print(f"\nProcessing: {pcl_id}")
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {successful}/{len(pcl_subset)}")
    print(f"Failed:     {failed}/{len(pcl_subset)}")
    print(f"Output dir: {config['output_dir']}")
    print()
    print(f"End time: {os.popen('date').read().strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
