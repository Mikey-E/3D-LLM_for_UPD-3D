#!/usr/bin/env python3
"""
Manually process the 4 missing Crops3D point clouds.
Uses the same preprocessing pipeline as the array job.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '3DLLM_BLIP2-base'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unified_pipeline'))

from ply_loader import load_ply
from lavis.models import load_model_and_preprocess

# Missing files to process
MISSING_FILES = [
    ('Cabbage', 'mvs_1123_06'),
    ('Cabbage', 'sl_901_12'),
    ('Wheat', '129'),
    ('Wheat', '58'),
]

PLY_BASE = '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D'
OUTPUT_DIR = '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed'


def render_point_cloud(points, colors, view_angle='front'):
    """Render point cloud to image from specified viewpoint."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0, 1]
    colors_norm = colors.astype(np.float32) / 255.0
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors_norm, s=1, marker='.')
    
    # Set viewpoint
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
    
    # Convert to PIL and resize
    img = Image.fromarray(img_array)
    img = img.resize((224, 224), Image.BILINEAR)
    
    return img


def extract_blip2_features(img, model, vis_processors):
    """Extract BLIP2 features from rendered image."""
    # Preprocess image
    image_tensor = vis_processors["eval"](img).unsqueeze(0)  # [1, 3, 224, 224]
    
    # Move to GPU
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Extract features
    with torch.no_grad():
        samples = {"image": image_tensor}
        image_embeds = model.ln_vision(model.visual_encoder(samples["image"]))
        
        # Use Q-Former
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        # Project to T5 space
        inputs_t5 = model.t5_proj(query_output.last_hidden_state)  # [1, 32, 2560]
        features = inputs_t5.mean(dim=1).cpu()  # [1, 2560]
        
    return features


def sample_points(points, colors, num_samples=8192):
    """Sample specified number of points."""
    num_points = len(points)
    
    if num_points >= num_samples:
        indices = np.random.choice(num_points, num_samples, replace=False)
    else:
        indices = np.random.choice(num_points, num_samples, replace=True)
    
    return points[indices], colors[indices]


def process_single_file(crop_type, filename, model, vis_processors):
    """Process a single point cloud file."""
    print(f"\n{'='*70}")
    print(f"Processing: {crop_type}@{filename}")
    print('='*70)
    
    # Construct paths
    ply_path = os.path.join(PLY_BASE, crop_type, f"{filename}.ply")
    pcl_id = f"{crop_type}_{filename}"
    feat_path = os.path.join(OUTPUT_DIR, f"{pcl_id}.pt")
    coord_path = os.path.join(OUTPUT_DIR, f"{pcl_id}.npy")
    
    # Check if already exists
    if os.path.exists(feat_path) and os.path.exists(coord_path):
        print(f"✓ Already processed (files exist)")
        return True
    
    try:
        # 1. Load PLY
        print(f"1. Loading PLY file...")
        print(f"   Path: {ply_path}")
        points, colors = load_ply(ply_path)
        print(f"   ✓ Loaded {len(points):,} points")
        print(f"   Points range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"   Colors range: [{colors.min()}, {colors.max()}]")
        
        # 2. Render to image
        print(f"\n2. Rendering to image...")
        img = render_point_cloud(points, colors, view_angle='front')
        print(f"   ✓ Rendered to {img.size} image")
        
        # 3. Extract BLIP2 features
        print(f"\n3. Extracting BLIP2 features...")
        features = extract_blip2_features(img, model, vis_processors)
        print(f"   ✓ Extracted features: {features.shape}")
        
        # 4. Sample and normalize points
        print(f"\n4. Sampling and normalizing points...")
        sampled_points, _ = sample_points(points, colors, num_samples=8192)
        
        # Normalize coordinates
        centroid = sampled_points.mean(axis=0)
        sampled_points_centered = sampled_points - centroid
        max_dist = np.abs(sampled_points_centered).max()
        sampled_points_norm = sampled_points_centered / (max_dist + 1e-8)
        print(f"   ✓ Sampled 8,192 points")
        print(f"   Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"   Max dist: {max_dist:.2f}")
        
        # 5. Expand features
        print(f"\n5. Expanding features...")
        features_expanded = features.repeat(8192, 1)  # [8192, 2560]
        
        # Slice to 1408 dimensions
        if features_expanded.shape[1] > 1408:
            features_expanded = features_expanded[:, :1408]
        elif features_expanded.shape[1] < 1408:
            pad_size = 1408 - features_expanded.shape[1]
            features_expanded = torch.nn.functional.pad(
                features_expanded, (0, pad_size), value=0
            )
        print(f"   ✓ Feature tensor: {features_expanded.shape}")
        
        # 6. Save
        print(f"\n6. Saving files...")
        torch.save(features_expanded, feat_path)
        np.save(coord_path, sampled_points_norm.astype(np.float32))
        print(f"   ✓ Saved: {feat_path}")
        print(f"   ✓ Saved: {coord_path}")
        
        print(f"\n✓ SUCCESS: {crop_type}@{filename}")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("MANUAL PROCESSING OF 4 MISSING CROPS3D FILES")
    print("=" * 70)
    print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script must run on GPU.")
        return
    
    device = "cuda"
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load BLIP2 model
    print("Loading BLIP2 model...")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=device
    )
    model.eval()
    print("✓ Model loaded successfully!")
    print()
    
    # Process each file
    results = []
    for crop_type, filename in MISSING_FILES:
        success = process_single_file(crop_type, filename, model, vis_processors)
        results.append((crop_type, filename, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for _, _, success in results if success)
    print(f"Successfully processed: {successful}/{len(MISSING_FILES)}")
    print()
    
    for crop_type, filename, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {crop_type}@{filename}")
    print()
    
    if successful == len(MISSING_FILES):
        print("✓ All files processed successfully!")
    else:
        print("⚠ Some files failed to process")


if __name__ == "__main__":
    main()
