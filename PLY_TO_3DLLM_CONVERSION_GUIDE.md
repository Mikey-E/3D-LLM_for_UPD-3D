# Converting .ply Point Clouds to 3D-LLM Format

## Overview

To test 3D-LLM on your `.ply` files, you need to convert them into the format the model expects:
- **Features**: `.pt` file with shape `[N, 1408]` - learned semantic features
- **Coordinates**: `.npy` file with shape `[N, 3]` - voxelized integer coordinates

## Three Approaches (Easiest → Most Accurate)

---

## 🚀 **Option 1: Simple Approximation (Fastest - Minutes)**

Use a pre-trained point cloud encoder to generate features directly from `.ply` coordinates.

### Pros:
- ✅ Fast (minutes per file)
- ✅ No rendering required
- ✅ Works with any .ply file immediately
- ✅ Good for quick prototyping/testing

### Cons:
- ⚠️ Features may not be as rich as multi-view BLIP2/CLIP
- ⚠️ Model trained on multi-view features, so performance may be lower
- ⚠️ Domain gap between training and test data

### Steps:

#### 1. Load your .ply file
```python
import open3d as o3d
import numpy as np
import torch

# Load point cloud
pcd = o3d.io.read_point_cloud("your_file.ply")
points = np.asarray(pcd.points)  # Shape: [N, 3]
colors = np.asarray(pcd.colors) if pcd.has_colors() else None  # Shape: [N, 3]

print(f"Loaded {points.shape[0]} points")
```

#### 2. Option 1A: Use ULIP PointBERT (Already in Codebase!)
```python
# The codebase has ULIP PointBERT - extract features directly from xyz
from lavis.models.ulip_models.ULIP_models import ULIP_PointBERT

# Initialize encoder (this is what 3D-LLM uses internally for point encoding)
point_encoder = ULIP_PointBERT(ulip_v=2)  # or ulip_v="shapenet", "objaverse"
point_encoder.eval()
point_encoder.cuda()

# Sample to 8192 points (PointBERT input size)
if points.shape[0] > 8192:
    indices = np.random.choice(points.shape[0], 8192, replace=False)
    points_sampled = points[indices]
else:
    # Pad if too few points
    points_sampled = np.pad(points, ((0, 8192 - points.shape[0]), (0, 0)))

# Normalize point cloud to unit sphere
centroid = points_sampled.mean(axis=0)
points_normalized = points_sampled - centroid
max_dist = np.sqrt((points_normalized ** 2).sum(axis=1)).max()
points_normalized = points_normalized / max_dist

# Get features
pc_tensor = torch.from_numpy(points_normalized).float().unsqueeze(0).cuda()  # [1, 8192, 3]
with torch.no_grad():
    pc_features = point_encoder(pc_tensor)  # [1, 512] global feature

# But we need per-point features [N, 1408], not global [1, 512]
# So we need to use a different approach...
```

#### 2. Option 1B: Create Dummy Features (For Quick Testing Only)
```python
# Generate random features as placeholder
# This won't work well but tests the pipeline
N = min(points.shape[0], 5000)  # 3D-LLM expects ~5000 points for scenes
dummy_features = torch.randn(N, 1408)  # Random features

# Normalize and voxelize coordinates
centroid = points[:N].mean(axis=0)
points_normalized = points[:N] - centroid
max_extent = np.abs(points_normalized).max()
points_normalized = points_normalized / max_extent  # Range: [-1, 1]

# Convert to voxel coordinates [0, 255]
voxel_coords = (points_normalized * 128 + 128).astype(np.int32)
voxel_coords = np.clip(voxel_coords, 0, 255)

# Save
torch.save(dummy_features, "scene_id.pt")
np.save("scene_id.npy", voxel_coords)
```

#### 2. Option 1C: Use Pre-trained Vision Model on Single View (Better)
```python
from PIL import Image
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt

# Render a quick view using Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=640, height=480)
vis.add_geometry(pcd)
vis.update_renderer()
img = vis.capture_screen_float_buffer(do_render=True)
img_np = (np.asarray(img) * 255).astype(np.uint8)
vis.destroy_window()

# Extract BLIP2 features from single view
device = torch.device("cuda")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_feature_extractor", 
    model_type="pretrain",
    is_eval=True, 
    device=device
)

image = Image.fromarray(img_np)
image = vis_processors["eval"](image).unsqueeze(0).to(device)

with torch.no_grad():
    sample = {"image": image}
    features = model.extract_features(sample)
    # features.image_embeds_proj: [1, 32, 256] - query features
    
# Broadcast features to all points (crude approximation)
N = 5000
point_features = features.image_embeds_proj[0].mean(0).repeat(N, 1)  # [N, 256]
# Pad to 1408 dimensions
point_features = torch.cat([
    point_features, 
    torch.zeros(N, 1408 - point_features.shape[1], device=device)
], dim=1)

# Save
torch.save(point_features.cpu(), "scene_id.pt")
```

---

## ⚙️ **Option 2: Simplified Multi-View Pipeline (Moderate - Hours)**

Render multiple views and extract features, but simplified version.

### Pros:
- ✅ Better features than Option 1
- ✅ Closer to training distribution
- ✅ Can customize number of views

### Cons:
- ⚠️ Requires Blender for rendering (~1-2 hours setup)
- ⚠️ Still requires feature extraction models
- ⚠️ Takes ~10-30 minutes per object

### Steps:

#### 1. Render Multi-View Images

**Using Blender (recommended):**
```bash
# Use the existing rendering script
cd 3DLanguage_data/ChatCaptioner_based/objaverse_render

# Convert .ply to .glb first (if needed)
# Then render with Blender
blender -b -P render.py --python-use-system-env -- \
    --object_path /path/to/your.ply \
    --output_dir ./output/your_object_id \
    --num_views 50
```

**Or using Python + Open3D (simpler):**
```python
import open3d as o3d
import numpy as np
from PIL import Image

pcd = o3d.io.read_point_cloud("your_file.ply")

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=320, height=240)
vis.add_geometry(pcd)

# Render from multiple viewpoints
num_views = 50
output_dir = "rendered_views"
os.makedirs(output_dir, exist_ok=True)

for i in range(num_views):
    # Set camera position (orbit around object)
    angle = (2 * np.pi * i) / num_views
    radius = 2.0
    
    # Update camera
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([
        np.cos(angle) * np.cos(np.pi/4),
        np.sin(angle) * np.cos(np.pi/4),
        np.sin(np.pi/4)
    ])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.5)
    
    # Capture
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    img_np = (np.asarray(img) * 255).astype(np.uint8)
    
    # Save
    Image.fromarray(img_np).save(f"{output_dir}/view_{i:04d}.png")
    
    # Also save depth if available
    depth = vis.capture_depth_float_buffer(do_render=True)
    np.save(f"{output_dir}/depth_{i:04d}.npy", np.asarray(depth))

vis.destroy_window()
```

#### 2. Extract BLIP2 Features from Views
```python
# Use existing script (modify for your data)
cd 3DLanguage_data/ChatCaptioner_based/gen_features

# Run BLIP feature extraction
python blip_oa.py --scene_path ./rendered_views --output_path ./features

# This extracts 1408-dim BLIP2 features per view
```

#### 3. Project Features to 3D
```python
# Adapt gen_scene_feat_blip.py to your data
# Key steps:
# 1. Load 2D features from each view
# 2. Back-project using depth maps and camera parameters
# 3. Voxelize and average features per voxel
# 4. Sample N points

# Simplified version:
features_3d = []
points_3d = []

for i in range(num_views):
    # Load
    feat_2d = torch.load(f"features/view_{i:04d}.pt")  # [H, W, 1408]
    depth = np.load(f"rendered_views/depth_{i:04d}.npy")  # [H, W]
    
    # Back-project to 3D (need camera intrinsics)
    # ... projection code ...
    
    features_3d.append(feat_3d)
    points_3d.append(pts_3d)

# Aggregate
all_features = torch.cat(features_3d, dim=0)  # [M, 1408]
all_points = np.concatenate(points_3d, axis=0)  # [M, 3]

# Voxelize (aggregate features per voxel)
points_voxelized = (all_points * 128 + 128).astype(int)
voxel_dict = defaultdict(list)
for pt, feat in zip(points_voxelized, all_features):
    voxel_dict[tuple(pt)].append(feat)

# Average features per voxel
final_points = []
final_features = []
for voxel, feat_list in voxel_dict.items():
    final_points.append(voxel)
    final_features.append(torch.stack(feat_list).mean(0))

final_points = np.array(final_points)  # [N, 3]
final_features = torch.stack(final_features)  # [N, 1408]

# Sample to 5000 points
if len(final_points) > 5000:
    indices = np.random.choice(len(final_points), 5000, replace=False)
    final_points = final_points[indices]
    final_features = final_features[indices]

# Save
torch.save(final_features, "your_scene.pt")
np.save("your_scene.npy", final_points)
```

---

## 🔬 **Option 3: Full Pipeline (Most Accurate - Best Results)**

Use the complete 3-step pipeline as designed for the paper.

### Pros:
- ✅ Best performance - matches training data distribution
- ✅ Highest quality features
- ✅ Paper-validated approach

### Cons:
- ⚠️ Complex setup (Blender, SAM, BLIP2, CLIP)
- ⚠️ Time-consuming (~hours per object)
- ⚠️ Requires significant compute (GPU for feature extraction)

### Steps:

Follow the full pipeline in the repository:

#### Step 1: Render Multi-View Images
```bash
cd 3DLanguage_data/ChatCaptioner_based/objaverse_render

# Convert .ply to format Blender can load (may need to convert to .glb)
# Then render 50+ views with depth and camera poses
blender -b -P render.py -- --uid your_object_id
```

#### Step 2: Extract SAM Masks + BLIP2/CLIP Features
```bash
cd ../gen_features

# Extract segmentation masks
python sam_mask.py --all_jobs 1

# Extract BLIP2 features (1408-dim)
python blip_oa.py --all_jobs 1

# Alternative: CLIP features (1024-dim)
# python clip_oa.py --all_jobs 1
```

#### Step 3: Construct 3D Features
```bash
# Project 2D features to 3D and voxelize
python gen_scene_feat_blip.py --all_jobs 1

# Output: {scene_id}.pt (features) and {scene_id}.npy (voxels)
```

**For detailed instructions, see:**
- `3DLanguage_data/ChatCaptioner_based/objaverse_render/README.md`
- `3DLanguage_data/ChatCaptioner_based/gen_features/README.md`

---

## 📝 **Testing Your Converted Files**

Once you have `.pt` and `.npy` files, test with inference:

```python
cd 3DLLM_BLIP2-base

# Modify inference.py to point to your files:
# feature_path = "path/to/your_scene.pt"
# points_path = "path/to/your_scene.npy"

python inference.py
```

**Example inference script for your data:**
```python
import torch
import numpy as np
from lavis.common.registry import registry
from omegaconf import OmegaConf

# Load model
DEVICE = "cuda"
ckpt_path = "../checkpoints/pretrain_blip2_sam_flant5xl_v2.pth"

model_cfg = OmegaConf.create({
    "arch": "blip2_t5",
    "model_type": "pretrain_flant5xl",
    "use_grad_checkpoint": False,
})
model = registry.get_model_class(model_cfg.arch).from_pretrained(
    model_type=model_cfg.model_type
)
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
model.to(DEVICE)

# Load your converted data
pc_feature = torch.load("your_scene.pt")  # [N, 1408]
pc_points = np.load("your_scene.npy")  # [N, 3]

pc_feature = pc_feature.to(DEVICE).unsqueeze(0)  # [1, N, 1408]
pc_points = torch.from_numpy(pc_points).long().to(DEVICE).unsqueeze(0)  # [1, N, 3]

# Process text processor
processor_cfg = OmegaConf.create({"name": "blip_question", "prompt": ""})
text_processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)

# Query
prompt = "What is in this 3D scene?"
prompt = text_processor(prompt)

# Inference
model_inputs = {
    "text_input": prompt,
    "pc_feat": pc_feature,
    "pc": pc_points
}

with torch.no_grad():
    output = model.predict_answers(
        samples=model_inputs,
        max_len=50,
        length_penalty=1.2,
        repetition_penalty=1.5,
    )

print(f"Question: {prompt}")
print(f"Answer: {output[0]}")
```

---

## 🎯 **Recommendation for Your Use Case**

Based on your needs, I recommend:

1. **For quick prototyping/testing**: Start with **Option 1C** (single view + BLIP2)
   - Fast to implement
   - Tests your text queries
   - Validates the pipeline

2. **For better results**: Use **Option 2** (simplified multi-view)
   - Better features
   - Reasonable compute time
   - Good balance

3. **For publication-quality results**: Use **Option 3** (full pipeline)
   - Best performance
   - Matches training data
   - Required for fair comparison

---

## 📊 **Expected File Formats**

Your final files should be:

```python
# Features: PyTorch tensor
features = torch.load("scene_id.pt")
print(features.shape)  # Should be [N, 1408] where N ≈ 5000 for scenes

# Coordinates: NumPy array
coords = np.load("scene_id.npy")
print(coords.shape)  # Should be [N, 3] with integers in range [0, 255]
print(coords.dtype)  # Should be int32 or int64
```

---

## 🔧 **Helper Script Template**

I can create a conversion script for you if you specify which option you want to pursue. The script would:

1. Load your .ply file
2. Apply the chosen conversion method
3. Save .pt and .npy files
4. Validate the format
5. Test with inference

Would you like me to create that script?

---

## ⚠️ **Important Notes**

1. **Point cloud normalization**: The model expects points in range [-1, 1] before voxelization
2. **Feature dimension**: Must be exactly 1408 (BLIP2 output size) or pad to 1408
3. **Number of points**: ~5000 for scenes, ~50000 for objects (model handles variable)
4. **Voxel coordinates**: Integer values in [0, 255] range

## Questions?

Let me know which option you'd like to pursue, and I can provide more detailed code for that specific approach!
