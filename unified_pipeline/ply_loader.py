#!/usr/bin/env python3
"""
Unified PLY loader that handles both Crops3D and 3D-FRONT formats.

Crops3D format:
  - float x, y, z (4 bytes each)
  - ushort or uchar red, green, blue
  - No normals

3D-FRONT format:
  - double x, y, z (8 bytes each)
  - double nx, ny, nz (8 bytes each)
  - uchar red, green, blue (1 byte each)
"""

import struct
import numpy as np


def load_ply(ply_path):
    """
    Load a PLY file with automatic format detection.
    Supports both Crops3D and 3D-FRONT formats.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors (0-255)
    """
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header
        vertex_count = 0
        properties = []
        
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_type, prop_name))
        
        # Determine format based on properties
        has_normals = any(name in ['nx', 'ny', 'nz'] for _, name in properties)
        
        # Find coordinate and color types
        coord_type = None
        color_type = None
        
        for prop_type, prop_name in properties:
            if prop_name in ['x', 'y', 'z']:
                coord_type = prop_type
            elif prop_name in ['red', 'green', 'blue']:
                color_type = prop_type
        
        # Build struct format
        if coord_type == 'double' and has_normals and color_type == 'uchar':
            # 3D-FRONT format
            vertex_format = '<3d3d3B'  # 3 doubles (xyz), 3 doubles (normals), 3 uchars (RGB)
            vertex_size = struct.calcsize(vertex_format)
            print(f"  Format: 3D-FRONT (double xyz, double normals, uchar RGB)")
        elif coord_type == 'float':
            # Crops3D format - detect color type
            if color_type in ['ushort', 'short']:
                vertex_format = '<3f3Hf'  # float xyz, ushort RGB, float (padding/other)
                vertex_size = 22
                print(f"  Format: Crops3D (float xyz, ushort RGB)")
            elif color_type == 'uchar':
                vertex_format = '<3f3Bf'  # float xyz, uchar RGB, float
                vertex_size = 19
                print(f"  Format: Crops3D (float xyz, uchar RGB)")
            else:
                raise ValueError(f"Unknown Crops3D color format: {color_type}")
        else:
            raise ValueError(f"Unknown PLY format: coord={coord_type}, has_normals={has_normals}, color={color_type}")
        
        print(f"  Vertices: {vertex_count:,}")
        print(f"  Vertex size: {vertex_size} bytes")
        
        # Read binary data
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = np.zeros((vertex_count, 3), dtype=np.uint8)
        
        for i in range(vertex_count):
            vertex_data = f.read(vertex_size)
            
            if coord_type == 'double' and has_normals:
                # 3D-FRONT: unpack doubles + normals + uchars
                unpacked = struct.unpack(vertex_format, vertex_data)
                x, y, z = unpacked[0:3]
                # Skip normals (indices 3-5)
                r, g, b = unpacked[6:9]
            elif coord_type == 'float':
                # Crops3D: unpack floats + colors
                unpacked = struct.unpack(vertex_format, vertex_data)
                x, y, z = unpacked[0:3]
                r, g, b = unpacked[3:6]
                
                # Convert ushort colors to uchar if needed
                if color_type == 'ushort':
                    r = int(r / 256)
                    g = int(g / 256)
                    b = int(b / 256)
            
            points[i] = [x, y, z]
            colors[i] = [r, g, b]
        
        return points, colors


def test_loader():
    """Test the loader on both dataset formats."""
    print("=" * 70)
    print("TESTING UNIFIED PLY LOADER")
    print("=" * 70)
    print()
    
    # Test 3D-FRONT
    print("Testing 3D-FRONT format:")
    print("-" * 70)
    front_path = "/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT/00004f89-9aa5-43c2-ae3c-129586be8aaa/Library-4425/Library-4425.ply"
    try:
        points, colors = load_ply(front_path)
        print(f"  ✓ Loaded successfully")
        print(f"  Points shape: {points.shape}")
        print(f"  Colors shape: {colors.shape}")
        print(f"  Point range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"  Color range: [{colors.min()}, {colors.max()}]")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()
    
    # Test Crops3D
    print("Testing Crops3D format:")
    print("-" * 70)
    crops_path = "/project/3dllms/melgin/datasets/CEA/Crops3D/Cabbage/sl_1109_14.ply"
    try:
        points, colors = load_ply(crops_path)
        print(f"  ✓ Loaded successfully")
        print(f"  Points shape: {points.shape}")
        print(f"  Colors shape: {colors.shape}")
        print(f"  Point range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"  Color range: [{colors.min()}, {colors.max()}]")
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    test_loader()
