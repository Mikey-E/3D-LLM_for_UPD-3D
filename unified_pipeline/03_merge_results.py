#!/usr/bin/env python3
"""
Merge 3D-FRONT inference results from 30 array jobs into 12 final category files.
Each job created 12 category files, so we need to merge 30 files per category.
"""

import os
import json
import glob
from pathlib import Path

# Categories to merge
CATEGORIES = [
    'standard',
    'open_ended',
    'open_ended_additional_instruction',
    'aad_base',
    'aad_additional_option',
    'aad_additional_instruction',
    'iasd_base',
    'iasd_additional_option',
    'iasd_additional_instruction',
    'ivqd_base',
    'ivqd_additional_option',
    'ivqd_additional_instruction',
]

def merge_category(category, input_dir, output_dir):
    """Merge all files for a given category."""
    print(f"\nProcessing category: {category}")
    print("-" * 70)
    
    # Find all files for this category
    pattern = f"inf_rslts_3dllm_3D-FRONT_test_{category}_*.json"
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    print(f"  Found {len(files)} files to merge")
    
    if not files:
        print(f"  ✗ No files found for category: {category}")
        return False
    
    # Merge all data
    merged_data = {}
    total_entries = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Add entries to merged data
            for key, value in data.items():
                if key in merged_data:
                    print(f"  ⚠ Warning: Duplicate key found: {key}")
                merged_data[key] = value
                total_entries += 1
                
        except Exception as e:
            print(f"  ✗ Error reading {file_path}: {e}")
            continue
    
    print(f"  Total entries merged: {total_entries}")
    
    # Save merged data
    output_file = os.path.join(output_dir, f"inf_rslts_3dllm_3D-FRONT_test_{category}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"  ✓ Saved to: {output_file}")
        print(f"  ✓ Final size: {len(merged_data)} unique entries")
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving merged file: {e}")
        return False


def main():
    # Paths
    input_dir = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/results/3dfront_inference"
    output_dir = "/project/3dllms/melgin/3D-LLM_for_UPD-3D/results/3dfront_results_by_category"
    
    print("=" * 70)
    print("MERGING 3D-FRONT INFERENCE RESULTS")
    print("=" * 70)
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Categories: {len(CATEGORIES)}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each category
    successful = 0
    failed = 0
    
    for category in CATEGORIES:
        if merge_category(category, input_dir, output_dir):
            successful += 1
        else:
            failed += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {successful}/{len(CATEGORIES)}")
    print(f"Failed:     {failed}/{len(CATEGORIES)}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
