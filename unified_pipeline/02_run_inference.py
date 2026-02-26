#!/usr/bin/env python3
"""
Unified inference script for both Crops3D and 3D-FRONT datasets.
Runs 3D-LLM inference on preprocessed point clouds.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3DLLM_BLIP2-base'))


# Dataset configurations
DATASET_CONFIGS = {
    'Crops3D': {
        'processed_dir': '/cluster/medbow/project/3dllms/melgin/datasets/CEA/Crops3D_processed',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/Crops3D_test.txt',
        'questions_base': '/cluster/medbow/project/3dllms/melgin/UPD-3D/upd_text/Crops3D_gpt-5-nano',
        'output_dir': '/project/3dllms/melgin/3D-LLM_for_UPD-3D/results/crops3d_inference',
        'file_prefix': 'inf_rslts_3dllm_Crops3D_test',
    },
    '3D-FRONT': {
        'processed_dir': '/cluster/medbow/project/3dllms/melgin/datasets/3d-grand_unzipped/3D-FRONT_processed',
        'pcl_list': '/cluster/medbow/project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_test.txt',
        'questions_base': '/cluster/medbow/project/3dllms/melgin/UPD-3D/upd_text/3D-FRONT',
        'output_dir': '/project/3dllms/melgin/3D-LLM_for_UPD-3D/results/3dfront_inference',
        'file_prefix': 'inf_rslts_3dllm_3D-FRONT_test',
    },
    'GIW529': {
        'processed_dir': '/project/3dllms/melgin/datasets/GIW/giw529_processed_for_3dllm',
        'pcl_list': '/project/3dllms/melgin/UPD-3D/pcl_lists/GIW529_test.txt',
        'questions_base': '/project/3dllms/melgin/UPD-3D/upd_text/GIW529_gpt-5-nano',
        'output_dir': '/project/3dllms/melgin/3D-LLM_for_UPD-3D/results/giw529_inference',
        'file_prefix': 'inf_rslts_3dllm_GIW529_test',
    }
}

QUESTION_CATEGORIES = [
    'aad_base', 'aad_additional_instruction', 'aad_additional_option',
    'iasd_base', 'iasd_additional_instruction', 'iasd_additional_option',
    'ivqd_base', 'ivqd_additional_instruction', 'ivqd_additional_option',
    'open_ended', 'open_ended_additional_instruction', 'standard'
]


def get_processed_paths(pcl_id, dataset_name):
    """Get paths to preprocessed features and coordinates."""
    config = DATASET_CONFIGS[dataset_name]
    processed_dir = config['processed_dir']
    
    # Both datasets use _ instead of @ in filenames for safety
    safe_name = pcl_id.replace('@', '_')
    
    feat_path = os.path.join(processed_dir, f"{safe_name}.pt")
    coord_path = os.path.join(processed_dir, f"{safe_name}.npy")
    
    return feat_path, coord_path


def load_preprocessed_features(pcl_id, dataset_name):
    """Load preprocessed features and coordinates."""
    feat_path, coord_path = get_processed_paths(pcl_id, dataset_name)
    
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"Coordinates not found: {coord_path}")
    
    features = torch.load(feat_path)  # [N, 1408]
    coordinates = torch.from_numpy(np.load(coord_path)).long()  # [N, 3]
    
    return features, coordinates


def load_question(pcl_id, category, dataset_name):
    """Load question text for a specific category."""
    config = DATASET_CONFIGS[dataset_name]
    questions_base = config['questions_base']
    
    question_path = os.path.join(questions_base, category, f"{pcl_id}.txt")
    
    if not os.path.exists(question_path):
        return None
    
    with open(question_path, 'r') as f:
        question = f.read().strip()
    
    return question


def run_inference(model, pc_feat, pc_coord, question, device):
    """Run inference on a single question."""
    # Move to device
    pc_feat = pc_feat.to(device)
    pc_coord = pc_coord.to(device)
    
    # Add batch dimension
    pc_feat = pc_feat.unsqueeze(0)  # [1, N, 1408]
    pc_coord = pc_coord.unsqueeze(0)  # [1, N, 3]
    
    # Run inference
    with torch.no_grad():
        try:
            answers = model.predict_answers(
                samples={
                    "pc_feat": pc_feat,
                    "pc": pc_coord,
                    "text_input": [question]
                },
                max_len=50,
                length_penalty=1.2,
                repetition_penalty=1.5,
            )
            
            return answers[0] if answers else ""
        
        except Exception as e:
            print(f"    Error in inference: {e}")
            return ""


def clean_location_tokens(text):
    """Remove location tokens from model output."""
    import re
    text = text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = re.sub(r'<loc\d+>', '', text)
    text = ' '.join(text.split())
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference on preprocessed point clouds")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['Crops3D', '3D-FRONT', 'GIW529'],
        help="Dataset to run inference on"
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (pretrained or finetuned)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for output directory and file prefix (e.g., '3dllm_base', '3dllm_ft-Crops3D_gpt-5-nano')"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process only specific category (e.g., 'open_ended'). If not specified, processes all categories."
    )
    args = parser.parse_args()
    
    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    
    # Determine categories to process
    if args.category:
        # Validate the category
        if args.category not in QUESTION_CATEGORIES:
            raise ValueError(f"Invalid category: {args.category}. Must be one of: {', '.join(QUESTION_CATEGORIES)}")
        categories_to_process = [args.category]
        print(f"Processing single category: {args.category}")
    else:
        categories_to_process = QUESTION_CATEGORIES
        print(f"Processing all {len(QUESTION_CATEGORIES)} categories")
    
    print("=" * 70)
    print(f"{args.dataset.upper()} INFERENCE")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Processed dir: {config['processed_dir']}")
    print(f"Questions base: {config['questions_base']}")
    print(f"Output dir: {config['output_dir']}")
    print()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
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
    print(f"  Questions per cloud: {len(categories_to_process)}")
    print(f"  Total inferences: {len(pcl_list) * len(categories_to_process)}")
    print()
    
    # Load model
    print("Loading 3D-LLM model...")
    from lavis.common.registry import registry
    from omegaconf import OmegaConf
    
    # Create model architecture
    model_cfg = OmegaConf.create({
        "arch": "blip2_t5",
        "model_type": "pretrain_flant5xl",
        "use_grad_checkpoint": False,
    })
    model = registry.get_model_class(model_cfg.arch).from_pretrained(
        model_type=model_cfg.model_type
    )
    
    # Load checkpoint
    print(f"  Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    
    model.eval()
    model.to(args.device)
    
    print("  ✓ Model loaded successfully")
    print()
    
    # Process each point cloud
    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    
    # Store results by category
    results_by_category = {cat: {} for cat in categories_to_process}
    
    success_count = 0
    fail_count = 0
    
    for pcl_id in tqdm(pcl_list, desc="Processing point clouds"):
        try:
            # Load preprocessed features
            pc_feat, pc_coord = load_preprocessed_features(pcl_id, args.dataset)
            
            # Run inference for each category
            for category in categories_to_process:
                question = load_question(pcl_id, category, args.dataset)
                
                if question is None:
                    continue
                
                # Run inference
                answer = run_inference(model, pc_feat, pc_coord, question, args.device)
                
                # Clean location tokens
                answer = clean_location_tokens(answer)
                
                # Store result
                results_by_category[category][pcl_id] = {
                    'prompt': question,
                    'response': answer,
                    'timestamp': datetime.now().isoformat()
                }
                
                success_count += 1
        
        except Exception as e:
            print(f"  Error processing {pcl_id}: {e}")
            fail_count += 1
    
    # Save results by category
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Create model-specific output directory (no timestamp in folder name)
    run_output_dir = os.path.join(config['output_dir'], args.model_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    for category, results in results_by_category.items():
        if not results:
            continue
        
        # File format: inf_rslts_{model_name}_{dataset}_test_{category}.json
        filename = f"inf_rslts_{args.model_name}_{args.dataset}_test_{category}.json"
        output_path = os.path.join(run_output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  {category}: {len(results)} results -> {filename}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")
    print(f"Output dir: {run_output_dir}")


if __name__ == "__main__":
    main()
