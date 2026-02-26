#!/bin/bash
# Quick Start Training Script for 3D-LLM Finetuning
# Run this to start finetuning on ScanQA or SQA3D

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}3D-LLM Finetuning Quick Start${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found${NC}"
    exit 1
fi

# Check if lavis environment exists
if ! conda env list | grep -q "lavis"; then
    echo -e "${RED}Error: lavis conda environment not found${NC}"
    echo "Please run: conda create -n lavis python=3.8"
    exit 1
fi

# Navigate to correct directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/3DLLM_BLIP2-base"

echo -e "${GREEN}✓ Working directory: $(pwd)${NC}"
echo ""

# Show menu
echo "Select dataset to finetune:"
echo "  1) ScanQA  (24,969 samples)"
echo "  2) SQA3D   (26,182 samples)"
echo "  3) Test dataset loading"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "${BLUE}Starting ScanQA finetuning...${NC}"
        CONFIG="lavis/projects/blip2/train/finetune_scanqa.yaml"
        ;;
    2)
        echo -e "${BLUE}Starting SQA3D finetuning...${NC}"
        CONFIG="lavis/projects/blip2/train/finetune_sqa.yaml"
        ;;
    3)
        echo -e "${BLUE}Running dataset test...${NC}"
        conda run -n lavis python test_dataset_loading.py
        exit 0
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Check if running on GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Warning: nvidia-smi not found. Are you on a GPU node?${NC}"
    echo "To allocate a GPU node, run:"
    echo "  salloc -A 3dllms --nodes=1 -G 1 -t 8:00:00 --mem=48G --partition=mb-l40s"
    read -p "Continue anyway? [y/N]: " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ GPU available:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file: $CONFIG${NC}"
echo ""

# Warn about distributed training config
echo -e "${BLUE}Note: Config is set for world_size=16 (distributed training)${NC}"
echo "If training on single GPU, you may need to adjust the config."
read -p "Continue with training? [y/N]: " start_train

if [[ ! $start_train =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

# Start training
echo ""
echo -e "${GREEN}Starting training...${NC}"
echo "Command: conda run -n lavis python train.py --cfg-path $CONFIG"
echo ""

conda run -n lavis python train.py --cfg-path "$CONFIG"

echo ""
echo -e "${GREEN}Training complete!${NC}"
