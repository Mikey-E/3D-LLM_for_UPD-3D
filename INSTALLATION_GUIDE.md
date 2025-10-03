# 3D-LLM Installation Guide

This document provides detailed installation instructions and environment specifications for the 3D-LLM project.

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on NVIDIA L40S with 46GB memory)
- **CUDA Version**: 11.8 or compatible
- **Python**: 3.8
- **OS**: Linux (tested on Ubuntu/CentOS)

## Environment Setup

### Option 1: Using Conda Environment File (Recommended)

The `environment_lavis.yml` file contains the complete conda environment specification.

```bash
# Create environment from YAML file
conda env create -f environment_lavis.yml

# Activate the environment
conda activate lavis
```

### Option 2: Manual Installation (Step-by-Step)

Follow these steps for a manual installation:

#### 1. Allocate GPU Resources (on SLURM cluster)

```bash
salloc -A 3dllms --nodes=1 -G 1 -t 8:00:00 --mem=48G --partition=mb-l40s
```

#### 2. Create Conda Environment

```bash
# Create a new conda environment with Python 3.8
conda create -n lavis python=3.8 -y

# Activate the environment
conda activate lavis
```

#### 3. Clone and Install LAVIS

```bash
# Clone the SalesForce LAVIS repository (if not already present)
git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS

# Navigate to the directory
cd SalesForce-LAVIS

# Install LAVIS in editable mode
pip install -e .
```

#### 4. Install Additional Dependencies

```bash
# Install spacy (Python 3.8 compatible version)
pip install 'spacy<3.6'

# Install timm
pip install timm==0.4.12

# Install transformers
pip install transformers==4.33.2

# Install positional_encodings
pip install positional_encodings
```

### Option 3: Using Requirements File

```bash
# Create and activate conda environment
conda create -n lavis python=3.8 -y
conda activate lavis

# Install from requirements file
pip install -r requirements_lavis.txt
```

## Core Dependencies

### Key Packages and Versions

- **PyTorch**: 2.4.1+cu118 (with CUDA 11.8)
- **Transformers**: 4.33.2
- **timm**: 0.4.12
- **spacy**: 3.5.4
- **positional_encodings**: 6.0.3
- **LAVIS**: 1.0.1 (editable install from SalesForce-LAVIS/)

### CUDA Dependencies

The environment includes NVIDIA CUDA libraries:
- nvidia-cuda-nvrtc-cu11==11.8.89
- nvidia-cuda-runtime-cu11==11.8.89
- nvidia-cudnn-cu11==9.1.0.70
- nvidia-cublas-cu11==11.11.3.6
- And other CUDA toolkit components

## Verification

After installation, verify the setup:

```bash
python -c "
import torch
import transformers
import timm
import positional_encodings
import spacy

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
print('Transformers version:', transformers.__version__)
print('Spacy version:', spacy.__version__)
print('All packages imported successfully!')
"
```

Expected output should show:
- PyTorch version: 2.4.1+cu118
- CUDA available: True
- CUDA version: 11.8
- GPU: NVIDIA L40S (or your GPU model)
- Transformers version: 4.33.2
- Spacy version: 3.5.4

## Common Issues and Solutions

### Issue 1: Spacy Installation Fails

**Problem**: Newer spacy versions require numpy>=2.0.0, which is incompatible with Python 3.8.

**Solution**: Install an older compatible version:
```bash
pip install 'spacy<3.6'
```

### Issue 2: Missing timm or transformers

**Problem**: LAVIS installation may not install all dependencies correctly.

**Solution**: Install them explicitly:
```bash
pip install timm==0.4.12
pip install transformers==4.33.2
```

### Issue 3: CUDA Not Available

**Problem**: PyTorch cannot detect CUDA/GPU.

**Solution**: 
- Ensure you're on a GPU node (hostname should contain GPU partition name)
- Verify with `nvidia-smi`
- Reinstall PyTorch with CUDA support if needed

## File Descriptions

- **`requirements_lavis.txt`**: Complete pip freeze output of all installed packages
- **`environment_lavis.yml`**: Complete conda environment specification including all dependencies
- **`INSTALLATION_GUIDE.md`**: This file - comprehensive installation instructions

## Next Steps

After successful installation, you can:

1. **Download Checkpoints**: Follow instructions in the main README.md for downloading pretrained/finetuning checkpoints
2. **Run Inference**: See the "Quick Start: Inference" section in README.md
3. **Fine-tune Models**: See the "Finetuning" section in README.md

## Support

For issues specific to:
- **LAVIS**: Visit [SalesForce LAVIS GitHub](https://github.com/salesforce/LAVIS)
- **3D-LLM**: Visit the main repository or refer to the paper

## Environment Information

This installation was tested on:
- **Date**: October 3, 2025
- **System**: Linux cluster with SLURM
- **GPU**: NVIDIA L40S (46GB)
- **Driver Version**: 570.172.08
- **Python**: 3.8.20
