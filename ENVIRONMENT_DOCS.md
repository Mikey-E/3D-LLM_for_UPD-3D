# Environment Documentation Files

This directory contains several files to help you reproduce the exact environment setup for 3D-LLM.

## Files Overview

### 1. INSTALLATION_GUIDE.md
**Purpose**: Comprehensive installation instructions with troubleshooting
- Step-by-step installation guide
- Multiple installation options (conda, pip, manual)
- Common issues and solutions
- System requirements and verification steps

**Use this when**: You need detailed instructions or encounter installation issues

### 2. requirements_core.txt
**Purpose**: Minimal core dependencies for the project
- Lists essential packages with version constraints
- Faster installation with only necessary packages
- Easier to maintain and understand

**Installation**:
```bash
conda create -n lavis python=3.8 -y
conda activate lavis
pip install -r requirements_core.txt
cd SalesForce-LAVIS && pip install -e .
```

### 3. requirements_lavis.txt
**Purpose**: Complete pip freeze output
- Every single package in the working environment
- Exact versions for perfect reproducibility
- Generated from a verified working installation

**Installation**:
```bash
conda create -n lavis python=3.8 -y
conda activate lavis
pip install -r requirements_lavis.txt
```

### 4. environment_lavis.yml
**Purpose**: Complete conda environment specification
- Includes both conda and pip packages
- Preserves the exact environment state
- Can recreate the environment with one command

**Installation**:
```bash
conda env create -f environment_lavis.yml
conda activate lavis
```

## Quick Start

### Recommended Method (Fastest)
```bash
# Create environment
conda create -n lavis python=3.8 -y
conda activate lavis

# Install from core requirements
pip install -r requirements_core.txt

# Install LAVIS in editable mode
cd SalesForce-LAVIS
pip install -e .
cd ..

# Install final dependency
pip install positional_encodings
```

### Exact Reproduction Method
```bash
# Use the complete conda environment file
conda env create -f environment_lavis.yml
conda activate lavis
```

## Verification

After installation, verify everything works:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import lavis; print('LAVIS ready')"
```

## File Generation Details

These files were generated on:
- **Date**: October 3, 2025
- **System**: Linux cluster with SLURM
- **GPU**: NVIDIA L40S (46GB)
- **Python**: 3.8.20
- **CUDA**: 11.8
- **PyTorch**: 2.4.1+cu118

## Notes

- **Python 3.8 Required**: The environment specifically uses Python 3.8 due to compatibility constraints with certain dependencies (notably numpy and spacy versions)
- **CUDA 11.8**: PyTorch is built with CUDA 11.8. Ensure your system has compatible CUDA drivers
- **GPU Required**: This project requires a CUDA-capable GPU for optimal performance
- **Spacy Version**: Use spacy<3.6 for Python 3.8 compatibility

## Troubleshooting

If you encounter issues:
1. Check `INSTALLATION_GUIDE.md` for common problems and solutions
2. Ensure you're using Python 3.8
3. Verify GPU access with `nvidia-smi`
4. Try the exact reproduction method using `environment_lavis.yml`

## Updates

To update these files after modifying your environment:

```bash
# Update pip requirements
pip freeze > requirements_lavis.txt

# Update conda environment
conda env export --name lavis > environment_lavis.yml
```
