# 3D-LLM Environment Quick Reference Card

## 🚀 Quick Start Commands

### Method 1: Complete Environment (Exact Reproduction)
```bash
conda env create -f environment_lavis.yml
conda activate lavis
```

### Method 2: Core Dependencies (Faster)
```bash
conda create -n lavis python=3.8 -y
conda activate lavis
pip install -r requirements_core.txt
cd SalesForce-LAVIS && pip install -e . && cd ..
pip install positional_encodings
```

### Method 3: From Complete Requirements
```bash
conda create -n lavis python=3.8 -y
conda activate lavis
pip install -r requirements_lavis.txt
```

## ✅ Verification

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import lavis; import transformers; import timm; print('All imports OK')"
nvidia-smi  # Check GPU
```

## 📦 Core Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8.20 | Base environment |
| PyTorch | 2.4.1+cu118 | Deep learning framework |
| CUDA | 11.8 | GPU acceleration |
| Transformers | 4.33.2 | HuggingFace transformers |
| timm | 0.4.12 | Image models |
| spacy | 3.5.4 | NLP library |
| LAVIS | 1.0.1 | Multimodal framework |

## 🔧 Common Commands

### Check Environment
```bash
conda env list                    # List all environments
conda activate lavis             # Activate environment
which python                     # Verify Python path
pip list | grep -E "torch|lavis" # Check key packages
```

### GPU Allocation (SLURM)
```bash
salloc -A 3dllms --nodes=1 -G 1 -t 8:00:00 --mem=48G --partition=mb-l40s
nvidia-smi  # Verify GPU access
```

### Update Environment Files
```bash
pip freeze > requirements_lavis.txt
conda env export --name lavis > environment_lavis.yml
```

## 📁 Documentation Files

| File | Size | Purpose |
|------|------|---------|
| `INSTALLATION_GUIDE.md` | 4.8KB | Detailed setup instructions |
| `ENVIRONMENT_DOCS.md` | 3.3KB | Files overview & usage |
| `QUICK_REFERENCE.md` | This file | Quick commands reference |
| `requirements_core.txt` | 680B | Minimal dependencies |
| `requirements_lavis.txt` | 3.8KB | Complete package list |
| `environment_lavis.yml` | 6.2KB | Conda environment spec |

## 🐛 Troubleshooting Quick Fixes

### Problem: Spacy won't install
```bash
pip install 'spacy<3.6'
```

### Problem: Missing timm/transformers
```bash
pip install timm==0.4.12 transformers==4.33.2
```

### Problem: CUDA not detected
```bash
# Verify GPU access first
nvidia-smi
# If no GPU, request one via SLURM
salloc -A 3dllms --nodes=1 -G 1 -t 8:00:00 --mem=48G --partition=mb-l40s
```

### Problem: Import errors after installation
```bash
# Reinstall LAVIS in editable mode
cd SalesForce-LAVIS
pip install -e .
cd ..
```

## 📖 Full Documentation

For detailed information, see:
- **Full Installation Guide**: `INSTALLATION_GUIDE.md`
- **Environment Files Overview**: `ENVIRONMENT_DOCS.md`
- **Original README**: `README.md`

## 💡 Tips

- Always activate the environment: `conda activate lavis`
- Verify GPU access before long operations: `nvidia-smi`
- Use `requirements_core.txt` for faster reinstalls
- Use `environment_lavis.yml` for exact reproduction
- Check `INSTALLATION_GUIDE.md` for detailed troubleshooting

---
*Generated: October 3, 2025 | Environment: Python 3.8.20, PyTorch 2.4.1+cu118, CUDA 11.8*
