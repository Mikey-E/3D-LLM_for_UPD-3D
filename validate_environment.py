#!/usr/bin/env python3
"""
3D-LLM Environment Validation Script

This script validates that the environment is properly set up for 3D-LLM.
Run this after installation to ensure everything is configured correctly.

Usage:
    python validate_environment.py
"""

import sys
import subprocess

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_python_version():
    """Check Python version."""
    print("\n[1/8] Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor == 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro}")
        print(f"     Expected: Python 3.8.x")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("\n[2/8] Checking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: Yes")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     GPU device: {torch.cuda.get_device_name(0)}")
            print(f"     GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"  ⚠ CUDA available: No")
            print(f"     GPU acceleration will not be available!")
            return False
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False

def check_transformers():
    """Check Transformers installation."""
    print("\n[3/8] Checking Transformers...")
    try:
        import transformers
        print(f"  ✓ Transformers version: {transformers.__version__}")
        if transformers.__version__ == "4.33.2":
            print(f"     Expected version match")
        else:
            print(f"     Warning: Expected 4.33.2")
        return True
    except ImportError as e:
        print(f"  ✗ Transformers not found: {e}")
        return False

def check_timm():
    """Check timm installation."""
    print("\n[4/8] Checking timm...")
    try:
        import timm
        print(f"  ✓ timm installed")
        return True
    except ImportError as e:
        print(f"  ✗ timm not found: {e}")
        return False

def check_spacy():
    """Check spacy installation."""
    print("\n[5/8] Checking spacy...")
    try:
        import spacy
        print(f"  ✓ spacy version: {spacy.__version__}")
        if spacy.__version__.startswith("3.5"):
            print(f"     Compatible version")
        else:
            print(f"     Warning: Expected 3.5.x")
        return True
    except ImportError as e:
        print(f"  ✗ spacy not found: {e}")
        return False

def check_positional_encodings():
    """Check positional_encodings installation."""
    print("\n[6/8] Checking positional_encodings...")
    try:
        import positional_encodings
        print(f"  ✓ positional_encodings installed")
        return True
    except ImportError as e:
        print(f"  ✗ positional_encodings not found: {e}")
        return False

def check_lavis():
    """Check LAVIS installation."""
    print("\n[7/8] Checking LAVIS...")
    try:
        import lavis
        print(f"  ✓ LAVIS installed")
        # Try to import key components
        from lavis.models import load_model_and_preprocess
        print(f"     Core components accessible")
        return True
    except ImportError as e:
        print(f"  ✗ LAVIS not found or incomplete: {e}")
        print(f"     Try: cd SalesForce-LAVIS && pip install -e .")
        return False

def check_gpu_via_nvidia_smi():
    """Check GPU via nvidia-smi command."""
    print("\n[8/8] Checking GPU via nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"  ✓ GPU detected: {gpu_info}")
            return True
        else:
            print(f"  ✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print(f"  ✗ nvidia-smi not found (NVIDIA drivers may not be installed)")
        return False
    except Exception as e:
        print(f"  ⚠ Could not run nvidia-smi: {e}")
        return False

def main():
    """Run all validation checks."""
    print_header("3D-LLM Environment Validation")
    print("\nValidating environment setup for 3D-LLM project...")
    
    checks = [
        check_python_version(),
        check_pytorch(),
        check_transformers(),
        check_timm(),
        check_spacy(),
        check_positional_encodings(),
        check_lavis(),
        check_gpu_via_nvidia_smi(),
    ]
    
    print_header("Validation Summary")
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✓ Environment is fully configured and ready to use!")
        print("\nNext steps:")
        print("  1. Download checkpoints (see README.md)")
        print("  2. Run inference: python inference.py")
        sys.exit(0)
    elif passed >= 6:
        print("\n⚠ Environment is mostly configured but has some issues.")
        print("  Review the warnings above and fix as needed.")
        print("  See INSTALLATION_GUIDE.md for troubleshooting.")
        sys.exit(1)
    else:
        print("\n✗ Environment setup is incomplete.")
        print("  Please review the errors above.")
        print("  See INSTALLATION_GUIDE.md for installation instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
