# Installation Guide

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
pip install deep-symbolic-optimization
```

### Method 2: Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/sheydHD/deep-symbolic-optimization.git
cd deep-symbolic-optimization
pip install -e .
```

### Method 3: Conda Installation

```bash
conda install -c conda-forge deep-symbolic-optimization
```

## Virtual Environment Setup

It is recommended to use a virtual environment to avoid dependency conflicts:

### Using venv
```bash
python -m venv dso-env
source dso-env/bin/activate  # On Windows: dso-env\Scripts\activate
pip install deep-symbolic-optimization
```

### Using conda
```bash
conda create -n dso-env python=3.11
conda activate dso-env
pip install deep-symbolic-optimization
```

## Verify Installation

Test the installation by running:

```python
import dso
print(dso.__version__)
```

## Development Installation

For contributors and developers:

```bash
git clone https://github.com/sheydHD/deep-symbolic-optimization.git
cd deep-symbolic-optimization
pip install -e ".[dev]"
```

This installs additional development dependencies for testing and documentation.

## GPU Support Setup

To enable GPU acceleration:

1. Install CUDA toolkit and cuDNN
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Common Installation Issues

### Cython Compilation Errors
If you encounter Cython compilation issues:
```bash
pip install --upgrade setuptools cython
pip install deep-symbolic-optimization --no-cache-dir
```

### Missing C++ Compiler
On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
```

On macOS:
```bash
xcode-select --install
```

### Permission Errors
Use user installation if you encounter permission errors:
```bash
pip install --user deep-symbolic-optimization
```