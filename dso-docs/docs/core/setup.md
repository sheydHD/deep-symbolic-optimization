# Installation & Setup Guide

> Version: 1.0 • Last updated: 2025-09-01

This comprehensive guide covers installation, configuration, and troubleshooting for Deep Symbolic Optimization (DSO). Follow these steps to get DSO running on your system.

## System Requirements

### Operating Systems
- **Linux** (Ubuntu 18.04+, CentOS 7+, or similar)
- **macOS** (10.14+ with Xcode command line tools)
- **Windows** (with WSL2 recommended)

### Hardware Requirements
- **CPU**: Modern multi-core processor (Intel/AMD x64)
- **RAM**: Minimum 8GB, recommended 16GB+ for large problems
- **Storage**: 2GB free space for installation
- **GPU** (Optional): CUDA-compatible GPU for acceleration

### Software Prerequisites

**Required:**
- **Python 3.10 or 3.11** - Check with `python --version`
- **Git** - For repository cloning
- **C/C++ Compiler** - GCC (Linux), Clang (macOS), or MSVC (Windows)

**Recommended:**
- **uv package manager** - Fast Python package management
- **CUDA Toolkit** - For GPU acceleration (if using GPU)

## Installation Methods

### Method 1: Quick Install (Recommended)

This is the fastest way to get started with DSO:

```bash
# 1. Clone the repository
git clone https://github.com/sheydHD/deep-symbolic-optimization
cd deep-symbolic-optimization

# 2. Run automated setup
./main.sh
# When prompted, press '1' to setup environment

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import dso; print('DSO installed successfully!')"
```

### Method 2: Manual Installation

For more control over the installation process:

```bash
# 1. Clone and enter directory
git clone https://github.com/sheydHD/deep-symbolic-optimization
cd deep-symbolic-optimization

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# 3. Install uv package manager
pip install uv

# 4. Install dependencies
uv pip compile configs/requirements/in_files/requirements.in
uv pip install -r requirements.txt

# 5. Install DSO in development mode
pip install -e .

# 6. Compile Cython extensions
python setup.py build_ext --inplace
```

### Method 3: Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t dso:latest .

# Run DSO in container
docker run -it --gpus all -v $(pwd)/data:/data dso:latest

# Inside container, DSO is ready to use
python tools/python/benchmark/benchmark.py /data/config.json
```

## Understanding the Setup Script

The `main.sh` script provides a user-friendly interface for common operations:

### Script Options

When you run `./main.sh`, you get these options:

1. **Setup Environment** - Creates virtual environment and installs dependencies
2. **Run Tests** - Executes the full test suite  
3. **Run Benchmarks** - Launches benchmark experiments
4. **Clean Environment** - Removes virtual environment and build artifacts

### What Setup Does

The setup process performs these steps:

1. **Environment Creation**: Creates isolated Python virtual environment
2. **Dependency Resolution**: Compiles requirements from `.in` files
3. **Package Installation**: Installs all required Python packages
4. **Cython Compilation**: Builds fast numerical extensions
5. **DSO Installation**: Installs DSO package in editable mode
6. **Verification**: Runs basic import tests

## Configuration & Customization

### Environment Variables

Set these variables to customize DSO behavior:

```bash
# GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0

# Parallel processing
export OMP_NUM_THREADS=8

# Memory management  
export DSO_MEMORY_LIMIT=16G

# Add to your ~/.bashrc for persistence
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
```

### Custom Installation Paths

Install DSO in a custom location:

```bash
# Install to custom directory
export DSO_INSTALL_DIR=/opt/dso
git clone https://github.com/your-org/dso.git $DSO_INSTALL_DIR
cd $DSO_INSTALL_DIR
./main.sh

# Add to PATH
echo 'export PATH=$PATH:/opt/dso' >> ~/.bashrc
```

## Verification & Testing

### Basic Verification

Confirm DSO is properly installed:

```bash
# Test Python import
python -c "import dso; print(f'DSO version: {dso.__version__}')"

# Test Cython extensions
python -c "from dso.cyfunc import python_execute; print('Cython OK')"

# Test GPU support (if available)
python -c "import tensorflow as tf; print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Comprehensive Testing

Run the full test suite:

```bash
# Quick test (essential components)
pytest -q dso/dso/test/ -k "not slow"

# Full test suite (includes slow tests)
pytest dso/dso/test/ --verbose

# Test specific components
pytest dso/dso/test/test_core.py -v
pytest dso/dso/test/test_functions.py -v
pytest dso/dso/test/test_program.py -v
```

### Performance Benchmarks

Verify performance on standard benchmarks:

```bash
# Quick benchmark (2-3 minutes)
python tools/python/benchmark/benchmark.py dso/config/examples/regression/Nguyen-1.json

# Medium benchmark (10-15 minutes)  
python tools/python/benchmark/benchmark.py dso/config/examples/regression/Nguyen-7.json

# Check GPU acceleration (if available)
python tools/python/benchmark/benchmark.py dso/config/examples/regression/Nguyen-1.json --gpu
```

## Platform-Specific Setup

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev git

# For GPU support
sudo apt-get install nvidia-cuda-toolkit

# Clone and setup DSO
git clone https://github.com/your-org/dso.git && cd dso
./main.sh
```

### CentOS/RHEL Linux

```bash
# Install system dependencies  
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel git

# Clone and setup DSO
git clone https://github.com/your-org/dso.git && cd dso
./main.sh
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git (if needed)
brew install python@3.11 git

# Clone and setup DSO
git clone https://github.com/your-org/dso.git && cd dso
./main.sh
```

### Windows (WSL2)

```bash
# Install WSL2 and Ubuntu
wsl --install Ubuntu

# Inside WSL2, follow Ubuntu instructions
sudo apt-get update && sudo apt-get install build-essential python3-dev git
git clone https://github.com/your-org/dso.git && cd dso
./main.sh
```

## Troubleshooting

### Common Installation Issues

#### Python Version Conflicts

**Issue**: "Python 3.12 is not supported"
```bash
# Solution: Install supported Python version
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-dev python3.11-venv

# macOS
brew install python@3.11

# Update symbolic link
sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
```

#### Cython Compilation Errors

**Issue**: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Solution: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```

**Issue**: "gcc: command not found" (Linux)
```bash
# Solution: Install build tools
sudo apt-get install build-essential  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"  # CentOS/RHEL
```

#### Memory Issues

**Issue**: "MemoryError during installation"
```bash
# Solution: Reduce memory usage
export MAKEFLAGS="-j1"  # Single-threaded compilation
pip install --no-cache-dir -e .
```

#### Permission Errors

**Issue**: "Permission denied" errors
```bash
# Solution: Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Or fix permissions (not recommended)
sudo chown -R $USER:$USER /path/to/dso
```

### GPU Setup Issues

#### CUDA Not Found

**Issue**: "CUDA toolkit not found"
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### GPU Memory Issues

**Issue**: "Out of GPU memory"
```bash
# Solution: Limit GPU memory usage
export TF_MEMORY_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# Or use smaller batch sizes in config
# "batch_size": 500  # instead of 1000
```

### Runtime Issues

#### Import Errors

**Issue**: "ModuleNotFoundError: No module named 'dso'"
```bash
# Solution: Activate virtual environment
source .venv/bin/activate

# Or reinstall in development mode
pip install -e .
```

#### Performance Issues

**Issue**: Very slow training
```bash
# Check CPU utilization
htop

# Increase parallel threads
export OMP_NUM_THREADS=8

# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Reduce problem size for testing
# Edit config: "n_samples": 10000  # instead of 2000000
```

## Advanced Configuration

### Multi-GPU Setup

For multiple GPU systems:

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configure TensorFlow for multi-GPU
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
```

### Cluster/HPC Setup

For high-performance computing environments:

```bash
# Load required modules (example for SLURM)
module load python/3.11
module load gcc/9.3.0
module load cuda/11.8

# Setup in home directory
cd $HOME
git clone https://github.com/sheydHD/deep-symbolic-optimization
cd deep-symbolic-optimization
./main.sh

# Submit job
sbatch scripts/submit_dso_job.sh
```

### Docker Configuration

Create custom Docker image:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and setup DSO
WORKDIR /app
RUN git clone https://github.com/your-org/dso.git .
RUN ./main.sh

# Set entrypoint
ENTRYPOINT ["python", "tools/python/benchmark/benchmark.py"]
```

## Getting Help

### Documentation Resources
- **[Getting Started](getting_started.md)** - Quick start tutorial
- **[Core Concepts](concept.md)** - Understanding DSO fundamentals
- **[Troubleshooting FAQ](#troubleshooting)** - Common issues and solutions

### Community Support
- **GitHub Issues** - Report bugs and request features
- **Discussions** - Ask questions and share experiences
- **Stack Overflow** - Tag questions with `deep-symbolic-optimization`

### Professional Support
For enterprise deployments or consulting:
- **Email**: support@dso-project.org
- **Commercial License** - Contact for licensing options

---

**You're all set!** 🚀 DSO is now installed and ready for mathematical discovery. Proceed to the [Getting Started](getting_started.md) guide to run your first experiment.
