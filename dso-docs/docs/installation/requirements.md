# System Requirements

## Hardware Requirements

### Minimum Requirements
- **CPU**: x86_64 processor with SSE4.2 support
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB available disk space

### Recommended Configuration
- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 16 GB or more for large datasets
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)
- **Storage**: SSD for improved I/O performance

## Software Requirements

### Operating System
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent
- **macOS**: 10.14+ (Mojave or later)
- **Windows**: Windows 10+ with WSL2 recommended

### Python Environment
- **Python**: 3.10 or 3.11 (Python 3.12 not yet supported)
- **pip**: Latest version recommended
- **virtualenv** or **conda**: For environment isolation

### Development Tools
- **GCC/Clang**: C/C++ compiler for Cython extensions
- **Git**: For source code management
- **Make**: Build automation tool

## CUDA Support (Optional)

For GPU acceleration:
- **CUDA Toolkit**: 11.7 or 12.x
- **cuDNN**: Compatible version with CUDA toolkit
- **NVIDIA GPU**: Compute capability 6.0 or higher

## Dependency Overview

Core dependencies are automatically installed with DSO:
- **NumPy**: Numerical computing
- **Cython**: Performance-critical extensions
- **ConfigArgParse**: Configuration management
- **scikit-learn**: Machine learning utilities
- **numba**: Just-in-time compilation