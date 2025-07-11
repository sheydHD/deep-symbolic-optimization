# Deep Symbolic Optimization - Modernization Roadmap

> **Status**: Draft v1.0 (July 2025)
>
> This document outlines the step-by-step plan for modernizing the DSO codebase from legacy Python 3.6/TensorFlow 1.14 to current Python and ML frameworks.

## 1. Infrastructure & Environment Setup

### 1.1 Package Management

- [ ] **Implement uv-based environment management**
  - Create `tools/bash/setup/setup_modern.sh` for Unix/macOS
  - Create `tools/bat/setup/setup_modern.bat` for Windows
  - Update `main.sh`/`main.bat` with a "Modern Setup" option
  - Add helpers to detect/install uv CLI tool

### 1.2 Base Requirements

- [ ] **Create modern dependency specifications**
  - Define baseline dependencies in `configs/requirements/base.txt`
  - Define regression-specific dependencies in `configs/requirements/base_regression.txt`
  - Define ML framework dependencies in `configs/requirements/base_tf2.txt`/`base_pytorch.txt`

### 1.3 Documentation System

- [ ] **Upgrade docs infrastructure**
  - Update `mkdocs.yml` for modern MkDocs
  - Add Material for MkDocs theme and extensions
  - Create `tools/bash/docs/build_docs.sh` helper script
  - Integrate docs building into main CLI

## 2. Core Package Modernization

### 2.1 Python & Packaging

- [ ] **Update Python compatibility**
  - Target Python 3.10+ as minimum version
  - Convert `setup.py` to `pyproject.toml` for PEP 517/518 compliance
  - Update build system to use modern tooling (setuptools 65+)
  - Add mypy configuration and typing stubs
  - Set up Ruff for linting (replacing flake8)

### 2.2 TensorFlow Migration

- [ ] **Create TF compatibility layer**
  - Implement `dso/compat/__init__.py` with framework detection
  - Implement `dso/compat/tf.py` wrapper for TF 1.x → 2.x API conversion
  - Audit and replace all direct `tf.` imports with compat layer
  - Remove `tf.reset_default_graph()` and Session-based code
  - Move to eager execution with `tf.function` decorators
  - Replace TF 1.x optimizers with Keras optimizers

### 2.3 Cython/Numba Modernization

- [ ] **Update native code extensions**
  - Modernize `cyfunc.pyx` with typed memoryviews and modern Cython patterns
  - Update Numba-accelerated code to latest numba API
  - Benchmark alternative implementations (pure Python vs. Cython vs. Numba)
  - Move to `pyproject.toml` for Cython build configuration

## 3. Task-Specific Modernization

### 3.1 Regression Task

- [ ] **Update regression components**
  - Modernize NumPy usage for compatibility with NumPy 1.24+
  - Update scikit-learn integration for latest API
  - Re-optimize polynomial fitting code for modern libraries
  - Add type hints to regression task code

### 3.2 Control Task

- [ ] **Update reinforcement learning components**
  - Migrate from Gym to Gymnasium
  - Update from stable-baselines to stable-baselines3
  - Modernize custom environment code
  - Update visualization dependencies
  - Replace deprecated RL utilities

## 4. Testing & Validation

### 4.1 Test Infrastructure

- [ ] **Modernize test suite**
  - Update pytest configuration for modern pytest
  - Add fixtures for different framework backends
  - Set up test parameterization for Python 3.10/3.11
  - Add performance regression tests

### 4.2 CI/CD Setup

- [ ] **Set up continuous integration**
  - Configure GitHub Actions workflow
  - Add automated testing on multiple Python versions
  - Add documentation auto-publishing
  - Implement pre-commit hooks for code quality

### 4.3 Benchmarking

- [ ] **Performance validation**
  - Create benchmarking scripts for regression tasks
  - Create benchmarking scripts for control tasks
  - Compare performance between legacy and modern versions
  - Document performance differences

## 5. User Experience Improvements

### 5.1 CLI Modernization

- [ ] **Enhance command-line interface**
  - Create modern CLI with rich/typer for better UX
  - Add progress bars and colorful output
  - Implement improved error handling and reporting
  - Add interactive wizards for common tasks

### 5.2 Notebooks & Examples

- [ ] **Create modern examples**
  - Build Jupyter notebooks showcasing modern usage
  - Create Google Colab examples
  - Update README with modern examples
  - Add quick-start guides for new users

## 6. Implementation Phases

### Phase 1: Foundation (Week 1-2)

- Set up modern environment management with uv
- Create base requirements files
- Update documentation system
- Begin compatibility layer implementation

### Phase 2: Core Systems (Week 3-4)

- Complete TensorFlow 2.x migration
- Update Cython/Numba code
- Implement regression task modernization
- Add initial tests

### Phase 3: Extended Components (Week 5-6)

- Complete control task modernization
- Finish test suite updates
- Implement CI/CD pipeline
- Add benchmarking

### Phase 4: Polish & Release (Week 7-8)

- Finalize CLI improvements
- Create examples and notebooks
- Complete documentation
- Release version 2.0

## 7. Migration Strategy

To ensure backward compatibility while modernizing:

1. Keep the `compat-py36` branch for legacy code
2. Implement the compatibility layer early
3. Add framework detection to support both TF1 (legacy) and TF2 (modern)
4. Write thorough tests to validate each migration step
5. Document API changes for users

This approach allows gradual migration while maintaining functionality for users who cannot immediately upgrade.
