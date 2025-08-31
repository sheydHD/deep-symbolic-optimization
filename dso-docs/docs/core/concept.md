# DSO – Overview

> Version: 1.0 • Last updated: 2025-07-07

Deep Symbolic Optimization (DSO) is a framework for generating compact, human-readable mathematical expressions that approximate data or control systems.

## Scope

1. **Symbolic Regression** – discover analytic formulas that fit data.
2. **Control Tasks** – evolve interpretable controllers for simulated environments.
3. **Hybrid GP + RL** – meld genetic programming with gradient-based policy search.

## Key Features

- Modular task interface (`dso.task`) supporting regression and control back-ends.
- Pluggable policy representations (RNN, Transformer) with prior libraries.
- GPU-accelerated evaluation kernels via Cython/Numba.
- Extensible search spaces loaded from JSON definitions.

## Audience

- **Researchers** in symbolic AI and reinforcement learning.
- **Engineers** needing compact surrogate models or controllers.
- **Educators** demonstrating program synthesis techniques.

## Quick Links

- Getting started: [`core/getting_started.md`](core/getting_started.md)
- Configuration guide: [`core/setup.md`](core/setup.md)
