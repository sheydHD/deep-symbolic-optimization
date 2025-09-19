# -*- coding: utf-8 -*-
"""
Simplified Hook's Law Dataset Generator for MIMO Symbolic Regression

This creates a much simpler version of Hook's law that DSO can actually learn:
- Scaled coefficients to order 1-10 instead of 10^8
- Clear linear relationships
- Separate normal and shear components

Author: DSO MIMO Extension
"""

import numpy as np
from typing import Tuple, Optional

def generate_simple_hooks_data(N: int, 
                              normal_coeff: float = 2.0,
                              coupling_coeff: float = 1.0, 
                              shear_coeff: float = 1.0,
                              strain_range: Tuple[float, float] = (0.0, 1.0),
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simplified Hook's law data for DSO MIMO training.
    
    Creates relationships like:
    - σ₁ = 2*ε₁ + 1*ε₂ + 1*ε₃     (normal stress with coupling)
    - σ₂ = 1*ε₁ + 2*ε₂ + 1*ε₃     (normal stress with coupling)  
    - σ₃ = 1*ε₁ + 1*ε₂ + 2*ε₃     (normal stress with coupling)
    - σ₄ = 1*ε₄                    (pure shear)
    - σ₅ = 1*ε₅                    (pure shear)
    - σ₆ = 1*ε₆                    (pure shear)
    
    Parameters
    ----------
    N : int
        Number of samples to generate
    normal_coeff : float
        Diagonal coefficient for normal stresses (default = 2.0)
    coupling_coeff : float
        Off-diagonal coefficient for normal stress coupling (default = 1.0)
    shear_coeff : float
        Coefficient for shear stresses (default = 1.0)
    strain_range : tuple
        Range for strain values (min, max)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (N, 6)
        Strain vectors [ε₁, ε₂, ε₃, ε₄, ε₅, ε₆]
    y : ndarray of shape (N, 6) 
        Stress vectors [σ₁, σ₂, σ₃, σ₄, σ₅, σ₆]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random strain vectors (N, 6)
    X = np.random.uniform(strain_range[0], strain_range[1], size=(N, 6))
    
    # Build simplified stiffness relationships
    # Normal stresses (with coupling)
    sigma1 = normal_coeff * X[:, 0] + coupling_coeff * X[:, 1] + coupling_coeff * X[:, 2]
    sigma2 = coupling_coeff * X[:, 0] + normal_coeff * X[:, 1] + coupling_coeff * X[:, 2]  
    sigma3 = coupling_coeff * X[:, 0] + coupling_coeff * X[:, 1] + normal_coeff * X[:, 2]
    
    # Shear stresses (pure, no coupling)
    sigma4 = shear_coeff * X[:, 3]
    sigma5 = shear_coeff * X[:, 4] 
    sigma6 = shear_coeff * X[:, 5]
    
    y = np.column_stack([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6])
    
    return X, y


def create_simple_hooks_benchmark_data(train_samples: int = 1000, test_samples: int = 500,
                                     train_seed: int = 42, test_seed: int = 123) -> dict:
    """
    Create simplified Hook's law benchmark data in the format expected by DSO.
    
    Parameters
    ----------
    train_samples : int
        Number of training samples
    test_samples : int
        Number of test samples  
    train_seed : int
        Seed for training data
    test_seed : int
        Seed for test data
        
    Returns
    -------
    data : dict
        Dictionary with 'train' and 'test' data
    """
    # Generate training data
    X_train, y_train = generate_simple_hooks_data(train_samples, seed=train_seed)
    
    # Generate test data
    X_test, y_test = generate_simple_hooks_data(test_samples, seed=test_seed)
    
    data = {
        "train": [X_train, y_train],
        "test": [X_test, y_test]
    }
    
    print(f"Generated simplified Hook's law data:")
    print(f"  Training: X{X_train.shape}, y{y_train.shape}")
    print(f"  Test: X{X_test.shape}, y{y_test.shape}")
    print(f"  Expected relationships:")
    print(f"    σ₁ = 2*ε₁ + ε₂ + ε₃")
    print(f"    σ₂ = ε₁ + 2*ε₂ + ε₃")  
    print(f"    σ₃ = ε₁ + ε₂ + 2*ε₃")
    print(f"    σ₄ = ε₄")
    print(f"    σ₅ = ε₅")
    print(f"    σ₆ = ε₆")
    
    return data


# Test the simplified version
if __name__ == "__main__":
    print("Testing simplified Hook's law data generation...")
    X, y = generate_simple_hooks_data(5)
    print("Sample data:")
    for i in range(5):
        print(f"  Strain: {X[i]}")
        print(f"  Stress: {y[i]}")
        print(f"  Check σ₁: {2*X[i][0] + X[i][1] + X[i][2]:.6f} vs {y[i][0]:.6f}")
        print(f"  Check σ₄: {X[i][3]:.6f} vs {y[i][3]:.6f}")
        print()