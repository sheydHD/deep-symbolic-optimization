# -*- coding: utf-8 -*-
"""
Hook's Law 3D Dataset Generator for MIMO Symbolic Regression

This module generates stress-strain data based on Hook's law for 3D mechanics.
It creates a perfect MIMO test case with 6 inputs (strain components) and 
6 outputs (stress components) with known linear relationships.

The relationship follows: stress = C * strain
Where C is the 6x6 isotropic stiffness matrix.

Author: DSO MIMO Extension
"""

import numpy as np
import os
from typing import Tuple, Optional

def generate_hooks_law_data(N: int, E: float = 1e8, nu: float = 0.3, 
                           strain_range: Tuple[float, float] = (0.0, 0.01),
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Hook's law data for DSO MIMO training.
    
    Parameters
    ----------
    N : int
        Number of samples to generate
    E : float
        Young's modulus (default = 1e8 Pa)
    nu : float  
        Poisson's ratio (default = 0.3)
    strain_range : tuple
        Range for strain values (min, max)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (N, 6)
        Strain vectors in Voigt notation [ε₁₁, ε₂₂, ε₃₃, γ₁₂, γ₁₃, γ₂₃]
    y : ndarray of shape (N, 6) 
        Stress vectors in Voigt notation [σ₁₁, σ₂₂, σ₃₃, τ₁₂, τ₁₃, τ₂₃]
    C : ndarray of shape (6, 6)
        Isotropic stiffness matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random strain vectors (N, 6)
    X = np.random.uniform(strain_range[0], strain_range[1], size=(N, 6))
    
    # Use engineering shear strains directly (no doubling for simplicity)
    # This makes the relationships consistent with the benchmark expectations
    X_for_stress = X.copy()  # No modification needed
    
    # Build isotropic stiffness matrix using Lame parameters
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lame parameter
    mu = E / (2 * (1 + nu))                    # Second Lame parameter (shear modulus)
    
    C = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],  # σ₁₁ = (λ+2μ)ε₁₁ + λε₂₂ + λε₃₃
        [lam,        lam + 2*mu, lam,        0,  0,  0],  # σ₂₂ = λε₁₁ + (λ+2μ)ε₂₂ + λε₃₃  
        [lam,        lam,        lam + 2*mu, 0,  0,  0],  # σ₃₃ = λε₁₁ + λε₂₂ + (λ+2μ)ε₃₃
        [0,          0,          0,          mu, 0,  0],  # τ₁₂ = μγ₁₂
        [0,          0,          0,          0,  mu, 0],  # τ₁₃ = μγ₁₃
        [0,          0,          0,          0,  0,  mu]  # τ₂₃ = μγ₂₃
    ])
    
    # Compute stresses: σ = C * ε
    y = np.einsum('ij,nj->ni', C, X_for_stress)
    
    return X, y, C


def create_hooks_law_benchmark_data(train_samples: int = 1000, test_samples: int = 500,
                                  E: float = 1e8, nu: float = 0.3,
                                  train_seed: int = 42, test_seed: int = 123) -> dict:
    """
    Create Hook's law benchmark data in the format expected by DSO.
    
    Parameters
    ----------
    train_samples : int
        Number of training samples
    test_samples : int
        Number of test samples  
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
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
    X_train, y_train, C = generate_hooks_law_data(
        train_samples, E=E, nu=nu, seed=train_seed
    )
    
    # Generate test data
    X_test, y_test, _ = generate_hooks_law_data(
        test_samples, E=E, nu=nu, seed=test_seed  
    )
    
    # Store material properties for reference
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test},
        'material_properties': {
            'E': E, 'nu': nu, 'lambda': lam, 'mu': mu
        },
        'stiffness_matrix': C,
        'true_expressions': [
            f"({lam + 2*mu:.2e})*x1 + ({lam:.2e})*x2 + ({lam:.2e})*x3",  # σ₁₁
            f"({lam:.2e})*x1 + ({lam + 2*mu:.2e})*x2 + ({lam:.2e})*x3",  # σ₂₂
            f"({lam:.2e})*x1 + ({lam:.2e})*x2 + ({lam + 2*mu:.2e})*x3",  # σ₃₃
            f"({mu:.2e})*x4",                                              # τ₁₂
            f"({mu:.2e})*x5",                                              # τ₁₃  
            f"({mu:.2e})*x6"                                               # τ₂₃
        ]
    }


def save_hooks_law_data(filepath: str, **kwargs):
    """Save Hook's law data to file for later use."""
    data = create_hooks_law_benchmark_data(**kwargs)
    np.savez_compressed(filepath, **data)
    return data


if __name__ == "__main__":
    # Example usage and validation
    print("Generating Hook's Law MIMO dataset...")
    
    # Generate sample data
    X, y, C = generate_hooks_law_data(100, seed=42)
    
    print(f"Input shape (strains): {X.shape}")
    print(f"Output shape (stresses): {y.shape}")
    print(f"Stiffness matrix shape: {C.shape}")
    
    print("\nSample strain vector (first row):")
    print(f"[ε₁₁, ε₂₂, ε₃₃, γ₁₂, γ₁₃, γ₂₃] = {X[0]}")
    
    print("\nCorresponding stress vector:")
    print(f"[σ₁₁, σ₂₂, σ₃₃, τ₁₂, τ₁₃, τ₂₃] = {y[0]}")
    
    print("\nStiffness matrix C:")
    print(C)
    
    # Verify the relationship: σ = C * ε
    stress_check = C @ X[0]
    print(f"\nVerification (C @ ε): {stress_check}")
    print(f"Original stress:      {y[0]}")
    print(f"Match: {np.allclose(stress_check, y[0])}")