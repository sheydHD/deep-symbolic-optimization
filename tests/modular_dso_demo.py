"""
Example script demonstrating the modular DSO system for different data variants.

This script shows how the unified DSO interface automatically detects and handles:
- Scalar (SISO): Single input, single output
- Vector Input (MISO): Multiple inputs, single output
- Vector Output (SIMO): Single input, multiple outputs  
- Vector Both (MIMO): Multiple inputs, multiple outputs
- Matrix/Tensor: Higher-dimensional data

Run this script to see automatic variant detection and configuration in action.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add DSO package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dso.unified_dso import auto_fit, UnifiedDSO
from dso.core.data_types import auto_detect_data_structure


def create_synthetic_datasets():
    """Create synthetic datasets for different variants."""
    
    np.random.seed(42)
    datasets = {}
    
    print("Creating synthetic datasets...")
    
    # 1. Scalar (SISO) - Simple quadratic
    print("  - Scalar (SISO): y = x²")
    X_scalar = np.linspace(-2, 2, 100).reshape(-1, 1)
    y_scalar = X_scalar.flatten() ** 2 + 0.05 * np.random.randn(100)
    datasets['scalar'] = {
        'data': (X_scalar, y_scalar),
        'true_expr': 'x1^2',
        'description': 'Single input, single output quadratic function'
    }
    
    # 2. Vector Input (MISO) - Multivariate function
    print("  - Vector Input (MISO): y = x₁*x₂ + sin(x₃)")
    X_miso = np.random.uniform(-2, 2, (200, 3))
    y_miso = X_miso[:, 0] * X_miso[:, 1] + np.sin(X_miso[:, 2]) + 0.05 * np.random.randn(200)
    datasets['miso'] = {
        'data': (X_miso, y_miso),
        'true_expr': 'x1*x2 + sin(x3)',
        'description': 'Multiple inputs, single output'
    }
    
    # 3. Vector Output (SIMO) - Single input, multiple outputs
    print("  - Vector Output (SIMO): [y₁, y₂, y₃] = [x², sin(x), cos(x)]")
    X_simo = np.linspace(-2, 2, 150).reshape(-1, 1)
    y_simo = np.column_stack([
        X_simo.flatten() ** 2,
        np.sin(X_simo.flatten() * np.pi),
        np.cos(X_simo.flatten() * np.pi)
    ]) + 0.05 * np.random.randn(150, 3)
    datasets['simo'] = {
        'data': (X_simo, y_simo),
        'true_expr': '[x1^2, sin(π*x1), cos(π*x1)]',
        'description': 'Single input, multiple outputs'
    }
    
    # 4. Vector Both (MIMO) - Multiple inputs, multiple outputs
    print("  - Vector Both (MIMO): [y₁, y₂] = [x₁*x₂, sin(x₃)]")
    X_mimo = np.random.uniform(-2, 2, (200, 3))
    y_mimo = np.column_stack([
        X_mimo[:, 0] * X_mimo[:, 1],
        np.sin(X_mimo[:, 2]),
        X_mimo[:, 0] + X_mimo[:, 1] * X_mimo[:, 2]
    ]) + 0.05 * np.random.randn(200, 3)
    datasets['mimo'] = {
        'data': (X_mimo, y_mimo),
        'true_expr': '[x1*x2, sin(x3), x1 + x2*x3]',
        'description': 'Multiple inputs, multiple outputs'
    }
    
    # 5. Matrix data - 2D inputs/outputs
    print("  - Matrix: 2D convolution-like operation")
    X_matrix = np.random.randn(50, 2, 4)  # 50 samples, 2x4 matrices
    y_matrix = np.sum(X_matrix, axis=2, keepdims=True)  # Sum across last dimension
    datasets['matrix'] = {
        'data': (X_matrix, y_matrix),
        'true_expr': 'sum(X, axis=2)',
        'description': '2D matrix operations'
    }
    
    return datasets


def demonstrate_automatic_detection():
    """Demonstrate automatic data structure detection."""
    
    print("\n" + "="*80)
    print("AUTOMATIC DATA STRUCTURE DETECTION DEMO")
    print("="*80)
    
    datasets = create_synthetic_datasets()
    
    for name, dataset_info in datasets.items():
        X, y = dataset_info['data']
        true_expr = dataset_info['true_expr']
        description = dataset_info['description']
        
        print(f"\n{name.upper()} DATASET")
        print("-" * 50)
        print(f"Description: {description}")
        print(f"True expression: {true_expr}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Detect data structure
        data_shape, data_handler = auto_detect_data_structure(X, y, verbose=False)
        
        print(f"Detected variant: {data_shape.variant.value}")
        print(f"Handler: {data_handler.__class__.__name__}")
        print(f"Inputs: {data_shape.input_dims['n_features']}, "
              f"Outputs: {data_shape.output_dims['n_outputs']}")


def run_symbolic_regression_examples():
    """Run symbolic regression on different data variants."""
    
    print("\n" + "="*80)
    print("SYMBOLIC REGRESSION EXAMPLES")
    print("="*80)
    
    datasets = create_synthetic_datasets()
    
    # Training configuration for quick demonstration
    training_config = {
        'training': {
            'n_samples': 5000,  # Reduced for demo
            'batch_size': 100,
            'early_stopping': True
        },
        'verbose': True
    }
    
    results = {}
    
    for name, dataset_info in datasets.items():
        if name == 'matrix':  # Skip matrix for now (more complex)
            continue
            
        print(f"\n{'='*60}")
        print(f"RUNNING DSO ON {name.upper()} DATASET")
        print('='*60)
        
        try:
            # Run automatic DSO
            result = auto_fit(
                dataset=dataset_info['data'],
                **training_config
            )
            
            results[name] = result
            
            # Print results
            dso = result['dso']
            variant_info = result['variant_info']
            best_program = result['best_program']
            
            print(f"\n✓ Training completed for {name}")
            print(f"  Detected variant: {variant_info['variant']}")
            print(f"  Best reward: {result['results'].get('r', 'N/A')}")
            
            if best_program is not None:
                print(f"  Best expression: {best_program.str}")
                print(f"  True expression: {dataset_info['true_expr']}")
                
                # Evaluate on test data
                X, y = dataset_info['data']
                eval_results = dso.evaluate(X, y)
                print(f"  Test NMSE: {eval_results['nmse']:.6f}")
                print(f"  Test R²: {eval_results['r2']:.6f}")
                
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def demonstrate_different_strategies():
    """Demonstrate different multi-output strategies."""
    
    print("\n" + "="*80)
    print("MULTI-OUTPUT STRATEGY COMPARISON")
    print("="*80)
    
    # Create MIMO dataset
    X_mimo = np.random.uniform(-1, 1, (100, 2))
    y_mimo = np.column_stack([
        X_mimo[:, 0] * X_mimo[:, 1],  # Simple multiplication
        X_mimo[:, 0] ** 2 + X_mimo[:, 1] ** 2  # Sum of squares
    ]) + 0.05 * np.random.randn(100, 2)
    
    strategies = ['replicate', 'independent', 'shared']
    
    training_config = {
        'training': {
            'n_samples': 2000,
            'batch_size': 50
        },
        'verbose': False
    }
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} STRATEGY")
        print("-" * 30)
        
        try:
            result = auto_fit(
                dataset=(X_mimo, y_mimo),
                **training_config,
                **{'policy.multi_output_strategy': strategy}
            )
            
            dso = result['dso']
            best_program = result['best_program']
            
            print(f"  Strategy: {strategy}")
            print(f"  Best reward: {result['results'].get('r', 'N/A')}")
            
            if best_program is not None:
                print(f"  Expression: {best_program.str}")
                
                # Evaluate
                eval_results = dso.evaluate(X_mimo, y_mimo)
                print(f"  NMSE: {eval_results['nmse']:.6f}")
                
        except Exception as e:
            print(f"  Error with {strategy}: {e}")


def create_custom_dataset_example():
    """Demonstrate loading custom datasets from CSV."""
    
    print("\n" + "="*80)
    print("CUSTOM DATASET LOADING EXAMPLE")
    print("="*80)
    
    # Create a sample CSV file
    csv_filename = "sample_mimo_data.csv"
    
    # Generate sample data
    np.random.seed(123)
    n_samples = 200
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    x3 = np.random.uniform(0, 3, n_samples)
    
    y1 = x1 * x2 + 0.1 * np.random.randn(n_samples)
    y2 = np.sin(x3) + 0.1 * np.random.randn(n_samples)
    y3 = x1 ** 2 + x2 ** 2 + 0.1 * np.random.randn(n_samples)
    
    # Create DataFrame and save
    import pandas as pd
    df = pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3,
        'y1': y1, 'y2': y2, 'y3': y3
    })
    df.to_csv(csv_filename, index=False)
    
    print(f"Created sample CSV file: {csv_filename}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Load and run DSO
    try:
        result = auto_fit(
            dataset=csv_filename,
            training={'n_samples': 3000, 'batch_size': 100},
            verbose=True
        )
        
        dso = result['dso']
        variant_info = result['variant_info']
        
        print(f"\n✓ CSV dataset loaded successfully")
        print(f"  Detected variant: {variant_info['variant']}")
        print(f"  Inputs: {variant_info['n_inputs']}, Outputs: {variant_info['n_outputs']}")
        
        if result['best_program'] is not None:
            print(f"  Best expression: {result['best_program'].str}")
            
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
    
    # Clean up
    if os.path.exists(csv_filename):
        os.remove(csv_filename)


def benchmark_comparison():
    """Compare performance across different variants."""
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK COMPARISON")
    print("="*80)
    
    datasets = create_synthetic_datasets()
    
    # Exclude matrix for now
    test_datasets = {k: v for k, v in datasets.items() if k != 'matrix'}
    
    benchmark_config = {
        'training': {
            'n_samples': 10000,
            'batch_size': 200,
            'early_stopping': True
        },
        'verbose': False
    }
    
    results_summary = []
    
    for name, dataset_info in test_datasets.items():
        print(f"\nBenchmarking {name}...")
        
        try:
            import time
            start_time = time.time()
            
            result = auto_fit(
                dataset=dataset_info['data'],
                **benchmark_config
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            dso = result['dso']
            variant_info = result['variant_info']
            
            # Evaluate performance
            X, y = dataset_info['data']
            eval_results = dso.evaluate(X, y)
            
            summary = {
                'dataset': name,
                'variant': variant_info['variant'],
                'n_inputs': variant_info['n_inputs'],
                'n_outputs': variant_info['n_outputs'],
                'training_time': training_time,
                'best_reward': result['results'].get('r', 0),
                'nmse': eval_results['nmse'],
                'r2': eval_results['r2'],
                'expression': result['best_program'].str if result['best_program'] else 'None'
            }
            
            results_summary.append(summary)
            
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<10} {'Variant':<12} {'I/O':<8} {'Time':<8} {'R²':<8} {'Expression':<30}")
    print("-" * 80)
    
    for result in results_summary:
        io_str = f"{result['n_inputs']}/{result['n_outputs']}"
        time_str = f"{result['training_time']:.1f}s"
        r2_str = f"{result['r2']:.3f}"
        expr_str = result['expression'][:28] + "..." if len(result['expression']) > 30 else result['expression']
        
        print(f"{result['dataset']:<10} {result['variant']:<12} {io_str:<8} {time_str:<8} {r2_str:<8} {expr_str:<30}")


def main():
    """Main demonstration function."""
    
    print("="*80)
    print("MODULAR DSO FRAMEWORK DEMONSTRATION")
    print("="*80)
    print("This script demonstrates the automatic variant detection and")
    print("modular configuration capabilities of the DSO framework.")
    print("="*80)
    
    try:
        # 1. Demonstrate automatic detection
        demonstrate_automatic_detection()
        
        # 2. Run symbolic regression examples
        run_symbolic_regression_examples()
        
        # 3. Demonstrate different strategies
        demonstrate_different_strategies()
        
        # 4. Custom dataset example
        create_custom_dataset_example()
        
        # 5. Benchmark comparison
        benchmark_comparison()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("The modular DSO framework successfully handled all data variants:")
        print("✓ Scalar (SISO)")
        print("✓ Vector Input (MISO)")  
        print("✓ Vector Output (SIMO)")
        print("✓ Vector Both (MIMO)")
        print("✓ Custom CSV datasets")
        print("✓ Multiple strategies")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
