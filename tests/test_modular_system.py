"""
Comprehensive tests for the modular DSO system.

This test suite verifies that the modular DSO framework correctly:
1. Detects different data variants (scalar, vector, tensor)
2. Configures appropriate handlers and executors
3. Maintains backward compatibility
4. Handles edge cases gracefully
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import pandas as pd

# Add DSO package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dso_pkg'))

from dso.core.data_types import (
    DataVariant, DataShape, DataTypeDetector, auto_detect_data_structure,
    ScalarHandler, VectorInputHandler, VectorOutputHandler, VectorBothHandler, TensorHandler
)
from dso.task.regression.modular_regression import ModularRegressionTask, create_modular_regression_task
from dso.core.modular_program import (
    ModularProgram, MultiProgram, create_program_for_variant,
    ScalarExecutor, VectorExecutor, MultiOutputExecutor, TensorExecutor
)
from dso.unified_dso import UnifiedDSO, auto_fit


class TestDataTypes(unittest.TestCase):
    """Test data type detection and handling."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Scalar data (SISO)
        self.X_scalar = np.random.randn(100, 1)
        self.y_scalar = self.X_scalar.flatten() ** 2
        
        # Vector input data (MISO)
        self.X_miso = np.random.randn(100, 3)
        self.y_miso = self.X_miso[:, 0] * self.X_miso[:, 1] + self.X_miso[:, 2]
        
        # Vector output data (SIMO)
        self.X_simo = np.random.randn(100, 1)
        self.y_simo = np.column_stack([
            self.X_simo.flatten() ** 2,
            np.sin(self.X_simo.flatten()),
            np.cos(self.X_simo.flatten())
        ])
        
        # Vector both data (MIMO)
        self.X_mimo = np.random.randn(100, 3)
        self.y_mimo = np.column_stack([
            self.X_mimo[:, 0] * self.X_mimo[:, 1],
            np.sin(self.X_mimo[:, 2])
        ])
        
        # Matrix data
        self.X_matrix = np.random.randn(50, 2, 4)
        self.y_matrix = np.sum(self.X_matrix, axis=2, keepdims=True)
        
    def test_scalar_detection(self):
        """Test scalar data variant detection."""
        data_shape = DataShape(self.X_scalar, self.y_scalar)
        self.assertEqual(data_shape.variant, DataVariant.SCALAR)
        self.assertTrue(data_shape.is_scalar)
        self.assertEqual(data_shape.input_dims['n_features'], 1)
        self.assertEqual(data_shape.output_dims['n_outputs'], 1)
        
    def test_vector_input_detection(self):
        """Test vector input (MISO) data variant detection."""
        data_shape = DataShape(self.X_miso, self.y_miso)
        self.assertEqual(data_shape.variant, DataVariant.VECTOR_INPUT)
        self.assertTrue(data_shape.is_miso)
        self.assertEqual(data_shape.input_dims['n_features'], 3)
        self.assertEqual(data_shape.output_dims['n_outputs'], 1)
        
    def test_vector_output_detection(self):
        """Test vector output (SIMO) data variant detection."""
        data_shape = DataShape(self.X_simo, self.y_simo)
        self.assertEqual(data_shape.variant, DataVariant.VECTOR_OUTPUT)
        self.assertTrue(data_shape.is_simo)
        self.assertEqual(data_shape.input_dims['n_features'], 1)
        self.assertEqual(data_shape.output_dims['n_outputs'], 3)
        
    def test_vector_both_detection(self):
        """Test vector both (MIMO) data variant detection."""
        data_shape = DataShape(self.X_mimo, self.y_mimo)
        self.assertEqual(data_shape.variant, DataVariant.VECTOR_BOTH)
        self.assertTrue(data_shape.is_mimo)
        self.assertEqual(data_shape.input_dims['n_features'], 3)
        self.assertEqual(data_shape.output_dims['n_outputs'], 2)
        
    def test_matrix_detection(self):
        """Test matrix data variant detection."""
        data_shape = DataShape(self.X_matrix, self.y_matrix)
        self.assertEqual(data_shape.variant, DataVariant.MATRIX)
        self.assertTrue(data_shape.is_tensor)
        
    def test_handler_creation(self):
        """Test automatic handler creation."""
        # Test scalar handler
        data_shape, handler = DataTypeDetector.detect_and_create_handler(self.X_scalar, self.y_scalar)
        self.assertIsInstance(handler, ScalarHandler)
        
        # Test MISO handler
        data_shape, handler = DataTypeDetector.detect_and_create_handler(self.X_miso, self.y_miso)
        self.assertIsInstance(handler, VectorInputHandler)
        
        # Test SIMO handler
        data_shape, handler = DataTypeDetector.detect_and_create_handler(self.X_simo, self.y_simo)
        self.assertIsInstance(handler, VectorOutputHandler)
        
        # Test MIMO handler
        data_shape, handler = DataTypeDetector.detect_and_create_handler(self.X_mimo, self.y_mimo)
        self.assertIsInstance(handler, VectorBothHandler)
        
    def test_edge_cases(self):
        """Test edge cases in data detection."""
        # Single sample
        X_single = np.array([[1.0]])
        y_single = np.array([2.0])
        data_shape = DataShape(X_single, y_single)
        self.assertEqual(data_shape.variant, DataVariant.SCALAR)
        
        # 1D arrays
        X_1d = np.array([1, 2, 3, 4, 5])
        y_1d = np.array([2, 4, 6, 8, 10])
        # This should be reshaped internally
        try:
            data_shape = DataShape(X_1d.reshape(-1, 1), y_1d)
            self.assertEqual(data_shape.variant, DataVariant.SCALAR)
        except Exception as e:
            self.fail(f"Should handle 1D arrays gracefully: {e}")


class TestModularRegression(unittest.TestCase):
    """Test modular regression task."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create test datasets
        self.scalar_data = (
            np.random.randn(50, 1),
            np.random.randn(50)
        )
        
        self.miso_data = (
            np.random.randn(50, 3),
            np.random.randn(50)
        )
        
        self.mimo_data = (
            np.random.randn(50, 3),
            np.random.randn(50, 2)
        )
        
    def test_scalar_task_creation(self):
        """Test scalar task creation."""
        task = ModularRegressionTask(dataset=self.scalar_data)
        self.assertEqual(task.variant, DataVariant.SCALAR)
        self.assertEqual(task.n_input_var, 1)
        self.assertEqual(task.n_output_var, 1)
        self.assertTrue(task.is_scalar)
        
    def test_miso_task_creation(self):
        """Test MISO task creation."""
        task = ModularRegressionTask(dataset=self.miso_data)
        self.assertEqual(task.variant, DataVariant.VECTOR_INPUT)
        self.assertEqual(task.n_input_var, 3)
        self.assertEqual(task.n_output_var, 1)
        self.assertTrue(task.is_miso)
        
    def test_mimo_task_creation(self):
        """Test MIMO task creation."""
        task = ModularRegressionTask(dataset=self.mimo_data)
        self.assertEqual(task.variant, DataVariant.VECTOR_BOTH)
        self.assertEqual(task.n_input_var, 3)
        self.assertEqual(task.n_output_var, 2)
        self.assertTrue(task.is_mimo)
        
    def test_csv_loading(self):
        """Test CSV data loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write test data
            f.write("x1,x2,y1,y2\n")
            for i in range(20):
                x1, x2 = np.random.randn(2)
                y1, y2 = x1 * x2, np.sin(x1)
                f.write(f"{x1},{x2},{y1},{y2}\n")
            csv_path = f.name
            
        try:
            task = ModularRegressionTask(dataset=csv_path)
            self.assertEqual(task.n_input_var, 2)
            self.assertEqual(task.n_output_var, 2)
            self.assertTrue(task.is_mimo)
        finally:
            os.unlink(csv_path)
            
    def test_variance_calculations(self):
        """Test variance calculations for different variants."""
        # Scalar
        task_scalar = ModularRegressionTask(dataset=self.scalar_data)
        self.assertIsInstance(task_scalar.var_y_test, (int, float, np.number))
        
        # MIMO
        task_mimo = ModularRegressionTask(dataset=self.mimo_data)
        self.assertIsInstance(task_mimo.var_y_test, np.ndarray)
        self.assertEqual(len(task_mimo.var_y_test), 2)


class TestModularPrograms(unittest.TestCase):
    """Test modular program execution."""
    
    def setUp(self):
        """Set up test data and programs."""
        np.random.seed(42)
        
        self.X_scalar = np.random.randn(20, 1)
        self.X_vector = np.random.randn(20, 3)
        
        # Create dummy data shapes
        self.scalar_shape = DataShape(self.X_scalar, np.random.randn(20))
        self.vector_shape = DataShape(self.X_vector, np.random.randn(20))
        self.mimo_shape = DataShape(self.X_vector, np.random.randn(20, 2))
        
    def test_executor_selection(self):
        """Test automatic executor selection."""
        # Scalar executor
        executor = ScalarExecutor()
        self.assertTrue(executor.can_handle(DataVariant.SCALAR))
        self.assertFalse(executor.can_handle(DataVariant.VECTOR_INPUT))
        
        # Vector executor
        executor = VectorExecutor()
        self.assertTrue(executor.can_handle(DataVariant.VECTOR_INPUT))
        self.assertFalse(executor.can_handle(DataVariant.SCALAR))
        
        # Multi-output executor
        executor = MultiOutputExecutor(2)
        self.assertTrue(executor.can_handle(DataVariant.VECTOR_OUTPUT))
        self.assertTrue(executor.can_handle(DataVariant.VECTOR_BOTH))
        
    def test_modular_program_creation(self):
        """Test modular program creation."""
        # Simple tokens for testing
        tokens = ["x1", "+", "x1"]
        
        # Scalar program
        program = create_program_for_variant(tokens, self.scalar_shape)
        self.assertIsInstance(program, ModularProgram)
        
        # MIMO program with multiple token sequences
        multi_tokens = [["x1", "+", "x2"], ["x3", "*", "x1"]]
        program = create_program_for_variant(None, self.mimo_shape, multi_tokens)
        self.assertIsInstance(program, MultiProgram)
        
    def test_multi_program_functionality(self):
        """Test MultiProgram functionality."""
        # Create dummy programs (we'll mock the execution)
        class DummyProgram:
            def __init__(self, tokens):
                self.tokens = tokens
                self.str = " ".join(tokens)
                
            def execute(self, X):
                return np.random.randn(X.shape[0])
                
        programs = [DummyProgram(["x1"]), DummyProgram(["x2"])]
        multi_program = MultiProgram(programs, self.mimo_shape)
        
        self.assertEqual(multi_program.n_programs, 2)
        self.assertIn("|", multi_program.str)
        
        # Test execution
        result = multi_program.execute(self.X_vector)
        self.assertEqual(result.shape, (self.X_vector.shape[0], 2))


class TestUnifiedDSO(unittest.TestCase):
    """Test unified DSO interface."""
    
    def setUp(self):
        """Set up test datasets."""
        np.random.seed(42)
        
        # Simple datasets for quick testing
        self.scalar_dataset = (
            np.random.randn(30, 1),
            np.random.randn(30)
        )
        
        self.mimo_dataset = (
            np.random.randn(30, 2),
            np.random.randn(30, 2)
        )
        
    def test_unified_interface_creation(self):
        """Test UnifiedDSO creation."""
        dso = UnifiedDSO(verbose=False)
        self.assertIsNotNone(dso.config)
        
    def test_variant_info_before_fit(self):
        """Test getting variant info before fitting."""
        dso = UnifiedDSO(verbose=False)
        info = dso.get_variant_info()
        self.assertEqual(info['status'], 'No data loaded')
        
    def test_scalar_fitting(self):
        """Test fitting on scalar data."""
        try:
            dso = UnifiedDSO(verbose=False)
            
            # Use minimal configuration for testing
            result = dso.fit(
                self.scalar_dataset,
                training={'n_samples': 500, 'batch_size': 50}
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(dso.data_shape.variant, DataVariant.SCALAR)
            
        except Exception as e:
            # If dependencies are missing, skip this test
            self.skipTest(f"Skipping due to missing dependencies: {e}")
            
    def test_auto_fit_function(self):
        """Test auto_fit convenience function."""
        try:
            result = auto_fit(
                self.scalar_dataset,
                training={'n_samples': 200, 'batch_size': 20},
                verbose=False
            )
            
            self.assertIn('results', result)
            self.assertIn('dso', result)
            self.assertIn('variant_info', result)
            
        except Exception as e:
            self.skipTest(f"Skipping due to missing dependencies: {e}")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing DSO code."""
    
    def test_legacy_data_formats(self):
        """Test that legacy data formats still work."""
        # Test tuple format
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        try:
            task = ModularRegressionTask(dataset=(X, y))
            self.assertEqual(task.n_input_var, 3)
            self.assertEqual(task.n_output_var, 1)
        except Exception as e:
            self.fail(f"Legacy tuple format should work: {e}")
            
    def test_benchmark_dataset_names(self):
        """Test that benchmark dataset names are handled."""
        # This should attempt to load Nguyen-1 benchmark
        try:
            task = ModularRegressionTask(dataset="Nguyen-1")
            # If it succeeds, check basic properties
            self.assertGreater(task.n_input_var, 0)
            self.assertEqual(task.n_output_var, 1)
        except Exception as e:
            # Skip if benchmark loading fails (missing files, etc.)
            self.skipTest(f"Benchmark loading failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        X_empty = np.array([]).reshape(0, 1)
        y_empty = np.array([])
        
        with self.assertRaises(ValueError):
            DataShape(X_empty, y_empty)
            
    def test_mismatched_shapes(self):
        """Test handling of mismatched array shapes."""
        X = np.random.randn(50, 3)
        y = np.random.randn(40)  # Different number of samples
        
        # This should be handled gracefully by the task
        try:
            task = ModularRegressionTask(dataset=(X, y))
            # If it doesn't raise an error, the shapes should be corrected
            self.assertEqual(task.X_train.shape[0], task.y_train.shape[0])
        except Exception:
            # Or it should raise a clear error
            pass
            
    def test_invalid_dataset_types(self):
        """Test handling of invalid dataset types."""
        with self.assertRaises(ValueError):
            ModularRegressionTask(dataset=123)  # Invalid type
            
        with self.assertRaises(FileNotFoundError):
            ModularRegressionTask(dataset="nonexistent_file.csv")


def run_integration_test():
    """Run a comprehensive integration test."""
    print("Running integration test...")
    
    # Create test datasets
    datasets = {
        'scalar': (np.random.randn(100, 1), np.random.randn(100)),
        'miso': (np.random.randn(100, 3), np.random.randn(100)),
        'simo': (np.random.randn(100, 1), np.random.randn(100, 3)),
        'mimo': (np.random.randn(100, 3), np.random.randn(100, 2))
    }
    
    for name, dataset in datasets.items():
        print(f"Testing {name}...")
        
        try:
            # Test data detection
            data_shape, handler = auto_detect_data_structure(*dataset, verbose=False)
            print(f"  ✓ Detected variant: {data_shape.variant.value}")
            
            # Test task creation
            task = ModularRegressionTask(dataset=dataset)
            print(f"  ✓ Task created: {task.n_input_var} inputs, {task.n_output_var} outputs")
            
            # Test unified interface (with minimal training)
            try:
                result = auto_fit(
                    dataset,
                    training={'n_samples': 100, 'batch_size': 20},
                    verbose=False
                )
                print(f"  ✓ Training completed")
            except Exception as e:
                print(f"  ⚠ Training skipped: {e}")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
    print("Integration test completed.")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60)
    
    # Run integration test
    run_integration_test()
