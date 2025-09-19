import numpy as np

def generate_simple_mimo_data(n_samples=500):
    """
    Generate very simple MIMO data that should be easy to discover:
    y1 = 2*x1 + x2
    y2 = x1 + 3*x2  
    y3 = x3
    y4 = x4 + x5
    y5 = 2*x6
    y6 = x1 + x6
    """
    
    # Generate random inputs
    np.random.seed(42)
    X = np.random.uniform(0, 1, (n_samples, 6))
    
    # Calculate outputs with simple linear relationships
    y = np.zeros((n_samples, 6))
    
    y[:, 0] = 2*X[:, 0] + X[:, 1]      # y1 = 2*x1 + x2
    y[:, 1] = X[:, 0] + 3*X[:, 1]      # y2 = x1 + 3*x2
    y[:, 2] = X[:, 2]                  # y3 = x3
    y[:, 3] = X[:, 3] + X[:, 4]        # y4 = x4 + x5
    y[:, 4] = 2*X[:, 5]                # y5 = 2*x6
    y[:, 5] = X[:, 0] + X[:, 5]        # y6 = x1 + x6
    
    return X, y

def load_simple_mimo():
    """Load function for DSO integration"""
    X_train, y_train = generate_simple_mimo_data(500)
    X_test, y_test = generate_simple_mimo_data(200)
    
    print("Generated simple MIMO dataset:")
    print(f"  y1 = 2*x1 + x2")
    print(f"  y2 = x1 + 3*x2") 
    print(f"  y3 = x3")
    print(f"  y4 = x4 + x5")
    print(f"  y5 = 2*x6")
    print(f"  y6 = x1 + x6")
    
    return {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}
    }

if __name__ == "__main__":
    # Test the data generation
    X, y = generate_simple_mimo_data(10)
    print("Sample input X:")
    print(X[:3])
    print("Sample output y:")
    print(y[:3])
    
    # Verify relationships
    print("\nVerifying relationships:")
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  2*{X[i,0]:.3f} + {X[i,1]:.3f} = {2*X[i,0] + X[i,1]:.3f}, y1={y[i,0]:.3f}")
        print(f"  {X[i,0]:.3f} + 3*{X[i,1]:.3f} = {X[i,0] + 3*X[i,1]:.3f}, y2={y[i,1]:.3f}")