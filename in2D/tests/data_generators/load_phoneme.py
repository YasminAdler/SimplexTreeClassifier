import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import MinMaxScaler
    
    print("Fetching phoneme dataset from OpenML...")
    phoneme = fetch_openml(name='phoneme', version=1, as_frame=False)
    
    X = phoneme.data
    y = phoneme.target
    
    # Convert target to numeric (-1, 1) for SVM
    unique_classes = np.unique(y)
    print(f"Original classes: {unique_classes}")
    y_numeric = np.where(y == unique_classes[0], -1, 1).astype(float)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Class distribution: {np.unique(y_numeric, return_counts=True)}")
    print(f"Feature ranges (before normalization):")
    for i in range(X.shape[1]):
        print(f"  V{i+1}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]")
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print(f"\nData normalized to range [0, 1]")
    
    np.savez('../datasets/dataset_phoneme.npz', X=X, y=y_numeric)
    print(f"Saved to dataset_phoneme.npz")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTry: pip install scikit-learn")
