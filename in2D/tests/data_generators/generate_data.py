import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_dataset(n_samples=1000, gamma=0.01, margin=0.10, dimensions=2, num_hyperplanes=8, seed=None):
    """
    Generate a dataset using random hyperplanes.
    
    Args:
        n_samples: Number of samples to generate
        gamma: Base offset for hyperplane thresholds
        margin: Gap/exclusion zone around the decision boundary.
                Points in the margin zone are removed (labeled 0).
                Valid range: 0.01 to 0.30
                - 0.05 = small gap
                - 0.10 = default gap
                - 0.20 = wide gap
                - 0.30 = very wide gap (more points removed)
        dimensions: Number of dimensions (default 2)
        num_hyperplanes: Number of random hyperplanes (default 8)
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_valid, dimensions)
        y: Labels array of shape (n_valid,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp margin to valid range
    original_margin = margin
    margin = max(0.01, min(0.40, margin))
    if margin != original_margin:
        print(f"WARNING: margin={original_margin} out of range, clamped to {margin}")
    
    X = np.random.normal(0, 1, (n_samples, dimensions))
    ws = np.random.normal(0, 1, (num_hyperplanes, dimensions))
    ws = ws / np.linalg.norm(ws, axis=1).reshape(-1, 1)
    
    u = np.random.uniform(0, 1, (n_samples, 1))
    X = X / np.linalg.norm(X, axis=1).reshape(-1, 1) * (u ** (1/dimensions))
    
    # Thresholds based on margin: larger margin = bigger gap between classes
    threshold_positive = 0.5 + gamma
    threshold_margin = threshold_positive - margin  # Points in this range are excluded
    
    y = np.ones(n_samples)
    for index in range(n_samples):
        z = 1
        for w in ws:
            dist = X[index].dot(w)
            if dist > threshold_positive:
                z = -1
            elif dist > threshold_margin:
                z = 0  # In margin zone, will be removed
        y[index] = z
    
    X = (X[y != 0] + [1, 1]) / 2
    y = y[y != 0]
    
    return X, y


if __name__ == "__main__":
    # === CONFIGURATION ===
    N_SAMPLES = 1000
    GAMMA = 0.01
    MARGIN = 0.10  # Gap around decision boundary (try 0.05, 0.10, 0.20, 0.30)
    SEED = 42
    
    X, y = generate_dataset(
        n_samples=N_SAMPLES,
        gamma=GAMMA,
        margin=MARGIN,
        dimensions=2,
        num_hyperplanes=8,
        seed=SEED
    )
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    np.savez('../datasets/dataset.npz', X=X, y=y)
    print(f"Saved dataset: {X.shape[0]} points, Classes: {np.unique(y)}")
    print(f"Data normalized to range [0, 1]")
    print(f"")
    print(f"=== CONFIGURATION ===")
    print(f"Margin: {MARGIN} (gap around decision boundary)")
    print(f"Gamma: {GAMMA}")
    print(f"Class distribution: +1={int((y == 1).sum())}, -1={int((y == -1).sum())}")

