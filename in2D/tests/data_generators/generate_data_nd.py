import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

def generate_dataset_nd(n_samples=1000, dimensions=3, gamma=0.01, margin=0.10, num_hyperplanes=8, seed=None):
    """
    Generate an N-dimensional dataset using random hyperplanes.
    
    Args:
        n_samples: Number of samples to generate
        dimensions: Number of dimensions
        gamma: Base offset for hyperplane thresholds
        margin: Gap/exclusion zone around the decision boundary.
                Points in the margin zone are removed (labeled 0).
                Valid range: 0.01 to 0.30
                - 0.05 = small gap
                - 0.10 = default gap
                - 0.20 = wide gap
                - 0.30 = very wide gap (more points removed)
        num_hyperplanes: Number of random hyperplanes
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
    
    offset = np.ones(dimensions)
    X = (X[y != 0] + offset) / 2
    y = y[y != 0]
    
    return X, y


def create_simplex_vertices(dimensions):
    vertices = []
    vertices.append(tuple([0.0] * dimensions))
    for i in range(dimensions):
        vertex = [0.0] * dimensions
        vertex[i] = 2.0
        vertices.append(tuple(vertex))
    return vertices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ND dataset for simplex tree testing')
    parser.add_argument('--dim', type=int, required=True, help='Number of dimensions (e.g., 3, 4, 5)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--gamma', type=float, default=0.01, help='Gamma parameter (default: 0.01)')
    parser.add_argument('--margin', type=float, default=0.10, help='Gap around decision boundary (default: 0.10)')
    parser.add_argument('--hyperplanes', type=int, default=8, help='Number of hyperplanes (default: 8)')
    
    args = parser.parse_args()
    
    X, y = generate_dataset_nd(
        n_samples=args.samples,
        dimensions=args.dim,
        gamma=args.gamma,
        margin=args.margin,
        num_hyperplanes=args.hyperplanes,
        seed=args.seed
    )
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    filename = f'../datasets/dataset_{args.dim}d.npz'
    np.savez(filename, X=X, y=y)
    print(f"Saved {args.dim}D dataset: {X.shape[0]} points, shape: {X.shape}, Classes: {np.unique(y)}")
    print(f"Data normalized to range [0, 1]")
    print(f"Margin: {args.margin} (gap around decision boundary)")

