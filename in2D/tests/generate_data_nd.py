import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

def generate_dataset_nd(n_samples=1000, dimensions=3, gamma=0.01, num_hyperplanes=8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(0, 1, (n_samples, dimensions))
    ws = np.random.normal(0, 1, (num_hyperplanes, dimensions))
    ws = ws / np.linalg.norm(ws, axis=1).reshape(-1, 1)
    
    u = np.random.uniform(0, 1, (n_samples, 1))
    X = X / np.linalg.norm(X, axis=1).reshape(-1, 1) * (u ** (1/dimensions))
    
    y = np.ones(n_samples)
    for index in range(n_samples):
        z = 1
        for w in ws:
            if (X[index].dot(w) - 0.5 - gamma) > 0:
                z = -1
            elif (X[index].dot(w) - 0.4 - gamma) > 0:
                z = 0
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
    parser.add_argument('--hyperplanes', type=int, default=8, help='Number of hyperplanes (default: 8)')
    
    args = parser.parse_args()
    
    X, y = generate_dataset_nd(
        n_samples=args.samples,
        dimensions=args.dim,
        gamma=args.gamma,
        num_hyperplanes=args.hyperplanes,
        seed=args.seed
    )
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    filename = f'in2D/tests/datasets/dataset_{args.dim}d.npz'
    np.savez(filename, X=X, y=y)
    print(f"Saved {args.dim}D dataset: {X.shape[0]} points, shape: {X.shape}, Classes: {np.unique(y)}")
    print(f"Data normalized to range [0, 1]")

