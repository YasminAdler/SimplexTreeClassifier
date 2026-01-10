import numpy as np

def generate_dataset(n_samples=1000, gamma=0.01, dimensions=2, num_hyperplanes=8, seed=None):
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
    
    X = (X[y != 0] + [1, 1]) / 2
    y = y[y != 0]
    
    return X, y


if __name__ == "__main__":
    X, y = generate_dataset(n_samples=1000, gamma=0.01, dimensions=2, num_hyperplanes=8, seed=42)
    np.savez('dataset.npz', X=X, y=y)
    print(f"Saved dataset: {X.shape[0]} points, Classes: {np.unique(y)}")

