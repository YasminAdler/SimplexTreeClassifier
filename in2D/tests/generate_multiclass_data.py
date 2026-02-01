import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler


def generate_multiclass_dataset(n_samples=1000, n_classes=3, dimensions=2, seed=None):
    """
    Generate a multiclass dataset with specified number of classes.
    
    Creates clusters of points in 2D space, each cluster representing a class.
    Points are generated around cluster centers with some noise.
    
    Args:
        n_samples: Total number of samples to generate
        n_classes: Number of classes (3, 4, 5, etc.)
        dimensions: Number of features (default 2)
        seed: Random seed for reproducibility
        
    Returns:
        X: Feature array of shape (n_samples, dimensions)
        y: Label array with values 0, 1, 2, ... (n_classes-1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples_per_class = n_samples // n_classes
    
    # Generate cluster centers evenly distributed in a circle
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    radius = 0.35  # Distance from center
    centers = np.column_stack([
        0.5 + radius * np.cos(angles),
        0.5 + radius * np.sin(angles)
    ])
    
    X_list = []
    y_list = []
    
    for class_idx in range(n_classes):
        center = centers[class_idx]
        # Generate points around this center with Gaussian noise
        noise_std = 0.12  # Controls cluster spread
        points = np.random.normal(center, noise_std, (samples_per_class, dimensions))
        X_list.append(points)
        y_list.append(np.full(samples_per_class, class_idx))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiclass 2D dataset')
    parser.add_argument('--classes', type=int, required=True, help='Number of classes (e.g., 3, 4, 5)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    X, y = generate_multiclass_dataset(
        n_samples=args.samples,
        n_classes=args.classes,
        seed=args.seed
    )
    
    # Normalize data to [0, 1] range
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Save to datasets folder
    filename = f'datasets/multiclass_{args.classes}classes_2d.npz'
    np.savez(filename, X=X, y=y)
    
    print(f"Saved {args.classes}-class dataset: {X.shape[0]} points, shape: {X.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Data normalized to range [0, 1]")
    print(f"Saved to: {filename}")
