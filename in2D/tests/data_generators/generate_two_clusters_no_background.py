import numpy as np

def generate_two_clusters_no_background(n_cluster1=200, n_cluster2=200, margin=0.25, 
                                         cluster_radius=0.08, seed=42):
    """
    Generate a dataset with TWO separate CIRCULAR clusters (same class),
    with NO background data - just empty space around them.
    
    Layout:
    
        ┌─────────────────────────────┐
        │                             │
        │     ○○○            ○○○      │
        │    ○○○○○          ○○○○○     │
        │   ○ RED ○        ○ RED ○    │  ← Two circular RED clusters
        │    ○○○○○          ○○○○○     │     (both class +1)
        │     ○○○            ○○○      │
        │                             │  ← Empty space (no data)
        │                             │
        └─────────────────────────────┘
    
    This is useful for:
    - OneClassSVM (anomaly detection)
    - Visualizing cluster shapes without background interference
    
    Args:
        n_cluster1: Number of points in first cluster
        n_cluster2: Number of points in second cluster
        margin: Distance between clusters (0.10 to 0.50)
        cluster_radius: Size of each cluster (standard deviation)
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,) - all +1
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp parameters
    margin = max(0.10, min(0.50, margin))
    cluster_radius = max(0.04, min(0.15, cluster_radius))
    
    # Cluster centers - spread apart based on margin
    # Upper-left and lower-right diagonal
    c1_x = 0.30 - margin * 0.3
    c1_y = 0.65 + margin * 0.2
    
    c2_x = 0.70 + margin * 0.3
    c2_y = 0.35 - margin * 0.2
    
    # Generate circular clusters (Gaussian distribution)
    cluster1 = np.random.normal([c1_x, c1_y], cluster_radius, (n_cluster1, 2))
    cluster2 = np.random.normal([c2_x, c2_y], cluster_radius, (n_cluster2, 2))
    
    # Combine clusters
    X = np.vstack([cluster1, cluster2])
    
    # All points are class +1 (RED)
    y = np.ones(len(X))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == "__main__":
    # Configuration
    N_CLUSTER = 200       # Points per cluster
    MARGIN = 0.30         # Distance between clusters (0.10 to 0.50)
    CLUSTER_RADIUS = 0.08 # Size of each cluster
    
    X, y = generate_two_clusters_no_background(
        n_cluster1=N_CLUSTER,
        n_cluster2=N_CLUSTER,
        margin=MARGIN,
        cluster_radius=CLUSTER_RADIUS,
        seed=42
    )
    
    np.savez('../datasets/dataset_two_clusters_no_background.npz', X=X, y=y)
    
    print(f"")
    print(f"=" * 50)
    print(f"TWO CLUSTERS - NO BACKGROUND")
    print(f"=" * 50)
    print(f"")
    print(f"Total points: {X.shape[0]}")
    print(f"All points are class +1 (RED)")
    print(f"")
    print(f"Configuration:")
    print(f"  Margin: {MARGIN}")
    print(f"  Cluster radius: {CLUSTER_RADIUS}")
    print(f"")
    print(f"Layout:")
    print(f"  - Two CIRCULAR red clusters")
    print(f"  - NO background data points")
    print(f"  - Empty space between and around clusters")
    print(f"")
    print(f"Saved to: ../datasets/dataset_two_clusters_no_background.npz")
