import numpy as np

def generate_circular_clusters_dataset(n_clusters=3, n_points_per_cluster=200, n_background=400, 
                                        margin=0.15, cluster_radius=0.08, seed=42):
    """
    Generate a dataset with multiple CIRCULAR clusters, each a different class,
    plus optional background points as another class.
    
    Layout example (3 clusters):
    
        ┌─────────────────────────────┐
        │         ○○○                 │
        │        ○○○○○                │
        │       ○ C0 ○     ●●●        │  ← Cluster 0 (class 0)
        │        ○○○○○    ●●●●●       │
        │         ○○○    ● C1 ●       │  ← Cluster 1 (class 1)
        │                 ●●●●●       │
        │    ▲▲▲          ●●●        │
        │   ▲▲▲▲▲                     │
        │  ▲ C2 ▲   · · · · · · ·    │  ← Cluster 2 (class 2)
        │   ▲▲▲▲▲    background       │  ← Background (class n_clusters)
        └─────────────────────────────┘
    
    Args:
        n_clusters: Number of circular clusters (each is a different class: 0, 1, 2, ...)
        n_points_per_cluster: Number of points per cluster
        n_background: Number of background points (class = n_clusters). Set to 0 for no background.
        margin: Gap between clusters. Valid range: 0.05 to 0.40
        cluster_radius: Standard deviation of Gaussian clusters (0.05-0.15)
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,) with values 0, 1, ..., n_clusters (or n_clusters-1 if no background)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp parameters
    margin = max(0.05, min(0.40, margin))
    cluster_radius = max(0.03, min(0.15, cluster_radius))
    
    # Generate cluster centers spread around the space
    # Use a pattern that keeps clusters well-separated
    centers = _generate_cluster_centers(n_clusters, margin)
    
    X_list = []
    y_list = []
    
    # Generate each cluster
    for i in range(n_clusters):
        cluster = np.random.normal(centers[i], cluster_radius, (n_points_per_cluster, 2))
        X_list.append(cluster)
        y_list.append(np.full(n_points_per_cluster, i))  # Class = cluster index
    
    # Generate background points (if requested)
    if n_background > 0:
        background = _generate_background(n_background, centers, cluster_radius, margin)
        X_list.append(background)
        y_list.append(np.full(len(background), n_clusters))  # Background class = n_clusters
    
    # Combine all data
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def _generate_cluster_centers(n_clusters, margin):
    """Generate well-separated cluster centers."""
    
    if n_clusters == 2:
        # Two clusters: upper-left and lower-right
        return np.array([
            [0.30 - margin * 0.3, 0.65 + margin * 0.2],
            [0.70 + margin * 0.3, 0.35 - margin * 0.2]
        ])
    
    elif n_clusters == 3:
        # Three clusters: triangle pattern
        return np.array([
            [0.25, 0.70],  # Top-left
            [0.75, 0.70],  # Top-right
            [0.50, 0.25]   # Bottom-center
        ])
    
    elif n_clusters == 4:
        # Four clusters: corners
        return np.array([
            [0.25, 0.75],  # Top-left
            [0.75, 0.75],  # Top-right
            [0.25, 0.25],  # Bottom-left
            [0.75, 0.25]   # Bottom-right
        ])
    
    else:
        # N clusters: arrange in a circle
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        radius = 0.30
        centers = np.column_stack([
            0.5 + radius * np.cos(angles),
            0.5 + radius * np.sin(angles)
        ])
        return centers


def _generate_background(n_points, centers, cluster_radius, margin):
    """Generate background points avoiding cluster regions."""
    background_list = []
    exclusion_radius = cluster_radius * 3 + margin * 0.3
    max_attempts = n_points * 20
    attempts = 0
    
    while len(background_list) < n_points and attempts < max_attempts:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        
        # Check distance from all cluster centers
        far_enough = True
        for center in centers:
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < exclusion_radius:
                far_enough = False
                break
        
        if far_enough:
            background_list.append([x, y])
        
        attempts += 1
    
    return np.array(background_list) if background_list else np.zeros((0, 2))


if __name__ == "__main__":
    # === CONFIGURATION ===
    N_CLUSTERS = 3           # Number of circular clusters (each a different class)
    N_POINTS = 200           # Points per cluster
    N_BACKGROUND = 400       # Background points (set to 0 for no background)
    MARGIN = 0.15            # Gap between clusters (0.05 to 0.40)
    CLUSTER_RADIUS = 0.08    # Size of each cluster (0.05 to 0.15)
    
    X, y = generate_circular_clusters_dataset(
        n_clusters=N_CLUSTERS,
        n_points_per_cluster=N_POINTS,
        n_background=N_BACKGROUND,
        margin=MARGIN,
        cluster_radius=CLUSTER_RADIUS,
        seed=42
    )
    
    # Save with descriptive filename
    if N_BACKGROUND > 0:
        filename = f'../datasets/dataset_{N_CLUSTERS}clusters_with_background.npz'
    else:
        filename = f'../datasets/dataset_{N_CLUSTERS}clusters.npz'
    
    np.savez(filename, X=X, y=y)
    
    # Print info
    print(f"")
    print(f"=" * 50)
    print(f"CIRCULAR CLUSTERS DATASET")
    print(f"=" * 50)
    print(f"")
    print(f"Total points: {X.shape[0]}")
    print(f"Number of clusters: {N_CLUSTERS}")
    print(f"")
    print(f"Class distribution:")
    for i in range(N_CLUSTERS):
        count = int((y == i).sum())
        print(f"  Class {i} (cluster {i+1}): {count} points")
    if N_BACKGROUND > 0:
        count = int((y == N_CLUSTERS).sum())
        print(f"  Class {N_CLUSTERS} (background): {count} points")
    print(f"")
    print(f"Configuration:")
    print(f"  Margin: {MARGIN}")
    print(f"  Cluster radius: {CLUSTER_RADIUS}")
    print(f"")
    print(f"Saved to: {filename}")
