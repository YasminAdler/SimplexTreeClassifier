import numpy as np

def generate_two_clusters_dataset(n_cluster1=200, n_cluster2=200, n_background=0, margin=0.15, 
                                   same_class=False, seed=42):
    """
    Generate a dataset with TWO separate CIRCULAR clusters.
    
    Layout:
    
        ┌─────────────────────────────┐
        │     ○○○            ●●●      │
        │    ○○○○○          ●●●●●     │
        │   ○ C1 ○    gap  ● C2 ●    │  ← Two circular clusters
        │    ○○○○○          ●●●●●     │
        │     ○○○            ●●●      │
        └─────────────────────────────┘
    
    Modes:
    - same_class=False: Cluster1=RED(+1), Cluster2=BLUE(-1) → 2 classes, no background
    - same_class=True:  Both clusters=RED(+1), Background=BLUE(-1) → 2 classes
    
    Args:
        n_cluster1: Number of points in first cluster
        n_cluster2: Number of points in second cluster  
        n_background: Number of background points (only used if same_class=True)
        margin: Gap between clusters. Valid range: 0.05 to 0.40
        same_class: If False, each cluster is a different class (recommended for 2-class)
                    If True, both clusters are same class, background is other class
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp margin to valid range
    original_margin = margin
    margin = max(0.05, min(0.40, margin))
    if margin != original_margin:
        print(f"WARNING: margin={original_margin} out of range, clamped to {margin}")
    
    # Cluster parameters
    cluster_radius = 0.08  # Standard deviation of Gaussian cluster
    
    # Cluster 1 center: upper-left area
    c1_x, c1_y = 0.25, 0.65
    
    # Cluster 2 center: lower-right area (separated by margin)
    c2_x, c2_y = 0.75, 0.35
    
    # Adjust centers based on margin (larger margin = further apart)
    offset = margin * 0.5
    c1_x -= offset * 0.5
    c1_y += offset * 0.3
    c2_x += offset * 0.5
    c2_y -= offset * 0.3
    
    # Generate circular clusters (Gaussian distribution)
    cluster1 = np.random.normal([c1_x, c1_y], cluster_radius, (n_cluster1, 2))
    cluster2 = np.random.normal([c2_x, c2_y], cluster_radius, (n_cluster2, 2))
    
    # Generate background points (scattered, avoiding cluster centers)
    background_list = []
    attempts = 0
    max_attempts = n_background * 10
    
    # Minimum distance from cluster centers to be considered background
    exclusion_radius = cluster_radius * 3 + margin * 0.3
    
    while len(background_list) < n_background and attempts < max_attempts:
        # Random point in [0.05, 0.95] x [0.05, 0.95]
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        
        # Check distance from both cluster centers
        dist1 = np.sqrt((x - c1_x)**2 + (y - c1_y)**2)
        dist2 = np.sqrt((x - c2_x)**2 + (y - c2_y)**2)
        
        # Only add if far enough from both clusters
        if dist1 > exclusion_radius and dist2 > exclusion_radius:
            background_list.append([x, y])
        
        attempts += 1
    
    background = np.array(background_list) if background_list else np.zeros((0, 2))
    
    # Combine all data and assign labels based on mode
    if same_class:
        # Both clusters are same class (+1), background is other class (-1)
        X = np.vstack([cluster1, cluster2, background]) if len(background) > 0 else np.vstack([cluster1, cluster2])
        y = np.concatenate([
            np.ones(n_cluster1),       # Red cluster 1: class +1
            np.ones(n_cluster2),       # Red cluster 2: class +1
            -np.ones(len(background))  # Blue background: class -1
        ]) if len(background) > 0 else np.concatenate([
            np.ones(n_cluster1),
            np.ones(n_cluster2)
        ])
    else:
        # Each cluster is a different class (NO background needed)
        X = np.vstack([cluster1, cluster2])
        y = np.concatenate([
            np.ones(n_cluster1),        # Cluster 1: class +1 (RED)
            -np.ones(n_cluster2)        # Cluster 2: class -1 (BLUE)
        ])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == "__main__":
    # Configuration
    N_CLUSTER = 200      # Points per cluster
    MARGIN = 0.25        # Gap between clusters (0.05 to 0.40)
    SAME_CLASS = False   # False = each cluster different class (no background)
                         # True = both clusters same class + background
    
    X, y = generate_two_clusters_dataset(
        n_cluster1=N_CLUSTER,
        n_cluster2=N_CLUSTER,
        n_background=300 if SAME_CLASS else 0,
        margin=MARGIN,
        same_class=SAME_CLASS,
        seed=42
    )
    
    np.savez('../datasets/dataset_two_clusters.npz', X=X, y=y)
    
    n_red = int((y == 1).sum())
    n_blue = int((y == -1).sum())
    
    print(f"")
    print(f"=" * 50)
    print(f"TWO CIRCULAR CLUSTERS DATASET")
    print(f"=" * 50)
    print(f"")
    print(f"Total points: {X.shape[0]}")
    print(f"  RED (class +1): {n_red} points")
    print(f"  BLUE (class -1): {n_blue} points")
    print(f"")
    print(f"Margin (cluster separation): {MARGIN}")
    print(f"Mode: {'Both clusters same class + background' if SAME_CLASS else 'Each cluster = different class'}")
    print(f"")
    print(f"Layout:")
    if SAME_CLASS:
        print(f"  - Two RED clusters (both class +1)")
        print(f"  - BLUE background scattered around")
    else:
        print(f"  - Cluster 1: RED (class +1)")
        print(f"  - Cluster 2: BLUE (class -1)")
        print(f"  - NO background points between them")
    print(f"")
    print(f"Saved to: ../datasets/dataset_two_clusters.npz")
