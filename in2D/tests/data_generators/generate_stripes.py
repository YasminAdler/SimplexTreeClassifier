import numpy as np

def generate_stripes_dataset(n_points_per_stripe=200, n_stripes=4, margin=0.02, seed=42):
    """
    Generate a dataset with horizontal STRIPES of alternating classes.
    
    This forces the classifier to create zone boundaries.
    
    Layout (in [0,1] x [0,1] space with 4 stripes):
    
        y=1.0 ┌──────────────────┐
              │   BLUE (-1)      │  stripe 3
              ├ - - - GAP - - - -┤
              │   RED (+1)       │  stripe 2
              ├ - - - GAP - - - -┤
              │   BLUE (-1)      │  stripe 1
              ├ - - - GAP - - - -┤
              │   RED (+1)       │  stripe 0
        y=0.0 └──────────────────┘
              x=0              x=1
    
    Args:
        n_points_per_stripe: Number of points per stripe
        n_stripes: Number of horizontal stripes (alternating classes)
        margin: Gap between stripes (space with no data points).
                Valid range: 0.01 to 0.10
                - 0.02 = default (small gap)
                - 0.05 = medium gap
                - 0.10 = large gap
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp margin to valid range
    original_margin = margin
    max_margin = 0.5 / n_stripes - 0.01  # Max so stripes don't overlap
    margin = max(0.01, min(max_margin, margin))
    if margin != original_margin:
        print(f"WARNING: margin={original_margin} out of range, clamped to {margin:.3f}")
    
    stripe_height = 1.0 / n_stripes
    
    X_list = []
    y_list = []
    
    for i in range(n_stripes):
        # Y range for this stripe (with margin gap at top and bottom)
        y_min = i * stripe_height
        y_max = (i + 1) * stripe_height
        
        # Generate random points in this stripe, with margin gap from edges
        stripe_x = np.random.uniform(0.02, 0.98, n_points_per_stripe)
        stripe_y = np.random.uniform(y_min + margin, y_max - margin, n_points_per_stripe)
        stripe_points = np.column_stack([stripe_x, stripe_y])
        
        # Alternating classes: even stripes = +1 (red), odd stripes = -1 (blue)
        if i % 2 == 0:
            stripe_labels = np.ones(n_points_per_stripe)  # Red
        else:
            stripe_labels = -np.ones(n_points_per_stripe)  # Blue
        
        X_list.append(stripe_points)
        y_list.append(stripe_labels)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


def generate_vertical_stripes_dataset(n_points_per_stripe=200, n_stripes=4, seed=42):
    """
    Generate a dataset with VERTICAL stripes of alternating classes.
    
    Layout (in [0,1] x [0,1] space with 4 stripes):
    
        y=1.0 ┌────┬────┬────┬────┐
              │RED │BLUE│RED │BLUE│
              │ +1 │ -1 │ +1 │ -1 │
        y=0.0 └────┴────┴────┴────┘
             x=0  .25  .5  .75   1
    
    Args:
        n_points_per_stripe: Number of points per stripe
        n_stripes: Number of vertical stripes (alternating classes)
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    stripe_width = 1.0 / n_stripes
    
    X_list = []
    y_list = []
    
    for i in range(n_stripes):
        # X range for this stripe
        x_min = i * stripe_width
        x_max = (i + 1) * stripe_width
        
        # Generate random points in this stripe
        stripe_x = np.random.uniform(x_min + 0.02, x_max - 0.02, n_points_per_stripe)
        stripe_y = np.random.uniform(0.02, 0.98, n_points_per_stripe)
        stripe_points = np.column_stack([stripe_x, stripe_y])
        
        # Alternating classes: even stripes = +1 (red), odd stripes = -1 (blue)
        if i % 2 == 0:
            stripe_labels = np.ones(n_points_per_stripe)  # Red
        else:
            stripe_labels = -np.ones(n_points_per_stripe)  # Blue
        
        X_list.append(stripe_points)
        y_list.append(stripe_labels)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


if __name__ == "__main__":
    # Set margin - gap between stripes (where no data points exist)
    # Valid range depends on n_stripes:
    #   0.02 = default (small gap)
    #   0.05 = medium gap  
    #   0.10 = large gap
    MARGIN = 0.08  # <-- CHANGE THIS VALUE
    N_STRIPES = 4
    
    # Generate horizontal stripes
    X, y = generate_stripes_dataset(
        n_points_per_stripe=200,
        n_stripes=N_STRIPES,
        margin=MARGIN,
        seed=42
    )
    
    np.savez('../datasets/dataset_stripes.npz', X=X, y=y)
    
    stripe_height = 1.0 / N_STRIPES
    
    print(f"")
    print(f"=" * 50)
    print(f"Saved horizontal stripes dataset: {X.shape[0]} points")
    print(f"Margin: {MARGIN} (gap between stripes)")
    print(f"=" * 50)
    print(f"")
    print(f"Layout ({N_STRIPES} horizontal stripes):")
    for i in range(N_STRIPES):
        y_min = i * stripe_height
        y_max = (i + 1) * stripe_height
        class_name = "RED (+1)" if i % 2 == 0 else "BLUE (-1)"
        data_y_min = y_min + MARGIN
        data_y_max = y_max - MARGIN
        print(f"  Stripe {i}: y=[{data_y_min:.2f}, {data_y_max:.2f}] {class_name}")
        if i < N_STRIPES - 1:
            gap = 2 * MARGIN
            print(f"       GAP: {gap:.2f} (no data)")
    print(f"")
    print(f"Gap between stripes: {2*MARGIN:.2f}")
    print(f"")
    print(f"Try different MARGIN values: 0.02, 0.05, 0.08, 0.10")
