import numpy as np

def generate_triangle_disconnected_dataset(n_points_per_region=150, margin=0.05, seed=42):
    """
    Generate a dataset specifically designed for the TRIANGULAR root simplex
    that forces DISCONNECTED regions.
    
    Root simplex vertices: (0,0), (2,0), (0,2)
    Data normalized to [0,1] maps to lower-left portion.
    
    Layout for TRIANGULAR simplex (after normalization):
    
              (0,1)
               /\
              /  \
             / B  \      B = BLUE (-1) at top corner
            /      \
           /--------\   
          /  R    R  \   R = RED (+1) at two bottom corners  
         /____________\
      (0,0)         (1,0)
    
    AND we add BLUE in the CENTER to separate the two RED corners!
    
    This creates:
    - TWO disconnected RED regions (bottom-left and bottom-right corners)
    - TWO disconnected BLUE regions (top corner and center band)
    
    Args:
        n_points_per_region: Number of points per region
        margin: Gap between regions (space with no data).
                Valid range: 0.02 to 0.15
                - 0.03 = tight, regions almost touch
                - 0.05 = default
                - 0.10 = wider gap between regions
                - 0.15 = very wide gap
        seed: Random seed for reproducibility
    
    Returns:
        X: Data points array of shape (n_total, 2)
        y: Labels array of shape (n_total,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp margin to valid range
    original_margin = margin
    margin = max(0.02, min(0.20, margin))
    if margin != original_margin:
        print(f"WARNING: margin={original_margin} out of range, clamped to {margin}")
    
    # Calculate region boundaries based on margin
    edge = 0.02  # Edge margin from boundaries
    
    # Corner region extent (shrinks as margin increases)
    corner_size = 0.30 - margin * 0.5  # e.g., 0.275 for margin=0.05
    corner_max_sum = 0.35 - margin     # How far corners extend (x+y constraint)
    
    # Center band starts after corners + gap
    center_start = corner_max_sum + margin  # Gap between corners and center
    
    # Region definitions (in [0,1] x [0,1] space, must stay inside triangle x+y < 1)
    
    # RED region 1: Bottom-left corner (near vertex 0,0)
    # Constraint: x + y < corner_max_sum (stay in corner)
    r1_points = []
    while len(r1_points) < n_points_per_region:
        x = np.random.uniform(edge, corner_size)
        y = np.random.uniform(edge, corner_size)
        if x + y < corner_max_sum:  # Stay in triangle corner
            r1_points.append([x, y])
    r1 = np.array(r1_points)
    r1_labels = np.ones(n_points_per_region)  # RED
    
    # RED region 2: Bottom-right corner (near vertex 1,0 in normalized space)
    # Constraint: x + y < 1 (inside triangle), y < corner_size, x > (1 - corner_size - margin)
    r2_x_min = 1.0 - corner_size - margin
    r2_points = []
    while len(r2_points) < n_points_per_region:
        x = np.random.uniform(r2_x_min, 0.95)
        y = np.random.uniform(edge, corner_size - margin)
        if x + y < 0.98:  # Stay inside triangle
            r2_points.append([x, y])
    r2 = np.array(r2_points)
    r2_labels = np.ones(n_points_per_region)  # RED
    
    # BLUE region 1: Top corner (near vertex 0,1 in normalized space)
    # Constraint: x + y < 1 (inside triangle), y > (1 - corner_size - margin), x < corner_size
    b1_y_min = 1.0 - corner_size - margin
    b1_points = []
    while len(b1_points) < n_points_per_region:
        x = np.random.uniform(edge, corner_size - margin)
        y = np.random.uniform(b1_y_min, 0.95)
        if x + y < 0.98:  # Stay inside triangle
            b1_points.append([x, y])
    b1 = np.array(b1_points)
    b1_labels = -np.ones(n_points_per_region)  # BLUE
    
    # BLUE region 2: Center band (separates the two RED corners)
    # This is the KEY region that disconnects the RED corners!
    b2_points = []
    while len(b2_points) < n_points_per_region * 2:  # More points in center
        x = np.random.uniform(corner_size + margin, 0.70)
        y = np.random.uniform(corner_size + margin, 0.70)
        if x + y < 0.95 and x + y > center_start:  # Center band inside triangle
            b2_points.append([x, y])
    b2 = np.array(b2_points)
    b2_labels = -np.ones(len(b2))  # BLUE
    
    # Combine all regions
    X = np.vstack([r1, r2, b1, b2])
    y = np.concatenate([r1_labels, r2_labels, b1_labels, b2_labels])
    
    return X, y


if __name__ == "__main__":
    # === CONFIGURATION ===
    N_POINTS = 150
    MARGIN = 0.08  # Gap between regions (try 0.03, 0.05, 0.08, 0.12)
    SEED = 42
    
    X, y = generate_triangle_disconnected_dataset(
        n_points_per_region=N_POINTS,
        margin=MARGIN,
        seed=SEED
    )
    
    np.savez('../datasets/dataset_triangle_disconnected.npz', X=X, y=y)
    
    print(f"Saved triangle-optimized dataset: {X.shape[0]} points")
    print(f"")
    print(f"=== CONFIGURATION ===")
    print(f"Margin: {MARGIN}")
    print(f"")
    print(f"Layout (designed for triangular root simplex):")
    print(f"")
    print(f"         (0,1)")
    print(f"          /\\")
    print(f"         /  \\")
    print(f"        /BLUE\\     <- BLUE at top corner")
    print(f"       /      \\")
    print(f"      /--BLUE--\\   <- BLUE band in center (SEPARATES the reds!)")
    print(f"     / RED  RED \\  <- RED at both bottom corners")
    print(f"    /____________\\")
    print(f" (0,0)          (1,0)")
    print(f"")
    print(f"Classes: {np.unique(y)}")
    print(f"RED points: {int((y == 1).sum())}")
    print(f"BLUE points: {int((y == -1).sum())}")
    print(f"")
    print(f"The BLUE center band separates the two RED corners!")
    print(f"Margin={MARGIN} creates gaps between all regions.")
