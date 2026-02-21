import numpy as np

def generate_checkerboard_dataset(n_points_per_quadrant=200, margin=0.05, seed=42):
    """
    Generate a CHECKERBOARD dataset that forces DISCONNECTED regions.
    
    Layout (in [0,1] x [0,1] space):
    
        y=1.0 ┌─────────┬─────────┐
              │  BLUE   │   RED   │
              │  (-1)   │   (+1)  │
        y=0.5 ├─────────┼─────────┤
              │   RED   │  BLUE   │
              │  (+1)   │  (-1)   │
        y=0.0 └─────────┴─────────┘
             x=0      x=0.5      x=1
    
    This pattern FORCES the classifier to create:
    - TWO disconnected RED regions (bottom-left & top-right)
    - TWO disconnected BLUE regions (top-left & bottom-right)
    
    Because diagonal regions of the same class CANNOT be connected!
    
    Args:
        n_points_per_quadrant: Number of points per quadrant
        margin: Gap between quadrants (space with no data).
                Valid range: 0.02 to 0.15
                - 0.03 = tight
                - 0.05 = default
                - 0.10 = wide gap
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
    
    edge_margin = 0.02  # Small margin from outer edges
    
    # Bottom-left quadrant: RED (+1)
    q1_x = np.random.uniform(edge_margin, 0.5 - margin, n_points_per_quadrant)
    q1_y = np.random.uniform(edge_margin, 0.5 - margin, n_points_per_quadrant)
    q1 = np.column_stack([q1_x, q1_y])
    q1_labels = np.ones(n_points_per_quadrant)  # RED
    
    # Bottom-right quadrant: BLUE (-1)
    q2_x = np.random.uniform(0.5 + margin, 1.0 - edge_margin, n_points_per_quadrant)
    q2_y = np.random.uniform(edge_margin, 0.5 - margin, n_points_per_quadrant)
    q2 = np.column_stack([q2_x, q2_y])
    q2_labels = -np.ones(n_points_per_quadrant)  # BLUE
    
    # Top-left quadrant: BLUE (-1)
    q3_x = np.random.uniform(edge_margin, 0.5 - margin, n_points_per_quadrant)
    q3_y = np.random.uniform(0.5 + margin, 1.0 - edge_margin, n_points_per_quadrant)
    q3 = np.column_stack([q3_x, q3_y])
    q3_labels = -np.ones(n_points_per_quadrant)  # BLUE
    
    # Top-right quadrant: RED (+1)
    q4_x = np.random.uniform(0.5 + margin, 1.0 - edge_margin, n_points_per_quadrant)
    q4_y = np.random.uniform(0.5 + margin, 1.0 - edge_margin, n_points_per_quadrant)
    q4 = np.column_stack([q4_x, q4_y])
    q4_labels = np.ones(n_points_per_quadrant)  # RED
    
    # Combine all quadrants
    X = np.vstack([q1, q2, q3, q4])
    y = np.concatenate([q1_labels, q2_labels, q3_labels, q4_labels])
    
    return X, y


def generate_4quadrant_dataset(n_points_per_quadrant=200, margin=0.05, seed=42):
    """
    Alternative: 4 quadrants with alternating classes in a 2x2 grid.
    Same as checkerboard but explicitly named.
    """
    return generate_checkerboard_dataset(n_points_per_quadrant, margin, seed)


if __name__ == "__main__":
    # === CONFIGURATION ===
    N_POINTS = 200
    MARGIN = 0.08  # Gap between quadrants (try 0.05, 0.08, 0.12, 0.15)
    SEED = 42
    
    X, y = generate_checkerboard_dataset(
        n_points_per_quadrant=N_POINTS,
        margin=MARGIN,
        seed=SEED
    )
    
    np.savez('../datasets/dataset_checkerboard.npz', X=X, y=y)
    
    # Calculate actual data ranges
    edge = 0.02
    q_min = edge
    q_max_low = 0.5 - MARGIN
    q_min_high = 0.5 + MARGIN
    q_max = 1.0 - edge
    gap = 2 * MARGIN
    
    print(f"Saved checkerboard dataset: {X.shape[0]} points")
    print(f"")
    print(f"=== CONFIGURATION ===")
    print(f"Margin: {MARGIN}")
    print(f"Gap between quadrants: {gap:.3f}")
    print(f"")
    print(f"=== QUADRANT RANGES ===")
    print(f"Bottom-left  (RED):  x=[{q_min:.2f}, {q_max_low:.2f}], y=[{q_min:.2f}, {q_max_low:.2f}]")
    print(f"Bottom-right (BLUE): x=[{q_min_high:.2f}, {q_max:.2f}], y=[{q_min:.2f}, {q_max_low:.2f}]")
    print(f"Top-left     (BLUE): x=[{q_min:.2f}, {q_max_low:.2f}], y=[{q_min_high:.2f}, {q_max:.2f}]")
    print(f"Top-right    (RED):  x=[{q_min_high:.2f}, {q_max:.2f}], y=[{q_min_high:.2f}, {q_max:.2f}]")
    print(f"")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: RED={int((y == 1).sum())}, BLUE={int((y == -1).sum())}")
    print(f"")
    print(f"This pattern forces TWO DISCONNECTED red regions and TWO DISCONNECTED blue regions!")
