import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler


def generate_multiclass_dataset(n_samples=1000, n_classes=3, margin=0.05, pattern='spiral', dimensions=2, seed=None):
    """
    Generate a multiclass dataset with specified number of classes.
    
    Args:
        n_samples: Total number of samples to generate
        n_classes: Number of classes (3, 4, 5, etc.)
        margin: Gap between classes (meaning varies by pattern):
                - spiral: gap between spiral arms (0.02-0.15)
                - rings: gap between concentric rings (0.02-0.20)
                - moons: gap between crescent shapes (0.02-0.20)
                - clusters: gap between clusters (linearly separable!)
        pattern: Data distribution pattern:
                - 'spiral': Interleaved spirals (NON-LINEAR, recommended)
                - 'rings': Concentric rings (NON-LINEAR)
                - 'moons': Interleaved crescents (NON-LINEAR)
                - 'clusters': Simple Gaussian clusters (LINEAR - easy!)
        dimensions: Number of features (default 2, only 2D supported for non-linear)
        seed: Random seed for reproducibility
        
    Returns:
        X: Feature array of shape (n_samples, dimensions)
        y: Label array with values 0, 1, 2, ... (n_classes-1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clamp margin to valid range
    original_margin = margin
    margin = max(0.02, min(0.25, margin))
    if margin != original_margin:
        print(f"WARNING: margin={original_margin} out of range, clamped to {margin}")
    
    samples_per_class = n_samples // n_classes
    
    if pattern == 'spiral':
        X, y = _generate_spiral(n_classes, samples_per_class, margin)
    elif pattern == 'rings':
        X, y = _generate_rings(n_classes, samples_per_class, margin)
    elif pattern == 'moons':
        X, y = _generate_moons(n_classes, samples_per_class, margin)
    else:  # 'clusters' - linearly separable
        X, y = _generate_clusters(n_classes, samples_per_class, margin, dimensions)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def _generate_spiral(n_classes, samples_per_class, margin):
    """Generate interleaved spiral arms - NON-LINEAR separable."""
    X_list = []
    y_list = []
    
    # Very low noise for clean spirals - margin controls it slightly
    noise = max(0.008, 0.02 - margin * 0.1)
    
    # Number of turns the spiral makes (less = cleaner separation)
    n_turns = 1.5  # 1.5 rotations total
    
    for class_idx in range(n_classes):
        # Each class is a spiral arm starting at different angle
        start_angle = class_idx * (2 * np.pi / n_classes)
        
        # Parameter t goes from 0 to 1
        t = np.linspace(0.05, 1.0, samples_per_class)
        
        # Spiral: angle increases with t, radius increases with t
        angle = start_angle + t * n_turns * 2 * np.pi
        radius = t * 0.45  # Radius grows from 0 to 0.45
        
        # Add small noise perpendicular to the spiral direction
        x = 0.5 + radius * np.cos(angle) + np.random.normal(0, noise, samples_per_class)
        y_coord = 0.5 + radius * np.sin(angle) + np.random.normal(0, noise, samples_per_class)
        
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(samples_per_class, class_idx))
    
    return np.vstack(X_list), np.concatenate(y_list)


def _generate_rings(n_classes, samples_per_class, margin):
    """Generate concentric rings - NON-LINEAR separable."""
    X_list = []
    y_list = []
    
    # Ring thickness - thin rings for clear separation
    ring_width = max(0.01, 0.025 - margin * 0.1)
    
    # Calculate ring spacing based on number of classes
    max_radius = 0.45
    ring_spacing = max_radius / n_classes
    
    for class_idx in range(n_classes):
        # Each class is a ring at different radius
        # Inner ring starts small, outer rings are larger
        ring_radius = 0.08 + class_idx * (ring_spacing + margin * 0.5)
        
        angles = np.random.uniform(0, 2 * np.pi, samples_per_class)
        radii = ring_radius + np.random.normal(0, ring_width, samples_per_class)
        
        x = 0.5 + radii * np.cos(angles)
        y_coord = 0.5 + radii * np.sin(angles)
        
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(samples_per_class, class_idx))
    
    return np.vstack(X_list), np.concatenate(y_list)


def _generate_moons(n_classes, samples_per_class, margin):
    """Generate interleaved crescent/moon shapes - NON-LINEAR separable."""
    X_list = []
    y_list = []
    
    # Low noise for cleaner moons
    noise = max(0.015, 0.03 - margin * 0.1)
    
    radius = 0.20
    
    for class_idx in range(n_classes):
        # Generate half-circle (moon shape)
        angles = np.linspace(0, np.pi, samples_per_class) + np.random.normal(0, 0.05, samples_per_class)
        
        # Each moon is rotated and offset to interleave
        rotation = class_idx * np.pi / n_classes
        
        # Stagger positions so moons interlock
        offset_x = 0.12 * np.cos(class_idx * 2 * np.pi / n_classes)
        offset_y = 0.12 * np.sin(class_idx * 2 * np.pi / n_classes) + class_idx * margin * 0.3
        
        x = radius * np.cos(angles + rotation) + np.random.normal(0, noise, samples_per_class)
        y_coord = radius * np.sin(angles + rotation) + np.random.normal(0, noise, samples_per_class)
        
        # Translate to center + offset
        x = 0.5 + x + offset_x
        y_coord = 0.45 + y_coord + offset_y
        
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(samples_per_class, class_idx))
    
    return np.vstack(X_list), np.concatenate(y_list)


def _generate_clusters(n_classes, samples_per_class, margin, dimensions):
    """Generate simple Gaussian clusters - LINEARLY separable (easy!)."""
    X_list = []
    y_list = []
    
    # Generate cluster centers evenly distributed in a circle
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    radius = 0.35
    centers = np.column_stack([
        0.5 + radius * np.cos(angles),
        0.5 + radius * np.sin(angles)
    ])
    
    # Noise inversely related to margin
    noise_std = max(0.04, 0.15 - margin)
    
    for class_idx in range(n_classes):
        center = centers[class_idx]
        points = np.random.normal(center, noise_std, (samples_per_class, dimensions))
        X_list.append(points)
        y_list.append(np.full(samples_per_class, class_idx))
    
    return np.vstack(X_list), np.concatenate(y_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiclass 2D dataset')
    parser.add_argument('--classes', type=int, required=True, help='Number of classes (e.g., 3, 4, 5)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    parser.add_argument('--margin', type=float, default=0.05, help='Gap between classes (default: 0.05)')
    parser.add_argument('--pattern', type=str, default='spiral', 
                        choices=['spiral', 'rings', 'moons', 'clusters'],
                        help='Pattern: spiral, rings, moons (non-linear) or clusters (linear)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    X, y = generate_multiclass_dataset(
        n_samples=args.samples,
        n_classes=args.classes,
        margin=args.margin,
        pattern=args.pattern,
        seed=args.seed
    )
    
    # Normalize data to [0, 1] range
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Save to datasets folder - include pattern in filename
    filename = f'../datasets/multiclass_{args.classes}classes_{args.pattern}.npz'
    np.savez(filename, X=X, y=y)
    
    # Determine separability
    is_linear = args.pattern == 'clusters'
    separability = "LINEARLY separable (easy!)" if is_linear else "NON-LINEARLY separable (requires complex boundary)"
    
    print(f"Saved {args.classes}-class dataset: {X.shape[0]} points")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Data normalized to range [0, 1]")
    print(f"")
    print(f"=== CONFIGURATION ===")
    print(f"Pattern: {args.pattern}")
    print(f"Margin: {args.margin}")
    print(f"Separability: {separability}")
    print(f"")
    print(f"Saved to: {filename}")
