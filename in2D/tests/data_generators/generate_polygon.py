import numpy as np
from matplotlib.path import Path


def _regular_polygon_vertices(n_sides, center=(0.5, 0.4), radius=0.28, rotation=0.0):
    """Generate vertices of a regular n-sided polygon (always convex)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False) + rotation
    cx, cy = center
    return np.column_stack([cx + radius * np.cos(angles),
                            cy + radius * np.sin(angles)])


def generate_polygon_dataset(n_inside=250,
                             n_outside=350,
                             polygon_vertices=None,
                             n_sides=6,
                             center=(0.5, 0.4),
                             radius=0.28,
                             rotation=0.0,
                             boundary_margin=0.015,
                             inside_class=1,
                             outside_class=-1,
                             stay_in_triangle=True,
                             seed=42):
    """
    Generate a 2D dataset where one class fills a polygon and the other class
    is scattered as background around it.

    Layout (default regular hexagon, data normalised to [0,1]):

                  (0, 1)
                   /\
                  /  \
                 / .  \         . = OUTSIDE class (background)
                /  ___ \
               /  /   \.\       RED polygon interior = INSIDE class
              / . | R | \
             /    \___/  \
            /  .          \
           /_______________\
         (0,0)            (1,0)

    Because a regular polygon is always convex, the resulting decision region is
    convex (no "non-convex" blobs appear during classification). Pass custom
    `polygon_vertices` if you want a specific polygon (convex or not).

    Args:
        n_inside: Number of points inside the polygon (INSIDE class).
        n_outside: Number of background points outside the polygon (OUTSIDE class).
        polygon_vertices: Optional (N, 2) array of polygon vertex coordinates.
            If None, a regular polygon is generated from `n_sides`, `center`,
            `radius`, `rotation`.
        n_sides: Number of sides of the regular polygon (ignored if
            polygon_vertices is given). 3 = triangle, 4 = square, 5 = pentagon,
            6 = hexagon, ...
        center: (x, y) centre of the regular polygon, in [0,1]^2.
        radius: Circumradius of the regular polygon, in [0,1] units.
        rotation: Rotation of the regular polygon, in radians.
        boundary_margin: Clear band (in [0,1] units) kept around the polygon
            so background points don't sit right on the boundary. Set to 0.0 to
            let the two classes touch.
        inside_class: Label assigned to points inside the polygon (default +1).
        outside_class: Label assigned to background points (default -1).
        stay_in_triangle: If True, constrain all points to lie strictly inside
            the root simplex x + y < 1 (matching the triangular root simplex
            (0,0), (2,0), (0,2) after [0,1] normalisation).
        seed: Random seed for reproducibility.

    Returns:
        X: Data points array of shape (n_inside + n_outside, 2) in [0,1]^2.
        y: Labels array of shape (n_inside + n_outside,).
        polygon: (N, 2) array with the polygon vertices actually used.
    """
    if seed is not None:
        np.random.seed(seed)

    if polygon_vertices is None:
        polygon_vertices = _regular_polygon_vertices(n_sides, center, radius, rotation)
    else:
        polygon_vertices = np.asarray(polygon_vertices, dtype=float)

    if polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2 or polygon_vertices.shape[0] < 3:
        raise ValueError("polygon_vertices must be an array of shape (N, 2) with N >= 3.")

    polygon_path = Path(polygon_vertices, closed=True)

    xmin, ymin = polygon_vertices.min(axis=0)
    xmax, ymax = polygon_vertices.max(axis=0)

    tri_limit = 0.98 if stay_in_triangle else np.inf

    def inside_root_triangle(x, y):
        return (x + y) < tri_limit

    inside_points = []
    attempts = 0
    max_attempts = max(n_inside * 500, 50_000)
    while len(inside_points) < n_inside and attempts < max_attempts:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        if polygon_path.contains_point((x, y)) and inside_root_triangle(x, y):
            inside_points.append([x, y])
        attempts += 1

    if len(inside_points) < n_inside:
        print(f"WARNING: only generated {len(inside_points)}/{n_inside} inside points "
              f"(polygon may not fit inside the root triangle).")

    inside = np.asarray(inside_points) if inside_points else np.zeros((0, 2))

    outside_points = []
    attempts = 0
    max_attempts = max(n_outside * 500, 50_000)
    while len(outside_points) < n_outside and attempts < max_attempts:
        x = np.random.uniform(0.02, 0.98)
        y = np.random.uniform(0.02, 0.98)
        if not inside_root_triangle(x, y):
            attempts += 1
            continue
        if polygon_path.contains_point((x, y), radius=boundary_margin):
            attempts += 1
            continue
        if polygon_path.contains_point((x, y), radius=-boundary_margin):
            attempts += 1
            continue
        outside_points.append([x, y])
        attempts += 1

    if len(outside_points) < n_outside:
        print(f"WARNING: only generated {len(outside_points)}/{n_outside} outside points.")

    outside = np.asarray(outside_points) if outside_points else np.zeros((0, 2))

    X = np.vstack([inside, outside]) if len(outside) > 0 else inside
    y = np.concatenate([
        np.full(len(inside), inside_class, dtype=float),
        np.full(len(outside), outside_class, dtype=float),
    ])

    idx = np.random.permutation(len(X))
    return X[idx], y[idx], polygon_vertices


if __name__ == "__main__":
    import os

    N_INSIDE = 250
    N_OUTSIDE = 350
    N_SIDES = 6
    CENTER = (0.45, 0.35)
    RADIUS = 0.26
    ROTATION = 0.0
    BOUNDARY_MARGIN = 0.02
    SEED = 42

    X, y, polygon = generate_polygon_dataset(
        n_inside=N_INSIDE,
        n_outside=N_OUTSIDE,
        n_sides=N_SIDES,
        center=CENTER,
        radius=RADIUS,
        rotation=ROTATION,
        boundary_margin=BOUNDARY_MARGIN,
        seed=SEED,
    )

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, 'dataset_polygon.npz'))
    np.savez(out_path, X=X, y=y, polygon=polygon)

    n_pos = int((y == 1).sum())
    n_neg = int((y == -1).sum())

    print("")
    print("=" * 50)
    print(f"POLYGON DATASET ({N_SIDES}-gon)")
    print("=" * 50)
    print("")
    print(f"Total points: {X.shape[0]}")
    print(f"  RED  (inside,  +1): {n_pos} points")
    print(f"  BLUE (outside, -1): {n_neg} points")
    print("")
    print(f"Polygon centre: {CENTER}")
    print(f"Polygon radius: {RADIUS}")
    print(f"Polygon sides : {N_SIDES}")
    print(f"Boundary gap  : {BOUNDARY_MARGIN}")
    print("")
    print(f"Saved to: {out_path}")
