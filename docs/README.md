# Simplex Point Location Algorithm

This Python implementation solves the problem of determining which sub-triangle (simplex) a point lies within using barycentric coordinates.

## Algorithm Overview

Given a point P and a set of triangles (simplexes), the algorithm:

1. **Enumerates all simplexes** - Each triangle Δⱼ has vertices (Vⱼ,₀, Vⱼ,₁, Vⱼ,₂)
2. **For each simplex Δⱼ:**
   - Pick one vertex as "origin" (Vⱼ,₀)
   - Build the 2×2 matrix: `Aⱼ = [Vⱼ,₁ - Vⱼ,₀, Vⱼ,₂ - Vⱼ,₀]`
   - Compute the offset: `bⱼ = P - Vⱼ,₀`
   - Solve for barycentric coordinates: `(αⱼ,₁, αⱼ,₂) = Aⱼ⁻¹ bⱼ`
   - Calculate: `αⱼ,₀ = 1 - αⱼ,₁ - αⱼ,₂`
3. **Test containment:** Point P is inside Δⱼ if all coordinates are in [0,1]
4. **Return the first matching simplex**

## Key Features

- **Correct barycentric coordinate calculation** - Uses proper matrix inversion
- **Proper containment testing** - Checks that all αᵢ ∈ [0,1] (not just ≥ 0)
- **Handles degenerate cases** - Gracefully handles collinear vertices
- **Multiple search methods** - Find simplex with/without coordinates

## Example Usage

```python
from simplex_point_location import Triangle, SimplexPointLocator

# Create triangles
triangle1 = Triangle((0, 0), (4, 0), (2, 2))
triangle2 = Triangle((0, 0), (2, 2), (0, 4))

# Create locator
locator = SimplexPointLocator([triangle1, triangle2])

# Find which triangle contains point
point = (2, 0.5)
result = locator.find_containing_simplex_with_coords(point)

if result:
    index, triangle, coords = result
    print(f"Point {point} is in triangle {index}")
    print(f"Barycentric coordinates: {coords}")
```

## Mathematical Background

### Barycentric Coordinates
For a point P in triangle with vertices V₀, V₁, V₂, the barycentric coordinates (α₀, α₁, α₂) satisfy:
- P = α₀V₀ + α₁V₁ + α₂V₂  
- α₀ + α₁ + α₂ = 1
- P is inside the triangle iff all αᵢ ∈ [0,1]

### Matrix Formulation
Setting V₀ as origin, we solve:
```
[V₁-V₀, V₂-V₀] [α₁] = P - V₀
                [α₂]
```

Then α₀ = 1 - α₁ - α₂.

## Files

- `simplex_point_location.py` - Main implementation
- `test_additional_points.py` - Additional test cases
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.7+
- NumPy 1.21.0+

## Installation

```bash
pip install -r requirements.txt
```

## Running Examples

```bash
python simplex_point_location.py
python test_additional_points.py
```

## Performance Optimizations

The implementation includes several optimization opportunities mentioned in the original description:

1. **Bounding box pre-check** - Could reject triangles whose bounding box doesn't contain P
2. **Walking algorithm** - Could walk from one simplex to neighbor toward P
3. **Spatial indexing** - Could use quadtree/k-d tree for larger triangle sets

These optimizations become important for large numbers of triangles but are not implemented in this basic version. 