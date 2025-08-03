# 2D Simplex Tree Classifier

This folder contains the 2D implementation of the Simplex Tree Classifier, which provides hierarchical triangle subdivision and efficient point location in 2D space.

## Structure

```
in2D/
├── classes/                          # Core classes
│   ├── __init__.py
│   ├── simplex.py                   # Base 2D Simplex class (triangles)
│   ├── simplexTree.py               # Hierarchical 2D SimplexTree
│   ├── parentChildSimplexTreeNode.py
│   ├── leftChildRightSiblingSimplexTreeNode.py
│   └── simplexTreeImport.py
├── utilss/
│   ├── __init__.py
│   └── visualization.py             # 2D visualization tools
├── example_usage.py                 # Simple usage example
├── test_2d_simplex_tree.py          # Comprehensive test suite
└── README.md                        # This file
```

## Key Features

### 2D Simplex (Triangle)
- **Mathematical representation** of triangles using barycentric coordinates
- **Point-in-triangle testing** with proper matrix operations
- **Handles degenerate cases** gracefully (collinear vertices)
- **Barycentric coordinate calculation** for points inside triangles

### 2D SimplexTree
- **Hierarchical triangle subdivision** using splitting points
- **Tree traversal** (breadth-first, depth-first)
- **Point location algorithm** to find containing triangle
- **Recursive subdivision** with vertex replacement method

### Tree Node Wrappers
- **ParentChildSimplexTreeNode**: Traditional parent-child list structure
- **LeftChildRightSiblingSimplexTreeNode**: Memory-efficient left-child-right-sibling structure
- **Unified APIs** for tree operations

### 2D Visualization
- **Interactive 2D plots** using matplotlib
- **Color-coded hierarchical levels**
- **Triangle rendering** with transparency
- **Splitting point highlighting**

## Usage Examples

### Basic Triangle Creation
```python
from classes.simplexTree import SimplexTree

# Create a triangle
vertices = [(0, 0), (4, 0), (2, 3)]
triangle = SimplexTree(vertices)

# Test if a point is inside
point = (2, 1)
is_inside = triangle.point_inside_simplex(point)
print(f"Point {point} inside triangle: {is_inside}")
```

### Hierarchical Subdivision
```python
# Create tree and add splitting points
tree = SimplexTree([(0, 0), (4, 0), (2, 3)])
splitting_point = (1.5, 1.0)
children = tree.add_point_to_the_most_specific_simplex(splitting_point)

# Each splitting point creates 3 child triangles (branching factor = 3)
print(f"Created {len(children)} child triangles")
```

### Point Location
```python
# Find which triangle contains a point
test_point = (1.0, 0.5)
containing_simplex = tree.find_containing_simplex(test_point)

if containing_simplex:
    print(f"Point found in: {containing_simplex.get_vertices_as_tuples()}")
    coords = containing_simplex.embed_point(test_point)
    print(f"Barycentric coordinates: {coords}")
```

### Tree Node Operations
```python
from classes.parentChildSimplexTreeNode import ParentChildSimplexTreeNode

# Wrap tree in node for advanced operations
root_node = ParentChildSimplexTreeNode(tree)

# Check node types
if root_node.is_simplex_tree():
    print("This node contains a SimplexTree with children")
    
if root_node.is_simplex():
    print("This node contains a Simplex (leaf node)")
```

### Visualization
```python
from utilss.visualization import visualize_simplex_tree

# Visualize the tree with splitting points
visualize_simplex_tree(tree, splitting_point, "2D Triangle Subdivision")
```

## Mathematical Foundation

### Barycentric Coordinates
For a point P in triangle with vertices V₀, V₁, V₂, the barycentric coordinates (α₀, α₁, α₂) satisfy:
- P = α₀V₀ + α₁V₁ + α₂V₂
- α₀ + α₁ + α₂ = 1
- P is inside the triangle iff all αᵢ ∈ [0,1]

### Subdivision Method
The `add_splitting_point()` method creates children by replacing each vertex with the splitting point:
- **Child 1**: Replace V₀ with splitting point
- **Child 2**: Replace V₁ with splitting point  
- **Child 3**: Replace V₂ with splitting point

### Branching Factor
For 2D triangles, the branching factor is always 3 (equal to dimension + 1):
- Each splitting point creates exactly 3 child triangles
- This provides steady subdivision without exponential growth

## Performance Characteristics

- **Point location**: O(log n) average case for balanced trees
- **Memory usage**: O(n) where n is number of triangles
- **Subdivision cost**: O(1) per splitting point
- **Barycentric calculation**: O(1) using cached transformation matrices

## Running Examples

```bash
# Simple example
python example_usage.py

# Comprehensive test suite
python test_2d_simplex_tree.py
```

## Differences from 3D Version

1. **Dimension**: Works with 2D triangles instead of 3D tetrahedra
2. **Branching factor**: 3 instead of 4 (dimension + 1)
3. **Visualization**: 2D plots instead of 3D interactive plots
4. **Vertex count**: 3 vertices per simplex instead of 4
5. **Coordinate system**: (x, y) instead of (x, y, z)

## Dependencies

- Python 3.7+
- NumPy 1.21.0+
- Matplotlib 3.5.0+ (for visualization)

The 2D implementation provides the same functionality as the 3D version but optimized for 2D geometric operations, making it ideal for applications involving triangle meshes, 2D spatial queries, and planar geometric algorithms. 