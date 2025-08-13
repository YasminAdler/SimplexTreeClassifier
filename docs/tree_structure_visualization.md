# Tree Structure Visualization


### Initial Setup
```
Root Tetrahedron: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
Dimension: 3
Splitting Point: (0.3, 0.4, 0.3)
Branching Factor: 4 (equal to dimension + 1)
```

### Tree Structure After Subdivision

```
                    Root SimplexTree
                    [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
                    (is_simplex_tree() = True)
                           │
                    ┌──────┼──────┐
                    │      │      │
              Child 1  Child 2  Child 3  Child 4
        [(0.3,0.4,0.3), (1,0,0), (0,1,0), (0,0,1)]  [(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)]  [(0,0,0), (1,0,0), (0.3,0.4,0.3), (0,0,1)]  [(0,0,0), (1,0,0), (0,1,0), (0.3,0.4,0.3)]
        (is_simplex() = True)       (is_simplex() = True)       (is_simplex() = True)       (is_simplex() = True)
```

## Subdivision Method: Vertex Replacement

The `add_splitting_point()` method creates children by replacing each vertex with the splitting point:


### Child 0: The Tetrahedron
- **Original**: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]

![Original Tetrahedron](images/Tedhedron_clean.jpg "Original 3D tetrahedron before subdivision")

### Child 1: Replace V₀ with splitting point
- **Original**: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
- **Child 1**: [(0.3,0.4,0.3), (1,0,0), (0,1,0), (0,0,1)]

![First Child Subdivision](images/Tehdron_father.jpg "Tetrahedron with first vertex replaced by splitting point")

### Child 2: Replace V₁ with splitting point
- **Original**: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
- **Child 2**: [(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)]


### Child 3: Replace V₂ with splitting point
- **Original**: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
- **Child 3**: [(0,0,0), (1,0,0), (0.3,0.4,0.3), (0,0,1)]

### Child 4: Replace V₃ with splitting point
- **Original**: [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
- **Child 4**: [(0,0,0), (1,0,0), (0,1,0), (0.3,0.4,0.3)]

![Multiple Children Subdivision](images/Tehedron_grandfather.jpg "Tetrahedron with multiple vertices replaced by splitting points")

## Parent-Child List Structure

```
ParentChildSimplexTreeNode
├── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0,1,0), (0,0,1)])
├── children: 
│   ├── ParentChildSimplexTreeNode
│   │   └── simplex_tree: SimplexTree([(0.3,0.4,0.3), (1,0,0), (0,1,0), (0,0,1)])
│   ├── ParentChildSimplexTreeNode
│   │   └── simplex_tree: SimplexTree([(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)])
│   ├── ParentChildSimplexTreeNode
│   │   └── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0.3,0.4,0.3), (0,0,1)])
│   └── ParentChildSimplexTreeNode
│       └── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0,1,0), (0.3,0.4,0.3)])
```

## Left Child-Right Sibling Structure

```
LeftChildRightSiblingSimplexTreeNode
├── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0,1,0), (0,0,1)])
├── left_child: LeftChildRightSiblingSimplexTreeNode
│   ├── simplex_tree: SimplexTree([(0.3,0.4,0.3), (1,0,0), (0,1,0), (0,0,1)])
│   ├── left_child: None
│   └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│       ├── simplex_tree: SimplexTree([(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)])
│       ├── left_child: None
│       └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│           ├── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0.3,0.4,0.3), (0,0,1)])
│           ├── left_child: None
│           └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│               ├── simplex_tree: SimplexTree([(0,0,0), (1,0,0), (0,1,0), (0.3,0.4,0.3)])
│               ├── left_child: None
│               └── right_sibling: None
└── right_sibling: None
```

## Point Location Algorithm Flow

### Algorithm Visualization

```mermaid
flowchart TD
    A[Start: Point P] --> B{Point in root simplex?}
    B -->|No| C[Return None]
    B -->|Yes| D{Is leaf node?}
    D -->|Yes| E[Return this simplex]
    D -->|No| F[Check all children]
    F --> G{Point in child?}
    G -->|No| F
    G -->|Yes| H[Recurse on child]
    H --> D
    
    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style C fill:#ffcdd2
```

### Example: Finding point (0.2, 0.3, 0.2)

```
1. Start at Root Node
   ├── point_inside_node((0.2, 0.3, 0.2)) → True
   ├── is_simplex_tree() → True (has children)
   └── Check children recursively

2. Check Child 1 [(0.3,0.4,0.3), (1,0,0), (0,1,0), (0,0,1)]
   ├── point_inside_node((0.2, 0.3, 0.2)) → False
   └── Skip this branch

3. Check Child 2 [(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)]
   ├── point_inside_node((0.2, 0.3, 0.2)) → True
   ├── is_simplex() → True (no children)
   └── Return this SimplexTree

Result: Found containing simplex [(0,0,0), (0.3,0.4,0.3), (0,1,0), (0,0,1)]
```

## 2D Triangle Example

### Initial Setup
```
Root Triangle: [(0,0), (4,0), (2,3)]
Dimension: 2
Splitting Point: (1.5, 1.0)
Branching Factor: 3 (equal to dimension + 1)
```

### Tree Structure After Subdivision

```
                    Root SimplexTree
                    [(0,0), (4,0), (2,3)]
                    (is_simplex_tree() = True)
                           │
                    ┌──────┼──────┐
                    │      │      │
              Child 1  Child 2  Child 3
        [(1.5,1.0), (4,0), (2,3)]  [(0,0), (1.5,1.0), (2,3)]  [(0,0), (4,0), (1.5,1.0)]
        (is_simplex() = True)       (is_simplex() = True)       (is_simplex() = True)
```

## API Usage Examples

### Creating a Tree with Subdivision
```python
# Create root vertices (3D tetrahedron)
root_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Create SimplexTree
tree_3d = SimplexTree(root_vertices)

# Add splitting point to create hierarchy
splitting_point = (0.3, 0.4, 0.3)
children = tree_3d.add_point_to_the_most_specific_simplex(splitting_point)

# Wrap in tree node for advanced operations
root_node = ParentChildSimplexTreeNode(tree_3d)
```

### Checking Node Types
```python
# Check if node is a SimplexTree (has children)
if node.is_simplex_tree():
    print("This node contains a SimplexTree with children")
    # Traverse children recursively
    for child in node.get_children():
        result = child.find_containing_simplex(point)
        if result:
            return result

# Check if node is a Simplex (no children)
if node.is_simplex():
    print("This node contains a Simplex (leaf node)")
    # Use point_inside_simplex directly
    if node.point_inside_node(point):
        return node.simplex_tree
```

### Point Location
```python
# Find containing simplex
point = (0.2, 0.3, 0.2)
containing_simplex = root_node.find_containing_simplex(point)

if containing_simplex:
    print(f"Point {point} is in simplex with vertices: {containing_simplex.vertices}")
    # Get barycentric coordinates
    coords = containing_simplex.embed_point(point)
    print(f"Barycentric coordinates: {coords}")
else:
    print(f"Point {point} is not in any simplex")
```

### Tree Traversal
```python
# Get all leaves (simplexes without children)
leaves = tree_3d.get_leaves()
print(f"Number of leaf simplexes: {len(leaves)}")

# Get nodes at specific depth
depth_1_nodes = tree_3d.get_nodes_at_depth(1)
print(f"Number of nodes at depth 1: {len(depth_1_nodes)}")

# Breadth-first traversal
for node in tree_3d.traverse_breadth_first():
    print(f"Node at depth {node.depth}: {node.vertices}")
```

## Visualization Features

### 3D Interactive Visualization
```python
from in3D.utilss.visualization import visualize_simplex_tree

# Visualize tree with splitting point
visualize_simplex_tree(tree_3d, splitting_point, "3D Tetrahedron Subdivision")
```

**Visualization Features:**
- **Color-coded hierarchy**: Different colors for each depth level
- **Transparent overlapping**: Alpha blending for overlapping simplexes
- **Splitting point highlighting**: Yellow star marker for splitting points
- **Interactive 3D**: Rotation, zoom, and pan capabilities
- **Vertex markers**: Clear visualization of simplex vertices

### Color Scheme
- **Red**: Root simplex
- **Blue/Green/Orange/Purple**: First level children
- **Lighter shades**: Deeper levels
- **Yellow star**: Splitting points

## Memory Efficiency Comparison

### Parent-Child List
- **Pros**: Easy to access all children, intuitive structure
- **Cons**: Each node stores a list of children (more memory)
- **Use Case**: When you need frequent access to all children

### Left Child-Right Sibling
- **Pros**: More memory efficient, good for binary-like operations
- **Cons**: More complex traversal, need to follow sibling chain
- **Use Case**: When memory is a concern or you need binary-like operations

## Subdivision Properties

### Branching Factor
The branching factor is always equal to the dimension + 1:
- **2D Triangle**: Branching factor = 3 (3 vertices)
- **3D Tetrahedron**: Branching factor = 4 (4 vertices)
- **nD Simplex**: Branching factor = n + 1 (n + 1 vertices)

### Geometric Properties
- **Volume preservation**: Each child simplex shares the splitting point
- **Hierarchical refinement**: Deeper levels provide finer spatial resolution
- **Overlapping regions**: Children may overlap, providing redundancy
- **Adaptive subdivision**: Points are added to the most specific containing simplex

### Performance Characteristics
- **Point location**: O(log n) average case for balanced trees
- **Memory usage**: O(n) where n is number of simplexes
- **Subdivision cost**: O(d) where d is dimension
- **Traversal efficiency**: Depends on tree structure and node type

## Barycentric Subdivision

### Overview

- **Automatic centroid calculation**: `compute_barycentric_center()` computes the average of all vertices
- **Recursive subdivision**: `add_barycentric_centers_recursively(levels)` subdivides for multiple levels
- **Selective subdivision**: `add_barycentric_centers_to_all_leaves()` subdivides only leaf nodes
- **Depth-based subdivision**: `add_barycentric_centers_at_depth(depth)` targets specific tree levels
- **Mixed subdivision**: Combine manual point addition with automatic barycentric subdivision

### Example 1: Computing Barycentric Center

```python
vertices = [(0, 0), (1, 0), (0.5, 1)] 
tree = SimplexTree(vertices)

barycenter = tree.compute_barycentric_center()
print(f"Barycentric point of the initial triangle is: {barycenter}")
# Output: Barycentric point of the initial triangle is: (0.5, 0.3333333333333333)
```

### Example 2: Recursive Barycentric Subdivision

```python
tree_barycentric = SimplexTree(vertices)
tree_barycentric.add_barycentric_centers_recursively(2)

print("\nTree structure after barycentric subdivision:")
print(f"Total nodes: {tree_barycentric.get_node_count()}")
print(f"Tree depth: {tree_barycentric.get_depth()}")
print(f"Leaf nodes: {len(tree_barycentric.get_leaves())}")

# Output:
# Level 1: Subdivided 1 simplexes
# Level 2: Subdivided 3 simplexes
# 
# Tree structure after barycentric subdivision:
# Total nodes: 13
# Tree depth: 2
# Leaf nodes: 9

visualize_simplex_tree(tree_barycentric, None, "tree_barycentric_after_subdivision")
```

![Barycentric Subdivision](images/barycenter_recursively.png "Triangle after 2 levels of barycentric subdivision")

### Example 3: Mixed Subdivision (Manual + Barycentric)

```python
tree_mixed = SimplexTree(vertices)
test_point = (0.343, 0.2)

print(f"Adding custom point: {test_point}")
tree_mixed.add_point_to_the_most_specific_simplex(test_point)

visualize_simplex_tree(tree_mixed, test_point, "tree with manually added point")
```

![Manual Point Addition](images/manual_point.png "Triangle with manually added point")

```python
print("\nNow adding barycentric centers to all leaves...")
count = tree_mixed.add_barycentric_centers_to_all_leaves()
print(f"Subdivided {count} leaf simplexes")
# Output: Subdivided 3 leaf simplexes

visualize_simplex_tree(tree_mixed, None, "tree_mixed-manual point-barycentric centers")
```

![Mixed Subdivision](images/mixed_manual_and_barycentric.png "Triangle with both manual and barycentric subdivision")

### Subdivision Properties

#### Growth Pattern
When applying barycentric subdivision:
- **Level 1**: 1 simplex → 3 children (2D) or 4 children (3D)
- **Level 2**: Each child subdivides → 9 total leaves (2D) or 16 total leaves (3D)
- **Level n**: Number of leaves = (d+1)^n where d is dimension

#### Tree Structure After Barycentric Subdivision

```
2D Triangle Example:
                    Root SimplexTree
                    [(0,0), (1,0), (0.5,1)]
                    barycenter: (0.5, 0.333)
                           │
                    ┌──────┼──────┐
              Child 1  Child 2  Child 3
        [(0.5,0.333), (1,0), (0.5,1)]  [(0,0), (0.5,0.333), (0.5,1)]  [(0,0), (1,0), (0.5,0.333)]
              │                              │                              │
         (3 children)                   (3 children)                   (3 children)
```

#### Geometric Properties of Barycentric Subdivision
- **Uniform refinement**: All simplexes at the same level have similar sizes
- **Volume preservation**: Total volume remains constant
- **Centroid property**: Each child simplex contains the parent's barycentric center
- **Symmetry**: The subdivision pattern is symmetric with respect to the original simplex

### API Methods for Barycentric Subdivision

#### `compute_barycentric_center() → Tuple[float, ...]`
Computes the centroid of the simplex by averaging all vertices.

```python
barycenter = simplex.compute_barycentric_center()
# For triangle [(0,0), (1,0), (0.5,1)]: returns (0.5, 0.333...)
```

#### `add_barycentric_centers_to_all_leaves() → int`
Subdivides all leaf nodes by adding their barycentric centers.

```python
count = tree.add_barycentric_centers_to_all_leaves()
print(f"Subdivided {count} simplexes")
```

#### `add_barycentric_centers_at_depth(depth: int) → int`
Subdivides only the leaf nodes at a specific depth.

```python
count = tree.add_barycentric_centers_at_depth(1)
# Only subdivides depth-1 nodes that don't have children
```

#### `add_barycentric_centers_recursively(levels: int) → None`
Repeatedly applies barycentric subdivision for the specified number of levels.

```python
tree.add_barycentric_centers_recursively(3)
# Level 1: Subdivided 1 simplexes
# Level 2: Subdivided 3 simplexes  
# Level 3: Subdivided 9 simplexes
```