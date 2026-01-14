# SimplexTree Class Hierarchy and Architecture

## Class Hierarchy Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Simplex (Base Class)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ • vertices: List[np.ndarray]                                                │
│ • n_vertices: int                                                           │
│ • dimension: int                                                            │
│ • tolerance: float                                                          │
│ • A: np.ndarray (transformation matrix)                                     │
│ • det_A: float (determinant)                                                │
│ • is_degenerate: bool                                                       │
│ • A_inv: np.ndarray (inverse matrix)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Methods:                                                                    │
│ • __init__(vertices, tolerance)                                             │
│ • _build_transformation_matrix()                                            │
│ • is_linearly_independent() → bool                                          │
│ • get_determinant() → float                                                 │
│ • can_perform_test() → bool                                                 │
│ • _embed_point(point) → Optional[Tuple[float, ...]]                         │
│ • _point_inside_simplex(point) → bool                                        │
│ • convert_2d_to_homogeneous(point_2d) → Tuple[float, float, float]         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ inherits
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SimplexTree (Main Class)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Inherits ALL Simplex properties and methods                                 │
│ + Tree functionality:                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tree Properties:                                                            │
│ • children: List[SimplexTree]                                               │
│ • parent: Optional[SimplexTree]                                             │
│ • depth: int                                                                │
│ • _node_count: int                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tree Methods:                                                               │
│ • add_child(child_vertices) → SimplexTree                                   │
│ • _get_children() → List[SimplexTree]                                        │
│ • _is_leaf() → bool                                                          │
│ • get_child_count() → int                                                   │
│ • _traverse_breadth_first() → Iterator[SimplexTree]                         │
│ • traverse_depth_first() → Iterator[SimplexTree]                           │
│ • get_leaves() → List[SimplexTree]                                          │
│ • get_nodes_at_depth(target_depth) → List[SimplexTree]                     │
│ • find_containing_simplex(point) → Optional[SimplexTree]                   │
│ • _add_splitting_point(point) → List[SimplexTree]                           │
│ • add_point_to_the_most_specific_simplex(point) → List[SimplexTree]        │
│ • get_vertices_as_tuples() → List[Tuple[float, ...]]                       │
│ • print_tree() → None                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ contains
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Tree Node Wrappers (Two Methodologies)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │              ParentChildSimplexTreeNode                                │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Properties:                                                             │ │
│ │ • simplex_tree: SimplexTree                                             │ │
│ │ • children: List[ParentChildSimplexTreeNode]                           │ │
│ │ • parent: Optional[ParentChildSimplexTreeNode]                         │ │
│ │ • depth: int                                                            │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Methods:                                                                │ │
│ │ • add_child(child_node)                                                 │ │
│ │ • _get_children() → List[ParentChildSimplexTreeNode]                    │ │
│ │ • remove_child(child_node) → bool                                       │ │
│ │ • _is_leaf() → bool                                                      │ │
│ │ • get_child_count() → int                                               │ │
│ │ • is_simplex_tree() → bool                                              │ │
│ │ • is_simplex() → bool                                                   │ │
│ │ • point_inside_node(point) → bool                                       │ │
│ │ • find_containing_simplex(point) → Optional[SimplexTree]               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │           LeftChildRightSiblingSimplexTreeNode                         │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Properties:                                                             │ │
│ │ • simplex_tree: SimplexTree                                             │ │
│ │ • left_child: Optional[LeftChildRightSiblingSimplexTreeNode]           │ │
│ │ • right_sibling: Optional[LeftChildRightSiblingSimplexTreeNode]        │ │
│ │ • parent: Optional[LeftChildRightSiblingSimplexTreeNode]               │ │
│ │ • depth: int                                                            │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Methods:                                                                │ │
│ │ • add_child(child_node)                                                 │ │
│ │ • _get_children() → List[LeftChildRightSiblingSimplexTreeNode]          │ │
│ │ • remove_child(child_node) → bool                                       │ │
│ │ • _is_leaf() → bool                                                      │ │
│ │ • get_child_count() → int                                               │ │
│ │ • is_simplex_tree() → bool                                              │ │
│ │ • is_simplex() → bool                                                   │ │
│ │ • point_inside_node(point) → bool                                       │ │
│ │ • find_containing_simplex(point) → Optional[SimplexTree]               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ uses
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Visualization (Utilities)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Functions:                                                                  │
│ • visualize_simplex_tree(tree, splitting_point, title) → None              │
│ • _visualize_3d_simplex(vertices, ax, color, alpha, linewidth, s, label)   │
│ • _visualize_3d_children_recursive(node, ax, colors, depth)                │
│ • _get_tetrahedron_faces(vertices) → List[List[Tuple[float, ...]]]         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How It Works - Data Flow

### 1. **Simplex (Base Class)**
- **Purpose**: Mathematical representation of a simplex (triangle, tetrahedron, etc.)
- **Holds**: Vertices, transformation matrices, geometric calculations
- **Provides**: Point-in-simplex testing, barycentric coordinates via `_embed_point()`
- **Key Method**: `_point_inside_simplex(point)` - Tests if point is inside simplex

### 2. **SimplexTree (Main Class)**
- **Purpose**: A Simplex that can have children (other SimplexTrees)
- **Inherits**: ALL properties and methods from Simplex
- **Adds**: Tree functionality (children, traversal, hierarchical subdivision)
- **Key Concept**: Every SimplexTree IS a Simplex + tree capabilities
- **Key Methods**: 
  - `_add_splitting_point(point)` - Creates children by replacing vertices with point
  - `add_point_to_the_most_specific_simplex(point)` - Finds deepest containing simplex and splits it

### 3. **Tree Node Wrappers (Two Methodologies)**

#### **ParentChildSimplexTreeNode**
- **Purpose**: Wrapper around SimplexTree using parent-child list structure
- **Holds**: One SimplexTree + list of child nodes
- **Structure**: Each node has a list of children
- **Use Case**: When you need easy access to all children
- **Key Method**: `find_containing_simplex(point)` - Recursive point location

#### **LeftChildRightSiblingSimplexTreeNode**
- **Purpose**: Wrapper around SimplexTree using left child-right sibling structure
- **Holds**: One SimplexTree + left_child + right_sibling pointers
- **Structure**: Each node has one left child and one right sibling
- **Use Case**: More memory efficient, good for binary-like operations
- **Key Method**: `_get_children()` - Traverses sibling chain to collect all children

### 4. **Visualization (Utilities)**
- **Purpose**: Interactive 3D visualization of simplex trees
- **Function**: `visualize_simplex_tree()` - Creates matplotlib 3D plots
- **Features**: Color-coded hierarchy, transparent overlapping, splitting point highlighting

## Data Flow Example

```
1. Create root SimplexTree with vertices [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
   ↓
2. SimplexTree inherits all Simplex functionality (_point_inside_simplex, _embed_point)
   ↓
3. Add splitting point (0.3, 0.4, 0.3) using add_point_to_the_most_specific_simplex()
   ↓
4. This creates 4 children by replacing each vertex with the splitting point
   ↓
5. Wrap in ParentChildSimplexTreeNode or LeftChildRightSiblingSimplexTreeNode
   ↓
6. Each node can:
   - Check if it's a SimplexTree (has children) or Simplex (no children)
   - Test if a point is inside using _point_inside_simplex()
   - Traverse recursively to find containing simplex
   ↓
7. Visualize the entire tree structure with 3D interactive plot
```

## Key APIs

### **Node Type Checking**
```python
node.is_simplex_tree()  # True if this node contains a SimplexTree with children
node.is_simplex()       # True if this node contains a Simplex (no children)
```

### **Point Location**
```python
# Check if point is in this node
node.point_inside_node(point)

# Find containing simplex recursively
containing_simplex = node.find_containing_simplex(point)
```

### **Tree Traversal**
```python
# Get all children
children = node._get_children()

# Check if leaf
_is_leaf = node._is_leaf()

# Get child count
count = node.get_child_count()
```

### **Hierarchical Subdivision**
```python
# Add splitting point to create children
children = simplex_tree._add_splitting_point(point)

# Add point to most specific (deepest) containing simplex
children = simplex_tree.add_point_to_the_most_specific_simplex(point)
```

## Memory Layout

```
SimplexTree (contains Simplex data + tree pointers)
    ├── Simplex properties (vertices, matrices, etc.)
    └── Tree properties (children, parent, depth)

Tree Node Wrapper (contains SimplexTree + tree structure)
    ├── simplex_tree: SimplexTree
    └── Tree structure (children list OR left_child/right_sibling)
```

## Implementation Details

### **Barycentric Coordinate Calculation**
The `_embed_point()` method in Simplex class:
1. Sets first vertex as origin
2. Builds transformation matrix A from remaining vertices
3. Solves A⁻¹(P - V₀) for barycentric coordinates
4. Returns (α₀, α₁, α₂, ...) where α₀ = 1 - Σαᵢ

### **Point-in-Simplex Testing**
The `_point_inside_simplex()` method:
1. Computes barycentric coordinates via `_embed_point()`
2. Checks all coordinates are in [0,1]
3. Verifies sum equals 1 (within tolerance)
4. Returns True if all conditions met

### **Tree Subdivision**
The `_add_splitting_point()` method:
1. Verifies point is inside simplex
2. Creates n+1 children by replacing each vertex with the point
3. Each child simplex shares the splitting point as one vertex
4. Maintains hierarchical structure with proper parent-child relationships

This architecture allows you to:
1. **Use SimplexTree directly** for simple cases
2. **Wrap in tree nodes** for complex hierarchical structures
3. **Choose between two tree methodologies** based on your needs
4. **Build hierarchical subdivisions** by adding splitting points
5. **Traverse recursively** to find point locations
6. **Visualize complex structures** with interactive 3D plots 