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
│ • compute_barycentric_coordinates(point) → Optional[Tuple[float, ...]]     │
│ • point_inside_simplex(point) → bool                                        │
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
│ • get_children() → List[SimplexTree]                                        │
│ • is_leaf() → bool                                                          │
│ • get_child_count() → int                                                   │
│ • traverse_breadth_first() → Iterator[SimplexTree]                         │
│ • traverse_depth_first() → Iterator[SimplexTree]                           │
│ • get_leaves() → List[SimplexTree]                                          │
│ • get_nodes_at_depth(target_depth) → List[SimplexTree]                     │
│ • find_containing_simplex(point) → Optional[SimplexTree]                   │
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
│ │ • get_children() → List[ParentChildSimplexTreeNode]                    │ │
│ │ • remove_child(child_node) → bool                                       │ │
│ │ • is_leaf() → bool                                                      │ │
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
│ │ • get_children() → List[LeftChildRightSiblingSimplexTreeNode]          │ │
│ │ • remove_child(child_node) → bool                                       │ │
│ │ • is_leaf() → bool                                                      │ │
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
│                           TreeBuilder (Utilities)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Functions:                                                                  │
│ • create_parent_child_tree_with_steady_branching() → ParentChildSimplexTreeNode │
│ • create_left_child_right_sibling_tree_with_steady_branching() → LeftChildRightSiblingSimplexTreeNode │
│ • analyze_tree_structure(root_node, node_type) → None                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How It Works - Data Flow

### 1. **Simplex (Base Class)**
- **Purpose**: Mathematical representation of a simplex (triangle, tetrahedron, etc.)
- **Holds**: Vertices, transformation matrices, geometric calculations
- **Provides**: Point-in-simplex testing, barycentric coordinates

### 2. **SimplexTree (Main Class)**
- **Purpose**: A Simplex that can have children (other SimplexTrees)
- **Inherits**: ALL properties and methods from Simplex
- **Adds**: Tree functionality (children, traversal, etc.)
- **Key Concept**: Every SimplexTree IS a Simplex + tree capabilities

### 3. **Tree Node Wrappers (Two Methodologies)**

#### **ParentChildSimplexTreeNode**
- **Purpose**: Wrapper around SimplexTree using parent-child list structure
- **Holds**: One SimplexTree + list of child nodes
- **Structure**: Each node has a list of children
- **Use Case**: When you need easy access to all children

#### **LeftChildRightSiblingSimplexTreeNode**
- **Purpose**: Wrapper around SimplexTree using left child-right sibling structure
- **Holds**: One SimplexTree + left_child + right_sibling pointers
- **Structure**: Each node has one left child and one right sibling
- **Use Case**: More memory efficient, good for binary-like operations

### 4. **TreeBuilder (Utilities)**
- **Purpose**: Creates trees with steady branching factors
- **Function**: Takes a subdivision function and builds hierarchical trees
- **Branching Factor**: Equal to the dimension of the simplex (2 for triangles, 3 for tetrahedra)

## Data Flow Example

```
1. Create root SimplexTree with vertices [(0,0), (4,0), (2,3)]
   ↓
2. SimplexTree inherits all Simplex functionality
   ↓
3. Wrap in ParentChildSimplexTreeNode or LeftChildRightSiblingSimplexTreeNode
   ↓
4. Use TreeBuilder to create steady branching structure
   ↓
5. Each node can:
   - Check if it's a SimplexTree (has children) or Simplex (no children)
   - Test if a point is inside using point_inside_simplex()
   - Traverse recursively to find containing simplex
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
children = node.get_children()

# Check if leaf
is_leaf = node.is_leaf()

# Get child count
count = node.get_child_count()
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

This architecture allows you to:
1. **Use SimplexTree directly** for simple cases
2. **Wrap in tree nodes** for complex hierarchical structures
3. **Choose between two tree methodologies** based on your needs
4. **Maintain steady branching** equal to simplex dimension
5. **Traverse recursively** to find point locations 