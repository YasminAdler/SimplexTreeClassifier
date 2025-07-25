# Tree Structure Visualization

## Example: 2D Triangle with Steady Branching Factor = 2

### Initial Setup
```
Root Triangle: [(0,0), (4,0), (2,3)]
Dimension: 2
Branching Factor: 2 (equal to dimension)
```

### Tree Structure After Subdivision

```
                    Root SimplexTree
                    [(0,0), (4,0), (2,3)]
                    (is_simplex_tree() = True)
                           │
                    ┌──────┴──────┐
                    │             │
              Child 1          Child 2
        [(0,0), (2,0), (1,1.5)]  [(2,0), (4,0), (3,1.5)]
        (is_simplex_tree() = True)  (is_simplex_tree() = True)
                    │             │
              ┌─────┴─────┐   ┌─────┴─────┐
              │           │   │           │
         Child 1.1    Child 1.2    Child 2.1    Child 2.2
    [(0,0), (1,0), (0.5,0.75)]  [(1,0), (2,0), (1.5,0.75)]  [(2,0), (3,0), (2.5,0.75)]  [(3,0), (4,0), (3.5,0.75)]
    (is_simplex() = True)       (is_simplex() = True)       (is_simplex() = True)       (is_simplex() = True)
```

## Parent-Child List Structure

```
ParentChildSimplexTreeNode
├── simplex_tree: SimplexTree([(0,0), (4,0), (2,3)])
├── children: [
│   ├── ParentChildSimplexTreeNode
│   │   ├── simplex_tree: SimplexTree([(0,0), (2,0), (1,1.5)])
│   │   └── children: [
│   │       ├── ParentChildSimplexTreeNode
│   │       │   └── simplex_tree: SimplexTree([(0,0), (1,0), (0.5,0.75)])
│   │       └── ParentChildSimplexTreeNode
│   │           └── simplex_tree: SimplexTree([(1,0), (2,0), (1.5,0.75)])
│   └── ParentChildSimplexTreeNode
│       ├── simplex_tree: SimplexTree([(2,0), (4,0), (3,1.5)])
│       └── children: [
│           ├── ParentChildSimplexTreeNode
│           │   └── simplex_tree: SimplexTree([(2,0), (3,0), (2.5,0.75)])
│           └── ParentChildSimplexTreeNode
│               └── simplex_tree: SimplexTree([(3,0), (4,0), (3.5,0.75)])
```

## Left Child-Right Sibling Structure

```
LeftChildRightSiblingSimplexTreeNode
├── simplex_tree: SimplexTree([(0,0), (4,0), (2,3)])
├── left_child: LeftChildRightSiblingSimplexTreeNode
│   ├── simplex_tree: SimplexTree([(0,0), (2,0), (1,1.5)])
│   ├── left_child: LeftChildRightSiblingSimplexTreeNode
│   │   ├── simplex_tree: SimplexTree([(0,0), (1,0), (0.5,0.75)])
│   │   ├── left_child: None
│   │   └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│   │       ├── simplex_tree: SimplexTree([(1,0), (2,0), (1.5,0.75)])
│   │       ├── left_child: None
│   │       └── right_sibling: None
│   └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│       ├── simplex_tree: SimplexTree([(2,0), (4,0), (3,1.5)])
│       ├── left_child: LeftChildRightSiblingSimplexTreeNode
│       │   ├── simplex_tree: SimplexTree([(2,0), (3,0), (2.5,0.75)])
│       │   ├── left_child: None
│       │   └── right_sibling: LeftChildRightSiblingSimplexTreeNode
│       │       ├── simplex_tree: SimplexTree([(3,0), (4,0), (3.5,0.75)])
│       │       ├── left_child: None
│       │       └── right_sibling: None
│       └── right_sibling: None
└── right_sibling: None
```

## Point Location Algorithm Flow

### Example: Finding point (1.5, 0.5)

```
1. Start at Root Node
   ├── point_inside_node((1.5, 0.5)) → True
   ├── is_simplex_tree() → True (has children)
   └── Check children recursively

2. Check Child 1 [(0,0), (2,0), (1,1.5)]
   ├── point_inside_node((1.5, 0.5)) → True
   ├── is_simplex_tree() → True (has children)
   └── Check children recursively

3. Check Child 1.1 [(0,0), (1,0), (0.5,0.75)]
   ├── point_inside_node((1.5, 0.5)) → False
   └── Skip this branch

4. Check Child 1.2 [(1,0), (2,0), (1.5,0.75)]
   ├── point_inside_node((1.5, 0.5)) → True
   ├── is_simplex() → True (no children)
   └── Return this SimplexTree

Result: Found containing simplex [(1,0), (2,0), (1.5,0.75)]
```

## API Usage Examples

### Creating a Tree
```python
# Create root vertices
root_vertices = [(0, 0), (4, 0), (2, 3)]

# Create parent-child tree
parent_child_root = create_parent_child_tree_with_steady_branching(
    root_vertices, subdivide_triangle, max_depth=2
)

# Create left child-right sibling tree
left_child_right_sibling_root = create_left_child_right_sibling_tree_with_steady_branching(
    root_vertices, subdivide_triangle, max_depth=2
)
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
point = (1.5, 0.5)
containing_simplex = root_node.find_containing_simplex(point)

if containing_simplex:
    print(f"Point {point} is in simplex with vertices: {containing_simplex.vertices}")
    # Get barycentric coordinates
    coords = containing_simplex.compute_barycentric_coordinates(point)
    print(f"Barycentric coordinates: {coords}")
else:
    print(f"Point {point} is not in any simplex")
```

## Memory Efficiency Comparison

### Parent-Child List
- **Pros**: Easy to access all children, intuitive structure
- **Cons**: Each node stores a list of children (more memory)
- **Use Case**: When you need frequent access to all children

### Left Child-Right Sibling
- **Pros**: More memory efficient, good for binary-like operations
- **Cons**: More complex traversal, need to follow sibling chain
- **Use Case**: When memory is a concern or you need binary-like operations

## Steady Branching Factor

The branching factor is always equal to the dimension of the simplex:
- **2D Triangle**: Branching factor = 2
- **3D Tetrahedron**: Branching factor = 3
- **4D Simplex**: Branching factor = 4

This ensures consistent tree structure regardless of the simplex type. 