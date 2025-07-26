# SimplexTreeClassifier: Hierarchical Simplex Point Location

A comprehensive Python implementation for hierarchical simplex point location in both 2D and 3D spaces, featuring multiple tree data structures and efficient point location algorithms.

## 🎯 Overview

This project implements a sophisticated simplex tree classifier that can:
- **Locate points within simplexes** (triangles in 2D, tetrahedra in 3D) using barycentric coordinates
- **Build hierarchical tree structures** for efficient spatial queries
- **Support multiple tree implementations** (parent-child lists and left-child-right-sibling)
- **Handle both 2D and 3D geometries** with unified APIs
- **Visualize complex tree structures** with interactive 3D plots

## 🏗️ Architecture

### Core Components

#### 1. **Simplex (Base Class)**
- Mathematical representation of simplexes (triangles, tetrahedra, etc.)
- Barycentric coordinate calculations
- Point-in-simplex testing with proper matrix operations
- Handles degenerate cases gracefully

#### 2. **SimplexTree (Hierarchical Structure)**
- Inherits all Simplex functionality
- Adds tree capabilities (children, traversal, depth tracking)
- Supports recursive point location
- Enables hierarchical subdivision

#### 3. **Tree Node Wrappers**
- **ParentChildSimplexTreeNode**: Traditional parent-child list structure
- **LeftChildRightSiblingSimplexTreeNode**: Memory-efficient left-child-right-sibling structure
- Both provide unified APIs for tree operations

#### 4. **Visualization Tools**
- 3D interactive visualizations using matplotlib
- Color-coded hierarchical representations
- Support for complex tree structures

## 📁 Project Structure

```
SimplexTreeClassifier/
├── in2D/                          # 2D implementations
│   ├── simplex_point_location.py  # Basic 2D point location
│   ├── tree_of_simplexes.py       # 2D tree structures
│   ├── visualize_example.py       # 2D visualization
│   └── test_additional_points.py  # 2D test cases
├── in3D/                          # 3D implementations
│   ├── classes/                   # Core classes
│   │   ├── simplex.py            # Base Simplex class
│   │   ├── simplexTree.py        # Hierarchical SimplexTree
│   │   ├── parentChildSimplexTreeNode.py
│   │   ├── leftChildRightSiblingSimplexTreeNode.py
│   │   └── simplexTreeImport.py
│   └── utilss/
│       └── visualization.py      # 3D visualization tools
└── docs/                         # Documentation
    ├── README.md
    ├── class_hierarchy_diagram.md
    ├── tree_structure_visualization.md
    ├── requirements.txt
    ├── example_usage.py
    └── images/                   # Documentation images
        ├── class_hierarchy.png
        ├── 3d_tree_example.png
        └── 2d_point_location.png
```

## 🚀 Quick Start

### Basic 2D Point Location

```python
from in2D.simplex_point_location import Triangle, SimplexPointLocator

# Create triangles
triangle1 = Triangle((0, 0), (4, 0), (2, 2))
triangle2 = Triangle((0, 0), (2, 2), (0, 4))

# Create locator
locator = SimplexPointLocator([triangle1, triangle2])

# Find containing triangle
point = (2, 0.5)
result = locator.find_containing_simplex_with_coords(point)

if result:
    index, triangle, coords = result
    print(f"Point {point} is in triangle {index}")
    print(f"Barycentric coordinates: {coords}")
```

### 3D Hierarchical Tree

```python
from in3D.classes.simplexTree import SimplexTree
from in3D.classes.parentChildSimplexTreeNode import ParentChildSimplexTreeNode

# Create 3D tetrahedron
vertices_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
tree_3d = SimplexTree(vertices_3d)

# Add splitting points to create hierarchy
test_point = (0.3, 0.4, 0.3)
tree_3d.add_point_to_the_most_specific_simplex(test_point)

# Wrap in tree node for advanced operations
root_node = ParentChildSimplexTreeNode(tree_3d)

# Find containing simplex
containing_simplex = root_node.find_containing_simplex(test_point)
print(f"Containing simplex: {containing_simplex}")
```

### Tree Node Operations

```python
# Check node types
if node.is_simplex_tree():
    print("This node contains a SimplexTree with children")
    # Traverse children recursively
    for child in node.get_children():
        result = child.find_containing_simplex(point)
        if result:
            return result

if node.is_simplex():
    print("This node contains a Simplex (leaf node)")
    if node.point_inside_node(point):
        return node.simplex_tree
```

## 🔬 Mathematical Foundation

### Barycentric Coordinates
For a point P in simplex with vertices V₀, V₁, V₂, ..., Vₙ, the barycentric coordinates (α₀, α₁, α₂, ..., αₙ) satisfy:
- P = α₀V₀ + α₁V₁ + α₂V₂ + ... + αₙVₙ
- α₀ + α₁ + α₂ + ... + αₙ = 1
- P is inside the simplex iff all αᵢ ∈ [0,1]

### Matrix Formulation
Setting V₀ as origin, we solve:
```
[V₁-V₀, V₂-V₀, ..., Vₙ-V₀] [α₁] = P - V₀
                            [α₂]
                            [...]
                            [αₙ]
```

Then α₀ = 1 - α₁ - α₂ - ... - αₙ.

## 🌳 Tree Structures

### Steady Branching Factor
- **2D Triangle**: Branching factor = 2
- **3D Tetrahedron**: Branching factor = 3
- **nD Simplex**: Branching factor = n

### Tree Node Types

#### Parent-Child List Structure
```python
ParentChildSimplexTreeNode
├── simplex_tree: SimplexTree
├── children: List[ParentChildSimplexTreeNode]
├── parent: Optional[ParentChildSimplexTreeNode]
└── depth: int
```

#### Left Child-Right Sibling Structure
```python
LeftChildRightSiblingSimplexTreeNode
├── simplex_tree: SimplexTree
├── left_child: Optional[LeftChildRightSiblingSimplexTreeNode]
├── right_sibling: Optional[LeftChildRightSiblingSimplexTreeNode]
├── parent: Optional[LeftChildRightSiblingSimplexTreeNode]
└── depth: int
```

## 📊 Visualization

### 3D Interactive Visualization
```python
from in3D.utilss.visualization import visualize_simplex_tree

# Visualize tree with splitting point
visualize_simplex_tree(tree_3d, test_point, "3D Tetrahedron Splitting")
```

Features:
- Color-coded hierarchical levels
- Interactive 3D rotation and zoom
- Highlighted splitting points
- Transparent overlapping simplexes

## ⚡ Performance Features

### Optimizations Implemented
1. **Matrix caching** - Transformation matrices computed once and reused
2. **Degenerate case handling** - Graceful handling of collinear/coplanar vertices
3. **Efficient tree traversal** - Breadth-first and depth-first traversal options
4. **Memory-efficient structures** - Left-child-right-sibling option for large trees

### Future Optimizations
1. **Bounding box pre-check** - Reject simplexes whose bounding box doesn't contain P
2. **Walking algorithm** - Walk from one simplex to neighbor toward P
3. **Spatial indexing** - Quadtree/k-d tree for larger simplex sets

## 📋 Requirements

- Python 3.7+
- NumPy 1.21.0+
- Matplotlib 3.5.0+ (for visualization)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YasminAdler/SimplexTreeClassifier.git
cd SimplexTreeClassifier

# Install dependencies
pip install -r docs/requirements.txt
```

## 🧪 Running Examples

```bash
# 2D examples
python in2D/simplex_point_location.py
python in2D/test_additional_points.py
python in2D/visualize_example.py

# 3D examples
python in3D/classes/simplexTree.py
```

## 🔍 Key Features

- **✅ Correct barycentric coordinate calculation** - Uses proper matrix inversion
- **✅ Proper containment testing** - Checks that all αᵢ ∈ [0,1] (not just ≥ 0)
- **✅ Handles degenerate cases** - Gracefully handles collinear/coplanar vertices
- **✅ Multiple search methods** - Find simplex with/without coordinates
- **✅ Hierarchical structures** - Tree-based organization for complex geometries
- **✅ Multiple tree implementations** - Choose based on memory/performance needs
- **✅ 2D and 3D support** - Unified APIs across dimensions
- **✅ Interactive visualization** - 3D plots with rotation and zoom
- **✅ Comprehensive testing** - Extensive test cases and examples

## 🤝 Contributing

This project welcomes contributions! Please see the documentation for:
- Class hierarchy diagrams
- Tree structure visualizations
- API reference
- Performance benchmarks

## 📄 License

This project is open source and available under the MIT License. 