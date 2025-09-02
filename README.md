# SimplexTreeClassifier

A machine learning classifier that uses hierarchical simplex tree structures for feature representation and classification. This project implements a novel approach to data classification by embedding data points into barycentric coordinate spaces defined by simplex trees.

## 🎯 Overview

The SimplexTreeClassifier transforms data points into sparse barycentric coordinate representations using hierarchical simplex subdivisions. This creates a consistent feature space where machine learning algorithms can operate on spatial relationships within the simplex tree structure.

## 🏗️ Architecture

### Core Components

- **SimplexTree**: Hierarchical tree structure of 2D simplices (triangles)
- **SimplexTreeClassifier**: Main classifier that uses simplex trees for feature transformation
- **Barycentric Embedding**: Converts data points to barycentric coordinates within containing simplices
- **Sparse Matrix Representation**: Efficient storage of barycentric coordinates across all vertices

### Key Features

- **Hierarchical Subdivision**: Recursive barycentric center subdivision for adaptive resolution
- **Sparse Feature Representation**: Only non-zero coordinates for vertices in containing simplex
- **Consistent Column Mapping**: Each matrix column always represents the same vertex
- **Visualization Support**: Built-in plotting of simplex trees with data points
- **Multiple Classifiers**: Support for SVC and LinearSVC backends

## 📊 How It Works

### 1. Simplex Tree Construction
```python
# Create initial triangle
classifier = SimplexTreeClassifier(vertices=[(0,0), (1,0), (0,1)], subdivision_levels=2)
```

### 2. Barycentric Coordinate Transformation
Each data point is embedded into barycentric coordinates:
- Find containing simplex in the tree
- Compute barycentric coordinates (weights sum to 1.0)
- Map to global vertex indices in sparse matrix

### 3. Feature Matrix
```
Matrix content:
[0.1 0.0 0.6 0.3]  # Point 1: 10% vertex 0, 60% vertex 2, 30% vertex 3
[0.4 0.3 0.3 0.0]  # Point 2: 40% vertex 0, 30% vertex 1, 30% vertex 2
[0.0 0.6 0.3 0.1]  # Point 3: 60% vertex 1, 30% vertex 2, 10% vertex 3
```

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd SimplexTreeClassifier
pip install -r requirements.txt
```

### Basic Usage
```python
from in2D.classifying.classes.simplexTreeClassifier import SimplexTreeClassifier
import numpy as np

# Create classifier with subdivision level 2
classifier = SimplexTreeClassifier(
    vertices=[(0,0), (1,0), (0,1)], 
    subdivision_levels=2
)

# Prepare your data
data_points = np.array([
    [0.5, 0.2],
    [0.1, 0.4], 
    [0.2, 0.7]
])

# Transform data to barycentric coordinates
transformed_matrix = classifier.transform(data_points)

# Visualize the simplex tree with data points
classifier.visualize_with_data_points(data_points)

# Train classifier
X_train = np.array([[0.3, 0.3], [0.7, 0.1], [0.1, 0.8]])
y_train = np.array([0, 1, 0])
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(data_points)
```

## 📁 Project Structure

```
SimplexTreeClassifier/
├── in2D/
│   ├── classifying/
│   │   └── classes/
│   │       └── simplexTreeClassifier.py  # Main classifier
│   └── embadding/
│       ├── classes/
│       │   ├── simplexTree.py           # Tree structure
│       │   └── simplex.py              # Base simplex class
│       └── utilss/
│           └── visualization.py        # Plotting utilities
├── docs/                               # Documentation
└── README.md
```

## 🔧 API Reference

### SimplexTreeClassifier

#### Constructor
```python
SimplexTreeClassifier(
    vertices: List[Tuple[float, float]] = None,
    regularization: float = 1.0,
    subdivision_levels: int = 1,
    classifier_type: str = 'svc'
)
```

#### Key Methods

**`transform(data_points)`**
- Transforms data points to barycentric coordinate matrix
- Returns sparse matrix where each row is a data point, each column is a vertex

**`fit(X, y)`**
- Trains the classifier on transformed barycentric coordinates
- Normalizes data and fits underlying SVM classifier

**`predict(X)`**
- Makes predictions on new data points
- Automatically transforms and normalizes input data

**`visualize_with_data_points(data_points, title, figsize)`**
- Plots simplex tree with data points overlaid
- Customizable figure size and title

**`get_vertex_mapping()`**
- Returns dictionary mapping column indices to vertex coordinates
- Useful for understanding which vertex each matrix column represents

### SimplexTree

#### Key Methods

**`print_tree()`**
- Displays ASCII tree structure showing hierarchical relationships
- Shows vertex coordinates for each simplex

**`traverse_breadth_first()`**
- Iterates through all nodes in breadth-first order
- Used for consistent vertex ordering

**`find_containing_simplex(point)`**
- Finds the leaf simplex containing a given point
- Returns None if point is outside all simplices

## 🎨 Visualization

The classifier provides rich visualization capabilities:

```python
# Basic visualization
classifier.visualize_with_data_points(data_points)

# Custom size
classifier.visualize_with_data_points(data_points, figsize=(12, 8))

# Print tree structure
classifier.tree.print_tree()
```

Example tree output:
```
└── (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)
    ├── (0.0, 0.0), (1.0, 0.0), (0.3, 0.3)
    ├── (1.0, 0.0), (0.0, 1.0), (0.3, 0.3)
    └── (0.0, 1.0), (0.0, 0.0), (0.3, 0.3)
```

## 🧮 Mathematical Foundation

### Barycentric Coordinates
For a point P inside triangle ABC, barycentric coordinates (α, β, γ) satisfy:
- P = αA + βB + γC
- α + β + γ = 1
- α, β, γ ≥ 0

### Sparse Representation
The transformation creates a sparse matrix where:
- **Rows**: Data points
- **Columns**: All vertices in the simplex tree
- **Values**: Barycentric coordinates (most entries are 0)
- **Sparsity**: Only 3 non-zero entries per row (for triangle simplices)

## 🔍 Understanding the Matrix

For a row like `[0.4 0.3 0.3 0.0]`:
- **0.4**: 40% weight toward vertex at column 0
- **0.3**: 30% weight toward vertex at column 1  
- **0.3**: 30% weight toward vertex at column 2
- **0.0**: 0% weight toward vertex at column 3 (not in containing simplex)

Use `get_vertex_mapping()` to see which vertex each column represents.

## ⚙️ Configuration Options

### Subdivision Levels
- **Level 0**: Original triangle only
- **Level 1**: 3 child triangles
- **Level 2**: 9 total triangles
- **Level n**: 3^n total triangles

### Classifier Types
- **'svc'**: Support Vector Classifier with linear kernel
- **'linear_svc'**: Linear Support Vector Classifier

### Regularization
- Controls SVM regularization parameter C
- Higher values = less regularization, more complex decision boundaries

## 🧪 Example Workflows

### 1. Basic Classification
```python
# Create and train classifier
classifier = SimplexTreeClassifier(subdivision_levels=2)
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

### 2. Data Analysis
```python
# Transform data and analyze
transformed = classifier.transform(data_points)
vertex_mapping = classifier.get_vertex_mapping()

# Check which vertices influence each point
for i, row in enumerate(transformed.toarray()):
    nonzero_indices = np.where(row > 0)[0]
    print(f"Point {i} influenced by vertices: {nonzero_indices}")
```

### 3. Visualization
```python
# Show tree structure
classifier.tree.print_tree()

# Plot with data points
classifier.visualize_with_data_points(data_points, "My Analysis")
```

## 🔬 Research Applications

This classifier is particularly useful for:
- **Spatial Data Analysis**: When data has geometric structure
- **Hierarchical Feature Learning**: Multi-resolution representations
- **Sparse Feature Spaces**: When interpretability is important
- **Geometric Machine Learning**: Applications requiring spatial reasoning

## 📈 Performance Considerations

- **Memory**: Sparse matrices are memory-efficient for large trees
- **Speed**: BFS traversal ensures consistent vertex ordering
- **Scalability**: Subdivision levels control complexity vs. accuracy trade-off
- **Interpretability**: Each feature corresponds to a specific geometric location

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of scikit-learn for machine learning functionality
- Uses matplotlib for visualization
- Implements barycentric coordinate mathematics for geometric embeddings

---

For more detailed examples and advanced usage, see the `docs/` directory and inline code documentation.
