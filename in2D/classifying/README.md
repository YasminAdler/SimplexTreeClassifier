# SimplexTree Classifier

A geometric classification method that transforms 2D spatial data into high-dimensional barycentric feature space using hierarchical simplex trees.

## Overview

The SimplexTree Classifier uses a two-stage approach:
1. **Spatial Partitioning**: Creates hierarchical simplex trees through recursive barycentric subdivision
2. **Feature Transformation**: Maps points to sparse barycentric coordinates for linear classification

## Data Requirements

- **Input**: Data points as vectors of size **D** (dimension)
  - Example in 2D: `(0.5, 0.3)`
  - Example in 4D: `(1, 1, 1, 0)`
- **Preprocessing**: Data must be normalized to fit within initial simplex
  - For 2D: Triangle `[(0,0), (1,0), (0,1)]`
  - Points outside the simplex are filtered out

## Classification Pipeline

### Stage 1: Tree Construction
1. Start with initial simplex (triangle in 2D)
2. Apply recursive barycentric subdivision for `h` levels
3. Generate `(d+1)^h` leaf simplexes

### Stage 2: Data Transformation
1. **Point Location**: For each normalized point, find which leaf simplex contains it
2. **Barycentric Embedding**: Compute barycentric coordinates within that simplex
3. **Sparse Matrix**: Create global feature matrix where each row represents point's coordinates

### Stage 3: Classification
1. **Feature Vector**: Transform to global feature space of size `(d+1)^(h+1)`
2. **Sparse Representation**: Each point activates only `(d+1)` features (non-zero entries)
3. **Linear Classifier**: Apply single global classifier (SVM, Linear Regression) to transformed features

## Mathematical Formulas

### Tree Structure Parameters
| Parameter | Formula | Description |
|-----------|---------|-------------|
| **h** | User defined | Tree height (subdivision levels, starting from 0) |
| **d** | User defined | Spatial dimension |
| **b** | `d + 1` | Branching factor |
| **s** | `(b^h - 1) / (b - 1)` | Number of splitting points |
| **q** | `s × b + 1` | Total number of simplexes |
| **leaves** | `(d + 1)^h = b^h` | Number of leaf simplexes |

### Feature Space Size
- **Global Feature Vector Size**: `(d+1)^(h+1)`
- **Active Features per Point**: `d+1` (sparse representation)
- **Sparsity**: `~((d+1) / (d+1)^(h+1)) × 100%`

## Examples

### Example 1: 2D, Height 2
```
d = 2, b = 3, h = 2
s = (3² - 1) / 2 = 4 splitting points
q = 4 × 3 + 1 = 13 total simplexes
leaves = 3² = 9 leaf simplexes
Feature vector size = 3³ = 27 features
```

### Example 2: 2D, Height 1
```
d = 2, b = 3, h = 1
s = (3¹ - 1) / 2 = 1 splitting point
q = 1 × 3 + 1 = 4 total simplexes
leaves = 3¹ = 3 leaf simplexes
Feature vector size = 3² = 9 features
```

## Implementation Details

### Supported Classifiers
- Support Vector Machine (SVM)
- Linear SVM
- Linear Regression

### Key Features
- **Sparse Matrix Representation**: Memory-efficient storage using `scipy.sparse.lil_matrix`
- **Geometric Visualization**: 2D plotting of simplex boundaries and decision surfaces
- **Scalable Architecture**: Exponential feature growth with subdivision levels

## Usage Example

```python
from simplex_tree_classifier import SimplexTreeClassifier

# Create classifier
classifier = SimplexTreeClassifier(
    vertices=[(0, 0), (1, 0), (0.5, 1)],
    regularization=0.1,
    subdivision_levels=2,
    classifier_type='svc'
)

# Train
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Visualize
classifier.visualize_tree_and_classification(X_train, y_train)
```

## Future Work

- **N-Dimensional Extension**: Generalize from 2D triangles to n-dimensional simplexes
- **Adaptive Subdivision**: Dynamic tree construction based on data distribution
- **Performance Optimization**: Efficient algorithms for high-dimensional spaces

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `ucimlrepo` (for dataset loading)