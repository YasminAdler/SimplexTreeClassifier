# simplex-tree-classifier

A hierarchical **simplex-tree classifier**. It embeds points into the barycentric
coordinates of a subdivided simplex, then trains any scikit-learn estimator on
that sparse, geometry-aware feature space. The barycentric `transform` is
accelerated with **PyTorch** and runs on the GPU when one is available.

It works in two ways:

- **Dataset mode** - train directly on your data `(X, y)`.
- **Surrogate mode** - imitate an existing `model` (e.g. a neural network) to
produce an interpretable, piecewise-linear geometric surrogate.

Everything is **N-dimensional**.

## Installation

```bash
pip install simplex-tree-classifier          # core (numpy, scipy, scikit-learn, torch)
pip install simplex-tree-classifier[viz]      # + matplotlib for 2-D plots
```

From source:

```bash
cd simplex_tree_package
pip install -e .
```



## Quickstart



### Dataset mode

```python
import numpy as np
from simplex_tree_classifier import SimplexTreeClassifier

X = np.random.uniform(0, 1, size=(500, 4))
y = (X.sum(axis=1) > 2.0).astype(int)

clf = SimplexTreeClassifier(subdivision_levels=2)   # default classifier: LinearSVC
clf.fit(X, y)
print(clf.predict(X[:5]))
```



### Surrogate mode (imitate a model)

Pass any object with `.predict(X)` or any callable `X -> labels`, plus the input
dimensionality. Fill points are sampled in `[0, 1]^d` and labeled by the model.

```python
def black_box(X):                      # e.g. a wrapped PyTorch model
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] > 1.0).astype(int)

surrogate = SimplexTreeClassifier(
    model=black_box,
    n_features=2,
    subdivision_levels=4,
    n_fill=5000,
)
surrogate.fit()                        # samples + labels internally
```



### Data-driven subdivision

Instead of uniform (barycentric) subdivision, split only where the classifier is
still wrong:

```python
clf = SimplexTreeClassifier(
    subdivision_strategy="data_driven",
    subdivision_levels=6,     # here: maximum tree depth
    max_leaves=500,
)
clf.fit(X, y)
```



## API overview

Constructor: `SimplexTreeClassifier(classifier=None, model=None, n_features=None, subdivision_levels=1, subdivision_strategy="barycentric", n_fill=5000, max_leaves=None, normalize=None, margin=0.05, device=None, tolerance=1e-10, random_state=None, data_driven_max_iter=50)`

Core methods:

- `transform(X)` - sparse barycentric embedding (GPU-accelerated); rows sum to 1.
- `fit(X=None, y=None)` - trains the classifier. In surrogate mode `X`/`y` are
optional. Automatically records same-side simplices afterward (linear only).
- `predict(X)` - predicted labels.

Geometry queries:

- `find_containing_simplex(point)` - the leaf simplex a point falls in.
- `find_adjacent_simplexes(simplex)` - leaves sharing a `(d-1)`-face.
- `get_simplex_vertices()` - vertices of every leaf simplex (renamed from
`get_simplex_boundaries`).
- `is_in_simplex(point, simplex)` - boolean containment test.

Decision-boundary analysis (linear classifiers, multiclass-aware):

- `identify_crossing_simplices()` - leaves crossed by the decision boundary.
- `remove_nonconvex_leaves(criterion="distance"|"convexity_sign", removal_factor=0.15, epsilon=None, keep_frac=None, min_depth=0.0, max_remove_frac=0.25, max_iter=10, refit=True)` - iteratively prune leaves whose
boundary bends non-convexly.
- `find_same_side_simplices()` - subdivisions that don't touch the boundary (also
run automatically at the end of `fit`, stored in `same_side_keys_`).
- `compute_plane_equations()` - the boundary hyperplane within each crossing leaf.

Helpers: `SimplexTree`, `Simplex`, `VertexRegistry`, `make_enclosing_simplex(d)`,
`get_device()`.

## GPU acceleration

`transform` stacks each leaf's inverse edge matrix into tensors and embeds an
entire batch of points against all leaves at once. The device is auto-detected
(CUDA if present); override with `device="cpu"` / `device="cuda"`. Points outside
every leaf (or inside a degenerate one) fall back to an exact per-point search.

## TODO

- [ ] **Overleaf cross-check.** Reconcile this package's operations against the
  list of supported operations in the paper (Overleaf) and add anything missing.

