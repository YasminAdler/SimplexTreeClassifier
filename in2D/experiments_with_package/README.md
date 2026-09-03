# Paper experiments, running on the published package

These are the four paper notebooks from `to_save_for_paper/experiments_to_use`,
rewritten so that **everything simplex-tree related goes through the installed
`simplex-tree-classifier` package** instead of the in-repo `in2D` code.

```
diabetes_experiments.ipynb    binary  (8 features)
vehicle_experiments.ipynb     4-class (18 features)
wine_experiments.ipynb        multiclass
phoneme_experiments.ipynb     binary
nn_utils.py                   NN + LIME + metrics (NOT part of the package)
verify_package_experiment.py  fast end-to-end smoke of Experiment 1
models/                       disk cache (empty; notebooks repopulate it)
```

## How to run

```bash
cd in2D/experiments_with_package
python verify_package_experiment.py      # ~1-2 min, real openml data
# or open any *_experiments.ipynb with the myvenv kernel and Run All
```

The package is already installed in `myvenv` (`import simplex_tree_classifier`).
`nn_utils.py` is imported from this folder.

## What changed vs the originals

| Original (`in2D` / `paper_experiments_utils`) | Now (package) |
|---|---|
| `from in2D.classifying.classes.simplex_tree_classifier import SimplexTreeClassifier` | `from simplex_tree_classifier import SimplexTreeClassifier` |
| `SimplexTreeClassifier(vertices=create_simplex_vertices(n_features), ...)` | `SimplexTreeClassifier(n_features=n_features, margin=0.0, normalize=False, ...)` |
| `create_simplex_vertices(d)` (scale = d) | built internally by `make_enclosing_simplex(d, margin=0)` |
| `stc.find_nonconvex_simplexes(epsilon=E)` → set of keys | `set(stc.find_nonconvex_leaves(criterion="convexity_sign", epsilon=E).keys())` |
| `paper_experiments_utils.find_nonconvex_leaves(stc, epsilon=E)` | `stc.find_nonconvex_leaves(criterion="convexity_sign", epsilon=E)` |
| `paper_experiments_utils.build_convex_surrogate(mk, X, y, ...)` | `m = mk(); m.fit(X, y); m.remove_nonconvex_leaves(criterion="convexity_sign", ...)` |
| `stc.tree.get_leaves()`, `stc.tree.remove_by_leaf_key(k)`, `stc.identify_crossing_simplices()`, `stc.find_same_side_simplices()`, `stc.find_adjacent_simplexes(l)`, `stc.transform(X)`, `stc.classifier.decision_function(...)` | **unchanged** – all exist on the package |

Two important defaults were made explicit in every constructor:

* `margin=0.0` — matches the old `create_simplex_vertices` (axis intercept = `d`).
  The package default is `margin=0.05`; both enclose `[0,1]^d`, results are
  equivalent.
* `normalize=False` — the notebooks already scale features to `[0,1]`, and they
  call `stc.transform(...)` directly (bypassing `predict`), so normalization must
  be off for `transform` and `predict` to agree.
* `criterion="convexity_sign"` — reproduces the paper's geometric sign test.
  The package's default `criterion="distance"` is a *stricter* magnitude test and
  would flag fewer leaves.

## What lives in `nn_utils.py` (and why it is NOT in the package)

`nn_utils.py` holds only the experiment scaffolding that a classifier library
should not ship:

* **NN** – `FeatureMLP`, `train_nn`, `predict_classes`, `predict_proba` (the
  black box being explained; a plain feed-forward net on all raw features - no
  GRU, no recurrence, no reduction).
* **LIME** – `make_lime_explainer`, `lime_direction`, `push_along_lime`.
* **metrics** – `gen_coef`.

No feature reduction is used anywhere: every experiment feeds all raw features to
the network (no PCA, no RFE, no feature selection).

The three tree helpers still in `nn_utils.py`
(`find_nonconvex_leaves`, `build_convex_surrogate`, `leaf_key_for_point`) are
**thin forwarders** – they contain no simplex geometry, they only call package
methods. They exist so the notebooks read unchanged; you can drop them and call
the package methods directly.

## Nothing was missing to run the experiments

Every simplex-tree operation the experiments need is available on the package.
The full Experiment-1 chain (surrogate fit → `identify_crossing_simplices` →
`find_nonconvex_leaves` → `remove_nonconvex_leaves` → `decision_function` →
pickle cache) was verified end-to-end on real `vehicle` data
(`verify_package_experiment.py`).

## Suggested (optional) additions to the package

These are quality-of-life parity items, not blockers:

1. **Accept custom base vertices in the constructor.** The old API took
   `vertices=`. The package builds the simplex from `n_features`. Adding an
   optional `vertices=`/`base_simplex=` (falling back to `make_enclosing_simplex`)
   would make old code a drop-in and allow non-default root simplices.
2. **`find_nonconvex_simplexes` alias.** A one-liner returning
   `set(find_nonconvex_leaves(...).keys())` would spare callers the `.keys()`
   wrapping and match the name used in the diabetes notebook.
3. **`leaf_key_for_point(point)` convenience method** on the classifier
   (`frozenset(find_containing_simplex(point).vertex_indices)`), since several
   experiments key leaves this way.
4. **Reconsider the default `criterion`.** The paper results use the sign test
   (`convexity_sign`); the package defaults to `distance`. Either is fine, but the
   default determines out-of-the-box behaviour, so it is worth a deliberate choice
   (and documenting it in the README/quickstart).
