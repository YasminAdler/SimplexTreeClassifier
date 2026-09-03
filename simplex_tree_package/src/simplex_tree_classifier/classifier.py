"""The public :class:`SimplexTreeClassifier`.

Two ways to construct it:

* **Dataset mode** - pass a scikit-learn style ``classifier`` and call
  ``fit(X, y)``. The classifier learns from the barycentric embedding of your
  data.
* **Surrogate mode** - pass a ``model`` (any object with ``.predict`` or any
  callable ``X -> labels``) plus ``n_features``. ``fit()`` samples points in the
  unit hypercube ``[0, 1]^d``, labels them with the model, and learns to imitate
  it - a geometric surrogate.

Subdivision of the simplex tree can be ``"barycentric"`` (uniform) or
``"data_driven"`` (split only where the classifier still makes mistakes).

The barycentric ``transform`` is accelerated with PyTorch and runs on the GPU
when one is available.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

from .simplex_tree import SimplexTree, make_enclosing_simplex
from .backend import TransformBackend
from .plane_equation import PlaneEquation
from .convexity import (
    check_convexity,
    meeting_to_average_distance,
    shared_face_length,
)


class SimplexTreeClassifier:
    # Class-level defaults so instances unpickled from an older version (whose
    # __dict__ predates these attributes) resolve them to None instead of
    # raising AttributeError on access.
    _X_fit: Optional[np.ndarray] = None
    _y_fit = None

    def __init__(self,
                 classifier=None,
                 model=None,
                 n_features: Optional[int] = None,
                 subdivision_levels: int = 1,
                 subdivision_strategy: str = "barycentric",
                 n_fill: Optional[int] = None,
                 max_leaves: Optional[int] = None,
                 normalize: Optional[bool] = None,
                 margin: float = 0.05,
                 device=None,
                 tolerance: float = 1e-10,
                 random_state: Optional[int] = None,
                 data_driven_max_iter: int = 50):
        """Create a simplex-tree classifier.

        Args:
            classifier: Any sklearn-compatible estimator with ``.fit()`` /
                ``.predict()``. Defaults to ``LinearSVC(C=1.0)``. This is the
                estimator that learns from the barycentric features (used in both
                dataset and surrogate modes).
            model: If given, the classifier runs in *surrogate* mode and learns to
                imitate this model. May be an object exposing ``.predict(X)`` or a
                plain callable mapping ``X -> labels``. Requires ``n_features``.
            n_features: Input dimensionality ``d`` (required in surrogate mode;
                optional otherwise - inferred from ``X`` at ``fit`` time).
            subdivision_levels: For ``"barycentric"`` strategy, the number of
                uniform subdivision levels. For ``"data_driven"`` strategy, the
                maximum tree depth.
            subdivision_strategy: ``"barycentric"`` (uniform) or ``"data_driven"``
                (split leaves that still contain misclassified points).
            n_fill: Surrogate mode only - number of points sampled in
                ``[0, 1]^d`` to query the model with (default ``5000``).
            max_leaves: Optional cap on the number of leaves for data-driven
                subdivision.
            normalize: Whether to min-max normalize inputs to ``[0, 1]`` at fit
                time (reused at predict time). ``None`` auto-selects: ``True`` in
                dataset mode, ``False`` in surrogate mode (samples are already in
                ``[0, 1]``).
            margin: Relative slack of the enclosing simplex beyond ``[0, 1]^d``.
            device: Torch device for the accelerated transform (auto-detected).
            tolerance: Geometric tolerance for point-in-simplex tests.
            random_state: Seed for surrogate fill-point sampling.
            data_driven_max_iter: Safety cap on data-driven refinement passes.
        """
        if subdivision_strategy not in ("barycentric", "data_driven"):
            raise ValueError(
                "subdivision_strategy must be 'barycentric' or 'data_driven'"
            )
        if model is not None and n_features is None:
            raise ValueError("Surrogate mode (model=...) requires n_features.")

        self.classifier = classifier if classifier is not None else LinearSVC(C=1.0)
        self.model = model
        self.n_features = n_features
        self.subdivision_levels = subdivision_levels
        self.subdivision_strategy = subdivision_strategy
        self.n_fill = n_fill if n_fill is not None else 5000
        self.max_leaves = max_leaves
        self.normalize = normalize
        self.margin = margin
        self.device = device
        self.tolerance = tolerance
        self.random_state = random_state
        self.data_driven_max_iter = data_driven_max_iter

        self.tree: Optional[SimplexTree] = None
        self.dimension: Optional[int] = None
        self.leaf_simplexes: List = []
        self.all_nodes_lookup: Dict[frozenset, object] = {}
        self.backend = TransformBackend(device=device, tolerance=tolerance)
        self._synced = False

        # Stored training state (for refit during pruning).
        self._X_train_norm: Optional[np.ndarray] = None
        self._y_train = None
        self._min = None
        self._max = None

        # The raw (pre-normalization) data actually used to fit. In surrogate
        # mode this is the sampled fill set (plus any points passed to fit),
        # each labeled by the imitated model. Exposed via ``training_data_``.
        self._X_fit: Optional[np.ndarray] = None
        self._y_fit = None

        # Result of the automatic same-side pass at the end of fit().
        self.same_side_keys_: Set[frozenset] = set()

        if self.n_features is not None:
            self._ensure_tree(self.n_features)

    # ------------------------------------------------------------------
    # Tree construction / bookkeeping
    # ------------------------------------------------------------------
    def _mode(self) -> str:
        return "surrogate" if self.model is not None else "dataset"

    def _ensure_tree(self, dimension: int) -> None:
        """Build the base simplex tree for the given dimension if needed."""
        if self.tree is not None and self.dimension == dimension:
            return
        vertices = make_enclosing_simplex(dimension, margin=self.margin)
        self.tree = SimplexTree(vertices, tolerance=self.tolerance)
        self.dimension = dimension
        if self.subdivision_strategy == "barycentric" and self.subdivision_levels > 0:
            self.tree._add_barycentric_centers_recursively(self.subdivision_levels)
        self._mark_dirty()
        self._sync()

    def _mark_dirty(self) -> None:
        self._synced = False

    def _build_node_lookup(self) -> None:
        self.all_nodes_lookup = {}
        self.leaf_simplexes = []
        for node in self.tree._traverse_breadth_first():
            key = frozenset(node.vertex_indices)
            self.all_nodes_lookup[key] = node
            if node._is_leaf():
                self.leaf_simplexes.append(node)

    def _sync(self) -> None:
        """Rebuild the leaf lookup and the GPU backend after a tree change."""
        if self.tree is None:
            return
        self._build_node_lookup()
        self.backend.build(self.leaf_simplexes)
        self._synced = True

    def _ensure_synced(self) -> None:
        if not self._synced:
            self._sync()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _normalize_enabled(self) -> bool:
        if self.normalize is None:
            return self.model is None
        return self.normalize

    def _fit_scaler(self, X: np.ndarray) -> None:
        if not self._normalize_enabled():
            self._min = None
            self._max = None
            return
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._min is None:
            return X
        return (X - self._min) / (self._max - self._min + 1e-10)

    @staticmethod
    def _ensure_in_unit_cube(X: np.ndarray, tol: float = 1e-6) -> None:
        """Validate that inputs are normalized to ``[0, 1]`` per feature.

        Raises ``ValueError`` when normalization is disabled but the data lies
        outside the unit cube (so the caller either scales the data or passes
        ``normalize=True``).
        """
        if X.size == 0:
            return
        lo = float(np.min(X))
        hi = float(np.max(X))
        if lo < -tol or hi > 1.0 + tol:
            raise ValueError(
                f"Input features must be normalized to [0, 1] when normalize=False "
                f"(got value range [{lo:.4g}, {hi:.4g}]). Either min-max scale your "
                f"data to [0, 1] first, or construct with normalize=True to let the "
                f"classifier scale it for you."
            )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, data_points) -> csr_matrix: # TODO: use pytorch insead of matrix_csr
        """Embed points into sparse barycentric coordinates.

        For each point, finds its containing leaf simplex and writes that
        simplex's barycentric coordinates into the columns for its vertices.
        Points are expected to live in the tree's coordinate space (i.e. already
        normalized to ``[0, 1]^d`` when normalization is enabled); ``fit`` and
        ``predict`` handle normalization for you.

        Args:
            data_points: Array of shape ``(n_samples, d)``.

        Returns:
            Sparse ``csr_matrix`` of shape ``(n_samples, n_vertices)`` whose rows
            each hold ``d + 1`` non-zero barycentric weights summing to 1.
        """
        X = np.asarray(data_points, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.tree is None:
            self._ensure_tree(X.shape[1])
        self._ensure_synced()

        m = X.shape[0]
        n_cols = len(self.tree.registry)
        d1 = self.dimension + 1

        leaf_index, found, alphas = self.backend.embed(X)

        row_arrays: List[np.ndarray] = []
        col_arrays: List[np.ndarray] = []
        val_arrays: List[np.ndarray] = []

        found_idx = np.nonzero(found)[0]
        if found_idx.size:
            li = leaf_index[found_idx]
            gidx = self.backend.leaf_vertex_indices[li]      # (F, d+1)
            vals = alphas[found_idx]                          # (F, d+1)
            row_arrays.append(np.repeat(found_idx, d1))
            col_arrays.append(gidx.reshape(-1))
            val_arrays.append(vals.reshape(-1))

        # Fallback: exact per-point search for anything the GPU pass missed
        # (points outside every leaf, or inside a degenerate leaf).
        for i in np.nonzero(~found)[0]:
            point = tuple(X[i])
            leaf = self.tree.find_containing_simplex(point)
            if leaf is None:
                continue
            emb = leaf._embed_point(point)
            if emb is None:
                continue
            gidx = np.asarray(leaf.vertex_indices, dtype=np.int64)
            val = np.asarray(emb, dtype=float)
            row_arrays.append(np.full(gidx.shape[0], i, dtype=np.int64))
            col_arrays.append(gidx)
            val_arrays.append(val)

        if row_arrays:
            rows = np.concatenate(row_arrays)
            cols = np.concatenate(col_arrays)
            values = np.concatenate(val_arrays)
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=float)

        return csr_matrix((values, (rows, cols)), shape=(m, n_cols))

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        if callable(self.model):
            return np.asarray(self.model(X))
        raise TypeError(
            "model must expose a .predict(X) method or be callable X -> labels"
        )

    def _surrogate_training_data(self, X, y):
        d = self.n_features
        rng = np.random.default_rng(self.random_state)
        X_fill = rng.uniform(0.0, 1.0, size=(self.n_fill, d))
        if X is not None:
            X_extra = np.asarray(X, dtype=float)
            if X_extra.ndim == 1:
                X_extra = X_extra.reshape(1, -1)
            X_fill = np.vstack([X_fill, X_extra])
        y_fill = self._model_predict(X_fill)
        return X_fill, y_fill

    def _fit_estimator(self, X_transformed, y) -> None:
        if y is not None:
            self.classifier.fit(X_transformed, y)
        else:
            self.classifier.fit(X_transformed)

    def fit(self, X=None, y=None):
        """Train the classifier.

        Dataset mode: pass ``X`` (and ``y``). Surrogate mode: ``X``/``y`` are
        optional - fill points are sampled and labeled by the model; any ``X``
        given is added to the fill set.

        For linear classifiers, redundant same-side subdivisions are merged back
        automatically at the end (see ``remove_same_side_leaves``): this shrinks
        the tree without moving the decision boundary. Non-convex pruning stays a
        separate, explicit ``remove_nonconvex_leaves`` call.

        Returns:
            ``self``.
        """
        if self._mode() == "surrogate":
            X_fit, y_fit = self._surrogate_training_data(X, y)
        else:
            if X is None:
                raise ValueError("Dataset mode requires X (and usually y).")
            X_fit = np.asarray(X, dtype=float)
            if X_fit.ndim == 1:
                X_fit = X_fit.reshape(1, -1)
            y_fit = None if y is None else np.asarray(y)

        # When normalization is off, the data must already live in [0, 1]^d,
        # otherwise points fall outside the enclosing simplex and embed to
        # nothing. Fail loudly instead of silently producing empty rows.
        if not self._normalize_enabled():
            self._ensure_in_unit_cube(X_fit)

        self._X_fit = X_fit
        self._y_fit = y_fit

        d = X_fit.shape[1]
        self._fit_scaler(X_fit)
        X_norm = self._apply_scaler(X_fit)

        self._X_train_norm = X_norm
        self._y_train = y_fit

        if self.subdivision_strategy == "data_driven":
            self._fit_data_driven(X_norm, y_fit)
        else:
            self._ensure_tree(d)
            X_transformed = self.transform(X_norm)
            self._fit_estimator(X_transformed, y_fit)

        # Merge back redundant same-side subdivisions (linear only): splits the
        # boundary never crosses and whose leaves are all one class carry no
        # boundary information, so collapsing them removes leaves without moving
        # the decision surface. remove_same_side_leaves() refits and records the
        # final (empty on convergence) same_side_keys_ itself. For non-linear
        # classifiers there is nothing to merge, so we just record the state.
        if self.is_linear_classifier:
            self.remove_same_side_leaves()
        else:
            self._finalize_fit()
        return self

    @property
    def training_data_(self):
        """The ``(X, y)`` actually used to fit the classifier.

        In surrogate mode this is the sampled fill set in ``[0, 1]^d`` (plus any
        points passed to ``fit``), each labeled by the imitated model - i.e. the
        surrogate training set the package built for you. In dataset mode it is
        the raw ``X`` (and ``y``) you passed. Raises ``AttributeError`` if the
        classifier has not been fitted yet.
        """
        if self._X_fit is None:
            raise AttributeError(
                "training_data_ is unavailable: the classifier was not fitted, or "
                "it was built by an older version of the package (e.g. a stale "
                "Jupyter kernel or an old cached pickle). Restart the kernel so the "
                "updated package is re-imported, delete the cached model, then refit."
            )
        return self._X_fit, self._y_fit

    def _fit_data_driven(self, X_norm: np.ndarray, y) -> None:
        d = X_norm.shape[1]
        # Start from the root simplex (no uniform subdivision) and grow where
        # the classifier is wrong.
        vertices = make_enclosing_simplex(d, margin=self.margin)
        self.tree = SimplexTree(vertices, tolerance=self.tolerance)
        self.dimension = d
        self._mark_dirty()
        self._sync()

        max_depth = self.subdivision_levels

        X_transformed = self.transform(X_norm)
        self._fit_estimator(X_transformed, y)

        if y is None:
            return  # error-driven refinement needs labels

        for _ in range(self.data_driven_max_iter):
            preds = self.classifier.predict(self.transform(X_norm))
            wrong = np.nonzero(np.asarray(preds) != np.asarray(y))[0]
            if wrong.size == 0:
                break
            if self.max_leaves is not None and len(self.tree.get_leaves()) >= self.max_leaves:
                break

            leaves_to_split = {}
            for i in wrong:
                leaf = self.tree.find_containing_simplex(tuple(X_norm[i]))
                if leaf is None or leaf.depth >= max_depth:
                    continue
                leaves_to_split[id(leaf)] = leaf

            if not leaves_to_split:
                break

            split_any = False
            for leaf in leaves_to_split.values():
                if self.max_leaves is not None and len(self.tree.get_leaves()) >= self.max_leaves:
                    break
                try:
                    self.tree.subdivide_leaf(leaf)
                    split_any = True
                except ValueError:
                    continue

            if not split_any:
                break

            self._mark_dirty()
            self._sync()
            X_transformed = self.transform(X_norm)
            self._fit_estimator(X_transformed, y)

    def predict(self, X) -> np.ndarray:
        """Predict class labels for input points.

        Args:
            X: Data points of shape ``(n_samples, d)``.

        Returns:
            Array of predicted labels.
        """
        if self.tree is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        X_norm = self._apply_scaler(np.asarray(X, dtype=float))
        X_transformed = self.transform(X_norm)
        return self.classifier.predict(X_transformed)

    def _finalize_fit(self) -> None:
        """Automatic post-fit step: record same-side simplices (linear only)."""
        self._ensure_synced()
        if self.is_linear_classifier:
            try:
                self.same_side_keys_ = self.find_same_side_simplices()
            except Exception:
                self.same_side_keys_ = set()
        else:
            self.same_side_keys_ = set()

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------
    def find_containing_simplex(self, point):
        """Return the leaf simplex containing ``point`` (or ``None``).

        ``point`` is interpreted in the tree's coordinate space. Underpins
        ``find_adjacent_simplexes``.
        """
        if self.tree is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        self._ensure_synced()
        return self.tree.find_containing_simplex(tuple(point))

    def find_adjacent_simplexes(self, simplex) -> list:
        """Return all leaf simplexes adjacent to ``simplex`` (shared ``(d-1)``-face)."""
        if self.tree is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        self._ensure_synced()
        return self.tree.find_adjacent_simplexes(simplex)

    def get_simplex_vertices(self) -> List[List[Tuple[float, ...]]]:
        """Return vertex coordinates of every leaf simplex.

        (Formerly ``get_simplex_boundaries``.)
        """
        self._ensure_synced()
        return [leaf.get_vertices_as_tuples() for leaf in self.leaf_simplexes]

    def is_in_simplex(self, point, simplex) -> bool:
        """Return ``True`` if ``point`` lies inside ``simplex``.

        Args:
            point: Coordinates in the tree's space.
            simplex: A ``Simplex`` / ``SimplexTree`` node (e.g. from
                ``find_containing_simplex`` or ``leaf_simplexes``).
        """
        return bool(simplex._point_inside_simplex(tuple(point)))

    # ------------------------------------------------------------------
    # Linear-classifier helpers
    # ------------------------------------------------------------------
    @property
    def is_linear_classifier(self) -> bool:
        """Whether the internal classifier exposes linear weights."""
        return (hasattr(self.classifier, "coef_") and
                hasattr(self.classifier, "intercept_"))

    def get_weights_and_intercept(self):
        """Return the fitted linear classifier's weight vector and intercept.

        Raises ``AttributeError`` for non-linear classifiers.
        """
        return self._get_weights_and_intercept()

    def _get_weights_and_intercept(self):
        if not hasattr(self.classifier, "coef_"):
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        if not self.is_linear_classifier:
            raise AttributeError(
                f"{type(self.classifier).__name__} has no coef_ attribute. "
                "This requires a linear classifier (e.g. LinearSVC, "
                "LogisticRegression, Perceptron)."
            )
        weights = self.classifier.coef_[0]
        if hasattr(weights, "toarray"):
            weights = weights.toarray().flatten()
        elif hasattr(weights, "A"):
            weights = np.asarray(weights).flatten()
        intercept = self.classifier.intercept_[0]
        return weights, intercept

    def _hyperplanes(self):
        """Yield ``(weights, intercept)`` for every linear hyperplane.

        Binary -> one row; multiclass one-vs-rest -> one row per class.
        """
        coef = self.classifier.coef_
        intercept = self.classifier.intercept_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        coef = np.asarray(coef)
        intercept = np.atleast_1d(np.asarray(intercept))
        for h in range(coef.shape[0]):
            yield np.asarray(coef[h]).flatten(), float(intercept[h])

    def _predict_at_vertices(self, simplex_node) -> np.ndarray:
        n_vertices = len(self.tree.registry)
        predictions = []
        for vid in simplex_node.vertex_indices:
            one_hot = csr_matrix(([1.0], ([0], [vid])), shape=(1, n_vertices))
            predictions.append(self.classifier.predict(one_hot)[0])
        return np.array(predictions)

    @staticmethod
    def _simplex_crosses_boundary(simplex_node, weights, intercept) -> bool:
        decision_values = [weights[idx] + intercept for idx in simplex_node.vertex_indices]
        has_positive = any(val > 0 for val in decision_values)
        has_negative = any(val < 0 for val in decision_values)
        return has_positive and has_negative

    @staticmethod
    def _get_simplex_class(simplex_node, weights, intercept) -> bool:
        vals = [weights[i] + intercept for i in simplex_node.vertex_indices]
        return any(v > 0 for v in vals) or all(v == 0 for v in vals)

    def _are_siblings_same_side(self, parent_node, weights, intercept) -> bool:
        child0 = parent_node.children[0]
        if self._simplex_crosses_boundary(child0, weights, intercept):
            return False
        first_child_class = self._get_simplex_class(child0, weights, intercept)
        for child in parent_node.children[1:]:
            if self._simplex_crosses_boundary(child, weights, intercept):
                return False
            if self._get_simplex_class(child, weights, intercept) != first_child_class:
                return False
        return True

    # ------------------------------------------------------------------
    # Decision-boundary analysis
    # ------------------------------------------------------------------
    def identify_crossing_simplices(self) -> List[Dict]:
        """Find leaf simplices that the decision boundary crosses.

        Uses a weight-based test for linear classifiers and a prediction-based
        test for non-linear ones.

        Returns:
            List of dicts with keys ``'simplex'`` and ``'vertices'`` (plus
            ``'decision_values'`` when the classifier is linear).
        """
        self._ensure_synced()
        use_linear = self.is_linear_classifier
        if use_linear:
            weights, intercept = self._get_weights_and_intercept()

        crossing_simplices = []
        for leaf in self.leaf_simplexes:
            if use_linear:
                crosses = self._simplex_crosses_boundary(leaf, weights, intercept)
            else:
                preds = self._predict_at_vertices(leaf)
                crosses = not np.all(preds == preds[0])

            if crosses:
                info = {"simplex": leaf, "vertices": leaf.get_vertices_as_tuples()}
                if use_linear:
                    info["decision_values"] = np.array(
                        [weights[idx] for idx in leaf.vertex_indices]
                    )
                crossing_simplices.append(info)

        return crossing_simplices

    def find_same_side_simplices(self) -> Set[frozenset]:
        """Find leaf simplices whose siblings all lie on the same boundary side.

        These subdivisions do not contribute to the decision boundary and can be
        merged back into their parent. They are detected *and* merged away
        automatically at the end of ``fit`` for linear classifiers (see
        ``remove_same_side_leaves``); this method is also available directly.
        """
        weights, intercept = self._get_weights_and_intercept()
        same_side_keys: Set[frozenset] = set()
        leaf_parents = set()
        for leaf in self.leaf_simplexes:
            if leaf.parent:
                leaf_parents.add(leaf.parent)

        for parent in leaf_parents:
            if not all(child._is_leaf() for child in parent.children):
                continue
            if self._are_siblings_same_side(parent, weights, intercept):
                for child in parent.children:
                    same_side_keys.add(frozenset(child.vertex_indices))
        return same_side_keys

    def remove_same_side_leaves(self, max_iter: int = 50, refit: bool = True) -> int:
        """Merge back redundant same-side subdivisions (linear classifiers only).

        A parent split is *same-side* when the decision boundary crosses none of
        its (leaf) children and they all fall on the same side of it. Such a
        split adds no information about the boundary, so collapsing it removes
        leaves without moving the decision surface. Detection is repeated,
        refitting the classifier between passes, until no same-side splits
        remain (a new pass can expose parents that only became "all-leaf" after
        an earlier collapse). Called automatically at the end of ``fit``.

        Args:
            max_iter: Safety cap on the number of detect/collapse passes.
            refit: Whether to refit the classifier on the stored training data
                after each collapse pass.

        Returns:
            Total number of leaves removed (0 for non-linear classifiers).
        """
        if not self.is_linear_classifier:
            self.same_side_keys_ = set()
            return 0
        self._ensure_synced()
        total = 0
        try:
            keys = self.find_same_side_simplices()
        except Exception:
            self.same_side_keys_ = set()
            return 0
        for _ in range(max_iter):
            if not keys:
                break
            start = len(self.tree.get_leaves())
            for key in keys:
                # Collapsing one child un-splits the whole parent; the remaining
                # sibling keys for that parent then no-op harmlessly.
                self.tree.remove_by_leaf_key(key)
            removed = start - len(self.tree.get_leaves())
            if removed == 0:
                break
            total += removed
            self._mark_dirty()
            self._sync()
            if refit and self._X_train_norm is not None and self._y_train is not None:
                X_transformed = self.transform(self._X_train_norm)
                self._fit_estimator(X_transformed, self._y_train)
            # Re-detect on the new (pruned + refit) tree; this doubles as both the
            # next pass's work-list and the final same-side record, so we never
            # recompute find_same_side_simplices() redundantly afterwards.
            try:
                keys = self.find_same_side_simplices()
            except Exception:
                keys = set()
                break
        # keys already reflects the current tree/weights (empty on convergence).
        self.same_side_keys_ = keys
        return total

    def _sampling_epsilon(self, epsilon=None) -> float:
        """Boundary test-point placement fraction, defaulting to ``1 / (d + 1)``."""
        if epsilon is not None:
            return epsilon
        return 1.0 / (self.tree.dimension + 1)

    def find_nonconvex_leaves(self, criterion: str = "distance",
                              removal_factor: float = 0.15,
                              epsilon: Optional[float] = None,
                              keep_frac: Optional[float] = None,
                              min_depth: float = 0.0) -> Dict[frozenset, float]:
        """Flag leaf simplices whose decision boundary bends non-convexly.

        Multiclass-aware: every one-vs-rest hyperplane is checked and a leaf is
        flagged if any hyperplane bends the wrong way at it.

        Args:
            criterion: ``"distance"`` compares the meeting->average distance to
                ``removal_factor * shared_face_length``. ``"convexity_sign"`` uses
                the geometric side test in ``check_convexity``.
            removal_factor: (distance criterion) fraction of the shared-face
                length above which a bend counts as non-convex.
            epsilon: Boundary test-point placement fraction (default
                ``1 / (d + 1)``).
            keep_frac: Keep only the deepest ``keep_frac`` fraction of flagged
                leaves (data-dependent quantile gate).
            min_depth: Absolute floor on the normalized bend depth.

        Returns:
            Dict mapping each flagged leaf's vertex key (frozenset) to its bend
            depth (normalized by the shared-face length).
        """
        if criterion not in ("distance", "convexity_sign"):
            raise ValueError("criterion must be 'distance' or 'convexity_sign'")
        self._ensure_synced()
        eps = self._sampling_epsilon(epsilon)

        crossing = self.identify_crossing_simplices()
        crossing_ids = {id(info["simplex"]) for info in crossing}

        depths: Dict[frozenset, float] = {}
        for weights, intercept in self._hyperplanes():
            for info in crossing:
                leaf = info["simplex"]
                for nb in self.find_adjacent_simplexes(leaf):
                    if id(nb) not in crossing_ids:
                        continue
                    face_len = shared_face_length(leaf, nb)
                    if not face_len:
                        continue

                    if criterion == "convexity_sign":
                        is_convex, avg_pt, meeting, _, _ = check_convexity(
                            leaf, nb, weights, intercept,
                            global_tree=self.tree, epsilon=eps)
                        if is_convex or avg_pt is None or meeting is None:
                            continue
                        dist = float(np.linalg.norm(np.asarray(avg_pt) - np.asarray(meeting)))
                    else:
                        dist, _, meeting, _, _ = meeting_to_average_distance(
                            leaf, nb, weights, intercept, eps)
                        if dist is None or meeting is None:
                            continue
                        if dist <= removal_factor * face_len:
                            continue

                    depth = dist / face_len
                    for s in (leaf, nb):
                        key = frozenset(s.vertex_indices)
                        if depth > depths.get(key, 0.0):
                            depths[key] = depth

        return self._gate_depths(depths, keep_frac=keep_frac, min_depth=min_depth)

    @staticmethod
    def _gate_depths(depths: Dict[frozenset, float], keep_frac: Optional[float],
                     min_depth: float) -> Dict[frozenset, float]:
        if not depths:
            return dict(depths)
        thr = float(min_depth)
        if keep_frac is not None and 0.0 < keep_frac < 1.0:
            thr = max(thr, float(np.quantile(list(depths.values()), 1.0 - keep_frac)))
        if thr > 0.0:
            return {k: v for k, v in depths.items() if v >= thr}
        return dict(depths)

    def _remove_nonconvex_once(self, criterion, removal_factor, epsilon,
                               keep_frac, min_depth, remove_budget) -> int:
        depths = self.find_nonconvex_leaves(
            criterion=criterion, removal_factor=removal_factor, epsilon=epsilon,
            keep_frac=keep_frac, min_depth=min_depth)
        if not depths:
            return 0
        start = len(self.tree.get_leaves())
        for key, _ in sorted(depths.items(), key=lambda kv: -kv[1]):
            if remove_budget is not None and start - len(self.tree.get_leaves()) >= remove_budget:
                break
            self.tree.remove_by_leaf_key(key)
        return start - len(self.tree.get_leaves())

    def remove_nonconvex_leaves(self, criterion: str = "distance",
                                removal_factor: float = 0.15,
                                epsilon: Optional[float] = None,
                                keep_frac: Optional[float] = None,
                                min_depth: float = 0.0,
                                max_remove_frac: float = 0.25,
                                max_iter: int = 10,
                                refit: bool = True) -> int:
        """Iteratively prune non-convex leaves, deepest bend first.

        Removes at most ``max_remove_frac`` of the initial leaf count in total,
        over up to ``max_iter`` passes, refitting the classifier between passes
        (using the stored training data). Requires a linear classifier.

        Args:
            criterion: See ``find_nonconvex_leaves``.
            removal_factor: (distance criterion) shared-face-length fraction.
            epsilon: Boundary test-point placement fraction.
            keep_frac: Deepest-fraction gate on flagged leaves.
            min_depth: Absolute floor on normalized bend depth.
            max_remove_frac: Total removal budget as a fraction of initial leaves.
            max_iter: Maximum number of removal passes.
            refit: Whether to refit the classifier after each pass.

        Returns:
            Total number of leaves removed.
        """
        if not self.is_linear_classifier:
            raise AttributeError(
                f"{type(self.classifier).__name__} is not a linear classifier; "
                "non-convex removal requires linear weights."
            )
        self._ensure_synced()
        initial = len(self.tree.get_leaves())
        budget = int(initial * max_remove_frac)
        if budget < 1:
            return 0

        total = 0
        for _ in range(max_iter):
            remaining = budget - total
            if remaining <= 0:
                break
            removed = self._remove_nonconvex_once(
                criterion, removal_factor, epsilon, keep_frac, min_depth, remaining)
            total += removed
            if removed == 0:
                break
            self._mark_dirty()
            self._sync()
            if refit and self._X_train_norm is not None and self._y_train is not None:
                X_transformed = self.transform(self._X_train_norm)
                self._fit_estimator(X_transformed, self._y_train)

        self._finalize_fit()
        return total

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def compute_plane_equations(self) -> List[Dict]:
        """Compute the boundary hyperplane within each crossing simplex (linear).

        Returns:
            List of dicts with keys ``'simplex'``, ``'vertices'``,
            ``'coefficients'`` and ``'cartesian_form'``.
        """
        weights, _ = self._get_weights_and_intercept()
        plane_equations = []
        for info in self.identify_crossing_simplices():
            simplex = info["simplex"]
            plane_eq = PlaneEquation(simplex)
            coefficients = plane_eq.compute_plane_from_weights(weights)
            plane_equations.append({
                "simplex": simplex,
                "vertices": info["vertices"],
                "coefficients": coefficients,
                "cartesian_form": plane_eq.get_cartesian_form(),
            })
        return plane_equations

    def __repr__(self):
        n_leaves = len(self.leaf_simplexes) if self.tree is not None else 0
        return (f"SimplexTreeClassifier(mode={self._mode()}, "
                f"dimension={self.dimension}, strategy={self.subdivision_strategy}, "
                f"leaves={n_leaves}, device={self.backend.device})")
