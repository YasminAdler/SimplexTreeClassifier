"""Walk every public API and print, for each call:

    FUNCTION                     VARIABLE            = VALUE

Reading (and editing) it forces you to know what each call returns. It also
mirrors everything to a log file, and prints a pass/fail banner at the end.

Run:
    python api_test.py               # logs to ./api_run.log
    python api_test.py my.log        # custom log path
"""

import contextlib, io, logging, os, sys
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from simplex_tree_classifier import (
    SimplexTreeClassifier, SimplexTree, Simplex, VertexRegistry,
    PlaneEquation, make_enclosing_simplex, get_device, __version__,
)

# ---- logging to console + file --------------------------------------------
LOG_PATH = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_run.log")
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
for h in (logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)):
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

_count = 0
def show(function, variable, value):
    """Print one line: which FUNCTION produced which VARIABLE, and its VALUE."""
    global _count
    _count += 1
    logger.info(f"{function:<40} {variable:<18}= {value}")

def group(title):
    logger.info(f"\n# ---- {title} " + "-" * (58 - len(title)))

def round_coords(obj, ndigits=2):
    """Round every float in a nested list/tuple to ndigits decimals."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return type(obj)(round_coords(x, ndigits) for x in obj)
    return obj

def tree_to_str(tree):
    """Capture SimplexTree.print_tree() output as a string (it prints to stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tree.print_tree(show_only_splitting_points=False)
    return "\n" + buf.getvalue().rstrip()


def data_2d(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 2))
    return X, (X[:, 0] + X[:, 1] > 1.0).astype(int)

def data_nd(n=300, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    return X, (X.sum(1) > d / 2.0).astype(int)

def data_multiclass(n=600, d=4, k=4, seed=3):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=d, n_informative=d,
                               n_redundant=0, n_classes=k, random_state=seed)
    return (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9), y


logger.info(f"{'FUNCTION':<40} {'VARIABLE':<18}  VALUE")
logger.info("-" * 78)

try:
    # ---- make_enclosing_simplex / get_device ------------------------------
    group("make_enclosing_simplex / get_device")
    verts2d = make_enclosing_simplex(2, margin=0.0)
    show("make_enclosing_simplex(2)", "verts2d", verts2d)
    verts4d = make_enclosing_simplex(4)
    show("make_enclosing_simplex(4)", "verts4d", verts4d)
    corner_inside = SimplexTree(verts4d).find_containing_simplex((1., 1., 1., 1.)) is not None
    show("SimplexTree.find_containing_simplex", "corner_inside", corner_inside)
    device = get_device().type
    show("get_device", "device", device)

    # ---- VertexRegistry ---------------------------------------------------
    group("VertexRegistry")

    reg = VertexRegistry()
    indices = reg.register_vertices([(0., 0.), (1., 0.), (0., 1.)])
    show("VertexRegistry.register_vertices", "indices", indices)

    n_vertices = len(reg)
    show("len(VertexRegistry)", "n_vertices", n_vertices)

    same_index = reg.register_vertices([(0., 0.)])[0]
    show("register_vertices (duplicate)", "same_index", same_index)

    matrix = reg.as_matrix()
    show("VertexRegistry.as_matrix", "matrix", matrix)

    # ---- Simplex ----------------------------------------------------------
    group("Simplex")

    simplex = Simplex(indices, reg)
    show("Simplex(indices, reg)", "simplex", repr(simplex))

    dimension = simplex.dimension
    show("Simplex.dimension", "dimension", dimension)

    vertices = simplex.get_vertices_as_tuples()
    show("Simplex.get_vertices_as_tuples", "vertices", vertices)

    inside = simplex.contains_point((0.2, 0.2))
    show("Simplex.contains_point", "inside", inside)

    outside = simplex.contains_point((5., 5.))
    show("Simplex.contains_point", "outside", outside)

    # ---- SimplexTree ------------------------------------------------------
    group("SimplexTree")

    tree = SimplexTree(make_enclosing_simplex(2, margin=0.0))
    tree._add_barycentric_centers_recursively(2)

    n_leaves = len(tree.get_leaves())
    show("SimplexTree.get_leaves", "n_leaves", n_leaves)

    n_splits = len(tree.get_splitting_points())
    show("SimplexTree.get_splitting_points", "n_splits", n_splits)

    found = tree.find_containing_simplex((0.3, 0.3)) is not None
    show("SimplexTree.find_containing_simplex", "found", found)

    adjacent = tree.find_adjacent_simplexes(tree.get_leaves()[0])
    adjacent_vertices = round_coords([s.get_vertices_as_tuples() for s in adjacent])
    show("SimplexTree.find_adjacent_simplexes", "adjacent_vertices", adjacent_vertices)

    children = tree.subdivide_leaf(tree.get_leaves()[0])
    children_vertices = round_coords([c.get_vertices_as_tuples() for c in children])
    show("SimplexTree.subdivide_leaf", "children_vertices", children_vertices)
    show("SimplexTree.print_tree (after subdivide)", "tree_str", tree_to_str(tree))

    removed_key = sorted(children[0].vertex_indices)
    removed_ok = tree.remove_by_leaf_key(frozenset(children[0].vertex_indices))
    show("SimplexTree.remove_by_leaf_key", "removed_key", removed_key)
    show("SimplexTree.remove_by_leaf_key", "removed_ok", removed_ok)
    show("SimplexTree.print_tree (after removal)", "tree_str", tree_to_str(tree))

    # ---- Constructor modes ------------------------------------------------
    group("Constructor modes")

    # type 1: dataset mode (no model) - learns straight from (X, y)
    dataset_clf = SimplexTreeClassifier(subdivision_levels=2)
    show("SimplexTreeClassifier(dataset)", "mode", dataset_clf._mode())
    show("SimplexTreeClassifier(dataset)", "classifier", type(dataset_clf.classifier).__name__)
    show("SimplexTreeClassifier(dataset)", "tree_built_at_init", dataset_clf.tree is not None)

    # type 2: surrogate mode (model + n_features) - imitates the given model
    surrogate_clf = SimplexTreeClassifier(model=lambda X: np.zeros(len(X)),
                                          n_features=2, subdivision_levels=2)
    show("SimplexTreeClassifier(surrogate)", "mode", surrogate_clf._mode())
    show("SimplexTreeClassifier(surrogate)", "n_features", surrogate_clf.n_features)
    show("SimplexTreeClassifier(surrogate)", "tree_built_at_init", surrogate_clf.tree is not None)

    # validation: surrogate mode without n_features must raise
    try:
        SimplexTreeClassifier(model=lambda X: np.zeros(len(X)))
        surrogate_error = "NO error (unexpected)"
    except ValueError as e:
        surrogate_error = str(e)
    show("SimplexTreeClassifier(model=...)", "surrogate_error", surrogate_error)

    # ---- transform / fit / predict (2-D) ----------------------------------
    group("transform / fit / predict (2-D)")
    X, y = data_2d()
    clf = SimplexTreeClassifier(subdivision_levels=3)
    T = clf.transform(X)
    show("SimplexTreeClassifier.transform", "T_shape", T.shape)
    row_sums = (round(float(np.asarray(T.sum(1)).min()), 6),
                round(float(np.asarray(T.sum(1)).max()), 6))
    show("transform (row sums min,max)", "row_sums", row_sums)
    clf.fit(X, y)
    show("SimplexTreeClassifier.fit", "n_leaves", len(clf.leaf_simplexes))
    accuracy_2d = round(float((clf.predict(X) == y).mean()), 4)
    show("SimplexTreeClassifier.predict", "accuracy_2d", accuracy_2d)
    show("repr(clf)", "clf", repr(clf))

    # ---- fit / predict (N-D) ----------------------------------------------
    group("fit / predict (N-D, d=5)")
    Xn, yn = data_nd(d=5)
    clf_nd = SimplexTreeClassifier(subdivision_levels=2).fit(Xn, yn)
    show("fit/predict (d=5)", "n_leaves_nd", len(clf_nd.leaf_simplexes))
    show("fit/predict (d=5)", "accuracy_nd", round(float((clf_nd.predict(Xn) == yn).mean()), 4))

    # ---- Data-driven subdivision ------------------------------------------
    group("Data-driven subdivision")
    Xdd, ydd = data_2d(n=300)
    clf_dd = SimplexTreeClassifier(subdivision_levels=6,
                                   subdivision_strategy="data_driven", max_leaves=200)
    clf_dd.fit(Xdd, ydd)
    show("data_driven fit", "n_leaves_dd", len(clf_dd.leaf_simplexes))
    show("data_driven predict", "accuracy_dd", round(float((clf_dd.predict(Xdd) == ydd).mean()), 4))

    # ---- Surrogate mode ---------------------------------------------------
    group("Surrogate mode")
    def black_box(Z):
        Z = np.atleast_2d(Z)
        return (Z[:, 0] + Z[:, 1] > 1.0).astype(int)
    sur = SimplexTreeClassifier(model=black_box, n_features=2,
                                subdivision_levels=3, n_fill=1500, random_state=0)
    sur.fit()
    grid = np.random.default_rng(2).uniform(0, 1, size=(200, 2))
    agreement = round(float((sur.predict(grid) == black_box(grid)).mean()), 4)
    show("surrogate (callable) predict", "agreement", agreement)
    class Model:
        def predict(self, Z):
            return (np.atleast_2d(Z)[:, 0] > 0.5).astype(int)
    sur2 = SimplexTreeClassifier(model=Model(), n_features=2,
                                 subdivision_levels=3, random_state=0).fit()
    obj_preds = sur2.predict(np.array([[0.9, 0.1], [0.1, 0.9]])).tolist()
    show("surrogate (.predict obj)", "obj_preds", obj_preds)

    # ---- Classifier geometry queries --------------------------------------
    group("Classifier geometry queries")
    leaf = clf.find_containing_simplex((0.3, 0.3))
    show("clf.find_containing_simplex", "leaf_vertices", leaf.get_vertices_as_tuples())
    show("clf.is_in_simplex", "inside", clf.is_in_simplex((0.3, 0.3), leaf))
    far = tuple(np.array(leaf.get_vertices_as_tuples()).mean(0) + 100.0)
    show("clf.is_in_simplex", "outside", clf.is_in_simplex(far, leaf))
    show("clf.find_adjacent_simplexes", "n_adjacent",
         len(clf.find_adjacent_simplexes(clf.leaf_simplexes[0])))
    simplex_vertices = clf.get_simplex_vertices()
    show("clf.get_simplex_vertices", "n_leaf_vertexsets", len(simplex_vertices))
    show("clf.get_simplex_vertices", "first_leaf", simplex_vertices[0])

    # ---- Linear helpers + decision boundary (2-class) ---------------------
    group("Linear helpers + decision boundary (2-class)")
    show("clf.is_linear_classifier", "is_linear", clf.is_linear_classifier)
    weights, intercept = clf.get_weights_and_intercept()
    show("clf.get_weights_and_intercept", "weights_shape", np.shape(weights))
    show("clf.get_weights_and_intercept", "intercept", round(float(np.ravel(intercept)[0]), 4))
    crossing = clf.identify_crossing_simplices()
    show("clf.identify_crossing_simplices", "n_crossing", len(crossing))
    show("clf.identify_crossing_simplices", "crossing0_keys", sorted(crossing[0].keys()))
    show("clf.same_side_keys_", "n_same_side", len(clf.same_side_keys_))
    show("clf.find_same_side_simplices", "n_same_side2", len(clf.find_same_side_simplices()))
    nc_dist = clf.find_nonconvex_leaves(criterion="distance", removal_factor=0.1)
    show("clf.find_nonconvex_leaves(distance)", "n_nonconvex_dist", len(nc_dist))
    nc_sign = clf.find_nonconvex_leaves(criterion="convexity_sign", epsilon=0.05)
    show("clf.find_nonconvex_leaves(sign)", "n_nonconvex_sign", len(nc_sign))

    # ---- PlaneEquation ----------------------------------------------------
    group("PlaneEquation")
    planes = clf.compute_plane_equations()
    show("clf.compute_plane_equations", "n_planes", len(planes))
    pe = PlaneEquation(crossing[0]["simplex"])
    coefficients = np.round(pe.compute_plane_from_weights(clf.classifier.coef_[0]), 4).tolist()
    show("PlaneEquation.compute_plane_from_weights", "coefficients", coefficients)
    show("PlaneEquation.get_cartesian_form", "cartesian_form", pe.get_cartesian_form())

    # ---- Multiclass (4 classes, one-vs-rest) ------------------------------
    group("Multiclass (4 classes)")
    Xm, ym = data_multiclass()
    clfm = SimplexTreeClassifier(n_features=4, margin=0.0, normalize=False,
                                 classifier=LinearSVC(C=1000, max_iter=20000),
                                 subdivision_levels=2).fit(Xm, ym)
    show("multiclass classifier.coef_", "n_hyperplanes", clfm.classifier.coef_.shape[0])
    show("multiclass fit", "n_leaves_mc", len(clfm.leaf_simplexes))
    show("multiclass predict", "accuracy_mc", round(float(accuracy_score(ym, clfm.predict(Xm))), 4))
    show("multiclass find_nonconvex_leaves", "n_nonconvex_mc",
         len(clfm.find_nonconvex_leaves(criterion="convexity_sign", epsilon=0.05)))

    # ---- Non-convex pruning ----------------------------------------------
    group("Non-convex pruning")
    clf_prune = SimplexTreeClassifier(subdivision_levels=4).fit(*data_2d(n=400))
    leaves_before = len(clf_prune.tree.get_leaves())
    removed = clf_prune.remove_nonconvex_leaves(criterion="distance",
                                                removal_factor=0.1, max_remove_frac=0.25)
    show("clf.remove_nonconvex_leaves", "leaves_before", leaves_before)
    show("clf.remove_nonconvex_leaves", "removed", removed)
    show("clf.remove_nonconvex_leaves", "leaves_after", len(clf_prune.tree.get_leaves()))

    # By default (dataset AND surrogate mode) the inner classifier is LinearSVC,
    # so removal normally works in both. The only way to fail the removal is to explicitly pass
    # a non-linear classifier - which we allow - SHOULD WE ? 

    Xnl, ynl = data_2d()
    nonlinear = SimplexTreeClassifier(classifier=SVC(kernel="rbf"), subdivision_levels=2).fit(Xnl, ynl)
    show("non-linear (SVC rbf)", "is_linear", nonlinear.is_linear_classifier)
    show("non-linear predict", "accuracy", round(float((nonlinear.predict(Xnl) == ynl).mean()), 4))
    show("non-linear identify_crossing_simplices", "n_crossing",
         len(nonlinear.identify_crossing_simplices()))
    # ... but non-convex REMOVAL needs linear weights -> must raise
    try:
        nonlinear.remove_nonconvex_leaves()
        pruning_error = "NO error (unexpected)"
    except AttributeError as e:
        pruning_error = str(e)
    show("remove_nonconvex_leaves (non-linear)", "pruning_error", pruning_error)

except Exception as exc:
    logger.info("\n" + "=" * 78)
    logger.info(f"FAILED after {_count} values: {type(exc).__name__}: {exc}")
    logger.info("=" * 78)
    raise

logger.info("\n" + "=" * 78)
logger.info(f"ALL {_count} CALLS SUCCEEDED  ->  log saved to {LOG_PATH}")
logger.info("=" * 78)
