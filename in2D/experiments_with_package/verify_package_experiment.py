"""Quick end-to-end check that the package + nn_utils reproduce Experiment 1
(non-convex removal raises the generalisation coefficient) on real data.

This is the vehicle notebook's Experiment-1 path in miniature (fewer epochs),
using ONLY the published package for everything simplex-tree related. Run with:

    python verify_package_experiment.py
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from simplex_tree_classifier import SimplexTreeClassifier          # the package
from nn_utils import (FeatureMLP, train_nn, predict_classes,
                      find_nonconvex_leaves, build_convex_surrogate, gen_coef)

device = torch.device("cpu")
ds = fetch_openml(name="vehicle", version=1, as_frame=True, parser="auto")
X_df = ds.data.select_dtypes(include=[np.number]).astype(np.float64)
labels = np.unique(ds.target.astype(str))
y = np.array([{c: i for i, c in enumerate(labels)}[v] for v in ds.target.astype(str).values])
X_all = MinMaxScaler().fit_transform(X_df.values)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y)
n_features, n_classes = X_train.shape[1], len(np.unique(y))
SUBDIVISION_LEVELS = max(1, int(np.log(2000) / np.log(n_features + 1)))
print(f"vehicle: {X_all.shape}, {n_classes} classes, subdivision_levels={SUBDIVISION_LEVELS}")

torch.manual_seed(42); np.random.seed(42)
nn_model = FeatureMLP(n_features, n_classes=n_classes, hidden=64).to(device)
train_nn(nn_model, X_train, y_train, X_test, y_test, epochs=60, lr=0.01, batch_size=32, device=device)
nn_pred = lambda Z: predict_classes(nn_model, Z, device=device)
print(f"MLP test acc: {accuracy_score(y_test, nn_pred(X_test)):.4f}")

np.random.seed(42)
X_fill = np.random.uniform(0, 1, (5000, n_features))
y_fill = nn_pred(X_fill)
X_fill = np.vstack([X_fill, X_train]); y_fill = np.concatenate([y_fill, nn_pred(X_train)])

def make_model():
    return SimplexTreeClassifier(
        n_features=n_features, margin=0.0, normalize=False,
        classifier=LinearSVC(C=10000, max_iter=20000),
        subdivision_levels=SUBDIVISION_LEVELS)

stc = make_model(); stc.fit(X_fill, y_fill)
print(f"surrogate leaves: {len(stc.leaf_simplexes)}")
depths = find_nonconvex_leaves(stc, epsilon=0.05)
print(f"crossing: {len(stc.identify_crossing_simplices())}   non-convex leaves: {len(depths)}")

stc_convex = build_convex_surrogate(make_model, X_fill, y_fill,
                                    epsilon=0.05, max_remove_frac=0.25, max_iter=8)

# decision_function(transform(...)) path used by the LIME cells
D = np.asarray(stc_convex.classifier.decision_function(stc_convex.transform(X_test)))
print("decision_function shape:", D.shape)

_, _, gc_full = gen_coef(stc.predict,        X_train, y_train, X_test, y_test)
_, _, gc_prune = gen_coef(stc_convex.predict, X_train, y_train, X_test, y_test)
print(f"gen-coef: full={gc_full:.4f}  pruned={gc_prune:.4f}  "
      f"({'improved' if gc_prune >= gc_full else 'lower'})")

# cache round-trip (notebooks pickle the surrogate under models/)
blob = pickle.dumps(stc_convex)
stc2 = pickle.loads(blob)
assert np.array_equal(stc2.predict(X_test), stc_convex.predict(X_test))
print("pickle round-trip OK")
print("VERIFY OK")
