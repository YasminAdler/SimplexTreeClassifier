"""Experiment scaffolding for the paper notebooks - the parts that are NOT the
classifier itself.

Everything simplex-tree related now lives in the published ``simplex_tree_classifier``
package. This module keeps only the pieces that a *classifier library* has no
business shipping:

  1. NN         - FeatureMLP and train_nn / predict_classes / predict_proba
                  (the black box being explained). Plain feed-forward net, all
                  raw features in - no GRU, no recurrence, no reduction.
  2. LIME       - make_lime_explainer / lime_direction / push_along_lime.
  3. metrics    - gen_coef (generalisation coefficient).

No feature reduction anywhere: every experiment feeds all raw features to the
network (no PCA, no RFE, no feature selection).

The three "tree" helpers at the bottom (find_nonconvex_leaves,
build_convex_surrogate, leaf_key_for_point) are **thin forwarders**: they contain
no simplex geometry, they only call methods that the package already exposes.
They exist so the notebooks read the same as before; you can call the package
methods directly instead (see the docstrings).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 1. Neural network (feed-forward MLP), generalised to K classes
# ---------------------------------------------------------------------------


class FeatureMLP(nn.Module):
    """Feed-forward MLP (2 hidden layers of ``hidden``, ReLU).

    Takes ALL ``d`` raw features straight in - no recurrence, no feature
    reduction. Binary (K==2): single sigmoid output. Multiclass (K>2): softmax.
    """

    def __init__(self, d, n_classes=2, hidden=64):
        super().__init__()
        self.n_classes = n_classes
        out_dim = 1 if n_classes == 2 else n_classes
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, out_dim)

    def _logits(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)   # accept (B, d, 1) too
        return self.fc(self.net(x))

    def forward(self, x):
        logits = self._logits(x)
        if self.n_classes == 2:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


def train_nn(model, X_tr, y_tr, X_val, y_val, *,
             epochs=200, lr=0.01, batch_size=32, device=None):
    """Train the network. y: binary +/-1 or multiclass 0..K-1 ints.

    Adam(lr) + ReduceLROnPlateau on validation accuracy, keep the best-validation
    snapshot, and return the model with ``loss_history`` and ``val_acc_history``
    attached. Prints the best validation accuracy and the epoch it was reached.
    """
    if device is None:
        device = next(model.parameters()).device
    n_classes = model.n_classes

    if n_classes == 2:
        y_tr_t  = torch.tensor(((np.asarray(y_tr)  + 1) / 2).astype(np.float32)).unsqueeze(1)
        y_val_t = torch.tensor(((np.asarray(y_val) + 1) / 2).astype(np.float32)).unsqueeze(1)
        criterion = nn.BCELoss()
    else:
        y_tr_t  = torch.tensor(np.asarray(y_tr,  dtype=np.int64))
        y_val_t = torch.tensor(np.asarray(y_val, dtype=np.int64))
        criterion = nn.CrossEntropyLoss()

    X_tr_t  = torch.tensor(np.asarray(X_tr,  dtype=np.float32))
    X_val_t = torch.tensor(np.asarray(X_val, dtype=np.float32)).to(device)
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    opt     = optim.Adam(model.parameters(), lr=lr)
    # LR decay on a validation plateau: when val accuracy stops improving for `patience`
    # epochs, halve the LR. This calms the late-epoch spikes in the loss curve.
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

    best_acc, best_state, best_ep = 0.0, None, 0
    loss_history, val_acc_history = [], []
    for ep in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if n_classes == 2:
                loss = criterion(model(xb), yb)
            else:
                loss = criterion(model._logits(xb), yb)
            loss.backward(); opt.step()
            running += loss.item(); n_batches += 1
        loss_history.append(running / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            preds = predict_classes(model, X_val_t.cpu().numpy(), device=device)
        acc = accuracy_score(y_val, preds)
        val_acc_history.append(acc)
        sched.step(acc)                       # plateau scheduler steps on the val metric
        if acc > best_acc:
            best_acc, best_state, best_ep = acc, {k: v.clone() for k, v in model.state_dict().items()}, ep

    if best_state is not None:
        model.load_state_dict(best_state)
    model.loss_history = loss_history
    model.val_acc_history = val_acc_history
    print(f'Best validation accuracy: {best_acc:.4f} (epoch {best_ep + 1}/{epochs})')
    return model


def predict_classes(model, X, device=None):
    """Class labels (binary -> +/-1, multiclass -> 0..K-1)."""
    if device is None:
        device = next(model.parameters()).device
    arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(arr).to(device)).cpu().numpy()
    if model.n_classes == 2:
        p = probs.flatten()
        return np.where(p >= 0.5, 1.0, -1.0)
    return np.argmax(probs, axis=1)


def predict_proba(model, X, device=None):
    """Probability matrix (n_samples, n_classes).
    Binary returns the [P(-1), P(+1)] layout LIME expects."""
    if device is None:
        device = next(model.parameters()).device
    arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(arr).to(device)).cpu().numpy()
    if model.n_classes == 2:
        p = probs.flatten()
        return np.column_stack([1 - p, p])
    return probs


# ---------------------------------------------------------------------------
# 2. LIME helpers - target-class direction, step, NN P(target) vs LIME P(target)
# ---------------------------------------------------------------------------

def make_lime_explainer(X_train, feature_names, n_classes, seed=0):
    import lime.lime_tabular
    return lime.lime_tabular.LimeTabularExplainer(
        X_train, mode='classification',
        feature_names=list(feature_names),
        class_names=[str(c) for c in range(n_classes)],
        discretize_continuous=False, random_state=seed)


def lime_direction(explainer, nn_proba_fn, point, target_label,
                   num_features, num_samples=3000):
    """Fit LIME at ``point`` and return (w, intercept, lime_p_fn, grad) for
    class ``target_label`` in the ORIGINAL feature space."""
    exp = explainer.explain_instance(point, nn_proba_fn,
                                     num_features=num_features,
                                     num_samples=num_samples,
                                     labels=(target_label,))
    coef = np.zeros(num_features)
    for i, w in exp.local_exp[target_label]:
        coef[i] = w
    intercept = exp.intercept[target_label]
    mean_, scale_ = explainer.scaler.mean_, explainer.scaler.scale_

    def lime_p(x):
        xs = (np.atleast_2d(x) - mean_) / scale_
        return intercept + xs.dot(coef)

    grad = coef / scale_                                # dP(target)/dx
    return coef, intercept, lime_p, grad


def push_along_lime(point, explainer, nn_proba_fn, target_label, num_features,
                    alphas=None, clip=(0.0, 1.0)):
    """Push ``point`` along LIME's target-reinforcing direction and return
    (alphas, nn_P_target, lime_P_target_clipped, w, d)."""
    if alphas is None:
        alphas = np.linspace(0, 0.4, 81)
    point = np.asarray(point, dtype=np.float32)
    coef, _, lime_p, grad = lime_direction(
        explainer, nn_proba_fn, point, target_label, num_features)
    d = grad
    nrm = np.linalg.norm(d)
    if nrm > 0:
        d = d / nrm
    xs = np.clip(point + np.outer(alphas, d), *clip)
    nn_p_target  = nn_proba_fn(xs)[:, target_label]
    lim_p_target = np.clip(lime_p(xs), 0, 1)
    return alphas, nn_p_target, lim_p_target, coef, d


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def gen_coef(predict_fn, X_train, y_train, X_test, y_test):
    """Returns (train_acc, test_acc, test_acc / train_acc)."""
    tr = accuracy_score(y_train, predict_fn(X_train))
    te = accuracy_score(y_test,  predict_fn(X_test))
    return tr, te, te / max(tr, 1e-9)


# ---------------------------------------------------------------------------
# 4. Tree helpers - THIN FORWARDERS to the package (no simplex geometry here)
# ---------------------------------------------------------------------------
# These used to contain the multiclass non-convex logic. That logic now lives in
# simplex_tree_classifier.SimplexTreeClassifier. The wrappers below only forward
# to it, so the notebooks keep their old call sites. Equivalent direct calls:
#
#   find_nonconvex_leaves(stc, epsilon=E)
#       -> stc.find_nonconvex_leaves(criterion="convexity_sign", epsilon=E)
#   build_convex_surrogate(make_model, X, y, epsilon=E, max_remove_frac=f, max_iter=n)
#       -> m = make_model(); m.fit(X, y)
#          m.remove_nonconvex_leaves(criterion="convexity_sign", epsilon=E,
#                                    max_remove_frac=f, max_iter=n)
#   leaf_key_for_point(stc, x)
#       -> frozenset(stc.find_containing_simplex(tuple(x)).vertex_indices)

def find_nonconvex_leaves(model, epsilon=0.05, keep_frac=None, min_depth=0.0):
    """Forward to the package's multiclass non-convex detector.

    Returns {leaf_key: bend_depth}. Uses the geometric sign test
    (``criterion="convexity_sign"``) to match the original paper policy.
    """
    return model.find_nonconvex_leaves(
        criterion="convexity_sign", epsilon=epsilon,
        keep_frac=keep_frac, min_depth=min_depth)


def build_convex_surrogate(make_model, X_fit, y_fit, *,
                           epsilon=0.05, max_remove_frac=0.25, max_iter=10,
                           verbose=True):
    """Fit a surrogate, then prune non-convex leaves via the package."""
    model = make_model()
    model.fit(X_fit, y_fit)
    initial = len(model.tree.get_leaves())
    removed = model.remove_nonconvex_leaves(
        criterion="convexity_sign", epsilon=epsilon,
        max_remove_frac=max_remove_frac, max_iter=max_iter, refit=True)
    if verbose:
        print(f'build_convex_surrogate: {initial} -> '
              f'{len(model.tree.get_leaves())} leaves (removed {removed})')
    return model


def leaf_key_for_point(model, x):
    """Frozenset key of the leaf containing point x, or None."""
    leaf = model.find_containing_simplex(tuple(x))
    return frozenset(leaf.vertex_indices) if leaf is not None else None
