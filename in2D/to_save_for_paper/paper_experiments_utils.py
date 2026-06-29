"""Shared helpers for the multiclass simplex-tree paper experiments.

Three reusable building blocks:
  1. NN  - small GRU classifier (binary -> sigmoid, multiclass -> softmax) and
           a train_nn helper that mirrors the diabetes notebook.
  2. Tree - multiclass-aware non-convex removal: iterate over every LinearSVC
           hyperplane (coef_ row), flag a leaf non-convex if ANY hyperplane
           bends the wrong way at it, prune the deepest bends first up to a
           cap (default 25% of leaves).
  3. LIME - explain_target / step_along_lime: pick the NN's predicted class as
           the target, follow LIME's class-target direction and report
           NN P(target) vs LIME P(target) along the step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# 1. Neural network (paper's GRU, generalised to multiclass)
# ---------------------------------------------------------------------------

class FeatureGRU(nn.Module):
    """The diabetes-paper RFE-GRU, generalised to K classes.

    Binary case (K==2): single sigmoid output, BCE loss, predict +/-1.
    Multiclass    (K>2): softmax over K outputs, CE loss, predict 0..K-1.
    The d input features are reshaped as a length-d sequence (1 feature per
    time step) and fed to a GRU(hidden=64) -> Linear -> activation.
    """

    def __init__(self, d, n_classes=2, hidden=64):
        super().__init__()
        self.n_classes = n_classes
        out_dim = 1 if n_classes == 2 else n_classes
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def _logits(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)             # (B, d) -> (B, d, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def forward(self, x):
        """Returns probabilities (sigmoid for binary, softmax for multiclass)."""
        logits = self._logits(x)
        if self.n_classes == 2:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


def train_nn(model, X_tr, y_tr, X_val, y_val, *,
             epochs=200, lr=0.01, batch_size=32, device=None):
    """Train the GRU. y_tr/y_val: binary +/-1 or multiclass 0..K-1 ints.

    Mirrors the diabetes notebook: 200 epochs / batch=32 / Adam(lr=0.01),
    keep best validation snapshot, return the model with loss_history attached.
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

    best_acc, best_state = 0.0, None
    loss_history = []
    for _ in range(epochs):
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
        if n_classes == 2:
            val_true = y_val
        else:
            val_true = y_val
        acc = accuracy_score(val_true, preds)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.loss_history = loss_history
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
    """Full probability matrix (n_samples, n_classes).
    For binary returns the 2-column [P(-1), P(+1)] layout LIME expects."""
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
# 2. Tree helpers - multiclass-aware non-convex finding + removal
# ---------------------------------------------------------------------------

from in2D.classifying.classes.utilss.convexity_check import (
    check_convexity, get_shared_vertices)


def _hyperplanes(model):
    """Yield (weights, intercept) for every LinearSVC hyperplane.
    Binary => 1 row, Multiclass-OvR => n_classes rows."""
    coef = model.classifier.coef_
    intercept = model.classifier.intercept_
    if hasattr(coef, 'toarray'):
        coef = coef.toarray()
    for h in range(coef.shape[0]):
        yield np.asarray(coef[h]).flatten(), float(intercept[h])


def _shared_face_length(leaf_a, leaf_b):
    """Longest edge of the shared face between two adjacent leaves."""
    shared = get_shared_vertices(leaf_a, leaf_b)
    if len(shared) < 2:
        return 0.0
    return max(np.linalg.norm(shared[i] - shared[j])
               for i in range(len(shared))
               for j in range(i + 1, len(shared)))


def find_nonconvex_leaves(model, epsilon=0.05):
    """Returns {leaf_key: max_depth} across all hyperplanes (multiclass-aware).

    Implements the diabetes-notebook policy generalised to multiclass:
    for every adjacent boundary-crossing pair (leaf, neighbour) and every
    LinearSVC hyperplane, run check_convexity. If the average point lands on
    the wrong side, mark BOTH leaves non-convex and record the depth = normalised
    distance(meeting_point, average_point) for ranking deepest bends first.
    """
    crossing = model.identify_crossing_simplices()
    crossing_ids = {id(info['simplex']) for info in crossing}
    depths = {}
    for weights, intercept in _hyperplanes(model):
        for info in crossing:
            leaf = info['simplex']
            for nb in model.find_adjacent_simplexes(leaf):
                if id(nb) not in crossing_ids:
                    continue
                is_convex, average_pt, meeting, _, _ = check_convexity(
                    leaf, nb, weights, intercept,
                    global_tree=model.tree, epsilon=epsilon)
                if is_convex or average_pt is None or meeting is None:
                    continue
                face_len = _shared_face_length(leaf, nb)
                if face_len < 1e-12:
                    continue
                d = float(np.linalg.norm(np.asarray(average_pt) -
                                         np.asarray(meeting)) / face_len)
                for s in (leaf, nb):
                    key = frozenset(s.vertex_indices)
                    if d > depths.get(key, 0.0):
                        depths[key] = d
    return depths


def remove_nonconvex_once(model, *, epsilon=0.05, remove_budget=None):
    """One removal pass. Drops non-convex leaves deepest-first up to
    remove_budget (count of leaves dropped, NOT how many remain).
    Returns the number of leaves removed in this pass."""
    depths = find_nonconvex_leaves(model, epsilon=epsilon)
    if not depths:
        return 0
    start = len(model.tree.get_leaves())
    for key, _ in sorted(depths.items(), key=lambda kv: -kv[1]):
        if remove_budget is not None and start - len(model.tree.get_leaves()) >= remove_budget:
            break
        model.tree.remove_by_leaf_key(key)
    return start - len(model.tree.get_leaves())


def build_convex_surrogate(make_model, X_fit, y_fit, *,
                           epsilon=0.05, max_remove_frac=0.25, max_iter=10,
                           verbose=True):
    """Build the surrogate via make_model(), then iteratively prune non-convex
    leaves up to max_remove_frac of the INITIAL leaf count (total budget).
    Refits after every pass."""
    model = make_model()
    model.fit(X_fit, y_fit)
    initial = len(model.tree.get_leaves())
    budget = int(initial * max_remove_frac)
    total = 0
    if verbose:
        print(f'Start: {initial} leaves  (cap = {max_remove_frac:.0%} = {budget} leaves total)')
    for it in range(max_iter):
        remaining = budget - total
        if remaining <= 0:
            if verbose: print(f'  iter {it}: budget reached -> stop')
            break
        before = len(model.tree.get_leaves())
        removed = remove_nonconvex_once(model, epsilon=epsilon, remove_budget=remaining)
        total += removed
        if verbose:
            print(f'  iter {it}: {before} -> {len(model.tree.get_leaves())} leaves  '
                  f'(removed {removed}, total {total} = {total/initial:.1%})')
        if removed == 0:
            if verbose: print(f'  iter {it}: nothing to prune -> stop')
            break
        model._build_node_lookup()
        model.fit(X_fit, y_fit)
    return model


def leaf_key_for_point(model, x):
    """Frozenset key of the leaf containing point x, or None."""
    leaf = model.tree.find_containing_simplex(tuple(x))
    return frozenset(leaf.vertex_indices) if leaf is not None else None


# ---------------------------------------------------------------------------
# 3. LIME helpers - target-class direction, step, NN P(target) vs LIME P(target)
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
    """Fit LIME at `point`, extract weights for class `target_label` in the
    ORIGINAL feature space, return (w, intercept, lime_p_target, grad).

    target_label: index into nn_proba_fn(point)[0]. Binary case: pass 1 for the
    +1 class (LIME sees probabilities ordered [P(-1), P(+1)]).
    """
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
    """Push `point` along LIME's target-reinforcing direction d = grad / ||grad||
    (the direction LIME claims drives P(target_label) up). Returns
    (alphas, nn_P_target, lime_P_target_clipped, w, d).
    """
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
# 4. Generalisation-coefficient helpers (true-label)
# ---------------------------------------------------------------------------

def gen_coef(predict_fn, X_train, y_train, X_test, y_test):
    """Returns (train_acc, test_acc, test_acc / train_acc)."""
    tr = accuracy_score(y_train, predict_fn(X_train))
    te = accuracy_score(y_test,  predict_fn(X_test))
    return tr, te, te / max(tr, 1e-9)


# ---------------------------------------------------------------------------
# 5. Preprocessing - RFE down to N features for binary or multiclass
# ---------------------------------------------------------------------------

def rfe_select(X_train, y_train, all_feature_names, n_select):
    """Drop everything except the top-`n_select` features chosen by RFE on
    LogisticRegression (multinomial for multiclass). Returns (sel_mask,
    chosen_names). Apply the mask to test data afterwards."""
    n_classes = len(np.unique(y_train))
    lr = LogisticRegression(max_iter=2000,
                            multi_class=('multinomial' if n_classes > 2 else 'auto'),
                            solver='lbfgs')
    rfe = RFE(lr, n_features_to_select=n_select)
    rfe.fit(X_train, y_train)
    sel = rfe.support_
    names = [n for n, k in zip(all_feature_names, sel) if k]
    return sel, names
