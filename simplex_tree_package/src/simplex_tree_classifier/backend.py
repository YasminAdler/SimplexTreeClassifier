"""PyTorch backend that accelerates the barycentric ``transform``.

Leaves produced by barycentric subdivision are full ``d``-simplices (``d + 1``
vertices), so each non-degenerate leaf has an invertible edge matrix ``A`` with a
precomputed inverse. This backend stacks those per-leaf inverses into tensors and
embeds an entire batch of points against every leaf at once, on the GPU when one
is available. For each point it selects the leaf whose barycentric coordinates
are all non-negative and most interior.

Points that are not resolved on the GPU (outside every leaf, or inside a
degenerate leaf excluded from the tensor stack) are reported back so the caller
can fall back to the exact per-point CPU search.
"""

from typing import List, Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dependency, guard anyway
    torch = None
    _TORCH_AVAILABLE = False


def get_device(device=None):
    """Resolve a torch device, defaulting to CUDA when available, else CPU.

    Args:
        device: An explicit ``torch.device``/str, or ``None`` to auto-detect.

    Returns:
        A ``torch.device``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for simplex_tree_classifier. Install it with "
            "`pip install torch`."
        )
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformBackend:
    """Precomputes per-leaf tensors and batch-embeds points into barycentric coords."""

    def __init__(self, device=None, tolerance: float = 1e-10,
                 element_budget: int = 40_000_000):
        self.device = get_device(device)
        self.tolerance = tolerance
        # Caps the number of (chunk x leaves x dim) elements held at once.
        self.element_budget = int(element_budget)
        self.dtype = torch.float64
        self._built = False

        self.leaves: List = []
        self.n_leaves = 0
        self.dimension = 0
        # Global vertex indices per leaf, in the order alphas are produced.
        self.leaf_vertex_indices: Optional[np.ndarray] = None
        self._V0 = None      # (L, d)
        self._A_inv = None   # (L, d, d)

    @property
    def is_built(self) -> bool:
        return self._built and self.n_leaves > 0

    def build(self, leaves: List) -> None:
        """Stack tensors for all full, non-degenerate leaves.

        Args:
            leaves: Iterable of leaf ``SimplexTree`` nodes.
        """
        self._built = False
        self.leaves = []
        v0_list = []
        a_inv_list = []
        vidx_list = []

        dimension = None
        for leaf in leaves:
            d = leaf.dimension
            # Only stack full simplices with a usable inverse.
            if leaf.n_vertices != d + 1:
                continue
            if getattr(leaf, "is_degenerate", False) or getattr(leaf, "A_inv", None) is None:
                continue
            if dimension is None:
                dimension = d
            elif d != dimension:
                # Mixed dimensions should not happen within one tree; skip oddities.
                continue
            self.leaves.append(leaf)
            v0_list.append(np.asarray(leaf.vertices[0], dtype=np.float64))
            a_inv_list.append(np.asarray(leaf.A_inv, dtype=np.float64))
            vidx_list.append(np.asarray(leaf.vertex_indices, dtype=np.int64))

        self.n_leaves = len(self.leaves)
        self.dimension = dimension or 0

        if self.n_leaves == 0:
            self.leaf_vertex_indices = None
            self._V0 = None
            self._A_inv = None
            self._built = True
            return

        self.leaf_vertex_indices = np.stack(vidx_list, axis=0)  # (L, d+1)
        self._V0 = torch.as_tensor(np.stack(v0_list, axis=0),
                                   dtype=self.dtype, device=self.device)   # (L, d)
        self._A_inv = torch.as_tensor(np.stack(a_inv_list, axis=0),
                                      dtype=self.dtype, device=self.device)  # (L, d, d)
        self._built = True

    def _chunk_size(self) -> int:
        per_point = max(self.n_leaves * max(self.dimension, 1), 1)
        return max(1, self.element_budget // per_point)

    def embed(self, points: np.ndarray):
        """Embed a batch of points against every stacked leaf.

        Args:
            points: Array of shape ``(m, d)``.

        Returns:
            Tuple ``(leaf_index, found, alphas)`` where:
              * ``leaf_index`` (m,) int array indexes into ``self.leaves`` / rows of
                ``self.leaf_vertex_indices`` (``-1`` when unresolved),
              * ``found`` (m,) bool array marks GPU-resolved points,
              * ``alphas`` (m, d+1) float array holds the barycentric coordinates
                for the chosen leaf (rows for unresolved points are meaningless).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        m = points.shape[0]

        leaf_index = np.full(m, -1, dtype=np.int64)
        found = np.zeros(m, dtype=bool)
        alphas_out = np.zeros((m, self.dimension + 1), dtype=np.float64)

        if not self.is_built:
            return leaf_index, found, alphas_out

        tol = self.tolerance
        chunk = self._chunk_size()
        P_all = torch.as_tensor(points, dtype=self.dtype, device=self.device)

        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            P = P_all[start:end]                                   # (c, d)
            b = P[:, None, :] - self._V0[None, :, :]               # (c, L, d)
            alpha_r = torch.einsum("lij,clj->cli", self._A_inv, b)  # (c, L, d)
            alpha0 = 1.0 - alpha_r.sum(dim=-1)                     # (c, L)
            alphas = torch.cat([alpha0[..., None], alpha_r], dim=-1)  # (c, L, d+1)

            inside = (alphas >= -tol).all(dim=-1)                  # (c, L)
            min_alpha = alphas.amin(dim=-1)                        # (c, L)
            neg_inf = torch.full_like(min_alpha, float("-inf"))
            scored = torch.where(inside, min_alpha, neg_inf)       # (c, L)

            best_val, best_leaf = scored.max(dim=1)                # (c,), (c,)
            chunk_found = torch.isfinite(best_val)                 # (c,)

            rows = torch.arange(end - start, device=self.device)
            best_alphas = alphas[rows, best_leaf]                  # (c, d+1)

            cf = chunk_found.cpu().numpy()
            bl = best_leaf.cpu().numpy()
            ba = best_alphas.cpu().numpy()

            sl = slice(start, end)
            found[sl] = cf
            leaf_index[sl] = np.where(cf, bl, -1)
            alphas_out[sl] = ba

        return leaf_index, found, alphas_out
