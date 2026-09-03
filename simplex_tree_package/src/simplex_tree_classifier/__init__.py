"""simplex_tree_classifier - hierarchical simplex-tree classifier.

Public API:
    - ``SimplexTreeClassifier``: the main estimator (dataset or surrogate mode).
    - ``SimplexTree`` / ``Simplex`` / ``VertexRegistry``: the geometric core.
    - ``make_enclosing_simplex``: build an N-D simplex enclosing ``[0, 1]^d``.
    - ``get_device``: resolve the torch device used for the transform.
"""

from .vertex_registry import VertexRegistry
from .simplex import Simplex
from .simplex_tree import SimplexTree, make_enclosing_simplex
from .plane_equation import PlaneEquation
from .classifier import SimplexTreeClassifier
from .backend import get_device

__version__ = "0.1.0"

__all__ = [
    "SimplexTreeClassifier",
    "SimplexTree",
    "Simplex",
    "VertexRegistry",
    "PlaneEquation",
    "make_enclosing_simplex",
    "get_device",
    "__version__",
]
