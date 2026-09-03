"""Optional 2-D visualization helpers (requires the ``viz`` extra: matplotlib).

These only make sense for 2-dimensional data. Import lazily so the core package
works without matplotlib installed.
"""

from typing import List, Optional, Tuple


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import matplotlib.patches as patches  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Visualization requires matplotlib. Install the extra with "
            "`pip install simplex-tree-classifier[viz]`."
        ) from exc
    return plt, patches


def _draw_2d_simplex(vertices, ax, color, alpha=0.3, linewidth=1, s=50, label=None):
    import matplotlib.patches as patches
    n = len(vertices)
    if n == 0:
        return
    if n == 1:
        x, y = vertices[0]
        ax.scatter([x], [y], color=color, s=s * 2, label=label)
    elif n == 2:
        x, y = zip(*vertices)
        ax.plot(x, y, color=color, linewidth=linewidth * 2, alpha=alpha, label=label)
        ax.scatter(x, y, color=color, s=s, alpha=alpha)
    elif n == 3:
        triangle = patches.Polygon(vertices, facecolor=color, alpha=alpha,
                                   edgecolor="black", linewidth=linewidth)
        ax.add_patch(triangle)
        x, y = zip(*vertices)
        ax.scatter(x, y, color=color, s=s, alpha=alpha, label=label)
    else:
        x, y = zip(*vertices)
        ax.scatter(x, y, color=color, s=s, alpha=alpha, label=label)
        for i in range(n):
            for j in range(i + 1, n):
                ax.plot([vertices[i][0], vertices[j][0]],
                        [vertices[i][1], vertices[j][1]],
                        color=color, linewidth=linewidth, alpha=alpha)


def _draw_children_recursive(node, ax, colors, depth=0):
    for i, child in enumerate(node._get_children()):
        color = colors[(depth + i) % len(colors)]
        _draw_2d_simplex(child.get_vertices_as_tuples(), ax, color,
                         alpha=0.2, linewidth=1, s=30)
        _draw_children_recursive(child, ax, colors, depth + 1)


def visualize_simplex_tree(tree, data_points=None,
                           title: str = "Simplex Tree",
                           figsize: Tuple[int, int] = (10, 8)):
    """Plot a 2-D simplex tree and (optionally) overlaid data points."""
    plt, _ = _require_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["red", "blue", "green", "orange", "purple", "brown",
              "pink", "gray", "cyan", "magenta"]

    _draw_2d_simplex(tree.get_vertices_as_tuples(), ax, "red", alpha=0.3,
                     linewidth=2, s=20, label="Root simplex")
    _draw_children_recursive(tree, ax, colors, depth=0)

    if data_points is not None and len(data_points) > 0:
        xs = [p[0] for p in data_points]
        ys = [p[1] for p in data_points]
        ax.scatter(xs, ys, color="black", s=40, alpha=0.8,
                   edgecolors="white", linewidth=1, label="Data", zorder=10)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_leaf_boundaries(classifier, ax=None,
                         color: str = "red", linewidth: float = 0.5):
    """Draw the edges of every leaf simplex of a fitted 2-D classifier."""
    plt, _ = _require_matplotlib()
    if classifier.dimension != 2:
        raise ValueError("plot_leaf_boundaries only supports 2-D classifiers.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    for boundary in classifier.get_simplex_vertices():
        if len(boundary) >= 3:
            closed = list(boundary) + [boundary[0]]
            xs = [p[0] for p in closed]
            ys = [p[1] for p in closed]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.6)
    ax.set_aspect("equal")
    return ax
