from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from embedding.classes.simplex_tree import SimplexTree



def visualize_simplex_tree(tree, splitting_point: Tuple[float, float] = None,
                          data_points: List[Tuple[float, float]] = None,
                          title: str = "Simplex Tree Visualization",
                          figsize: Tuple[int, int] = (12, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    vertices = tree.get_vertices_as_tuples()
    _visualize_2d_simplex(vertices, ax, 'red', alpha=0.3, linewidth=2, s=20, label='Original vertices')

    _visualize_2d_children_recursive(tree, ax, colors, depth=0)

    if splitting_point:
        ax.scatter([splitting_point[0]], [splitting_point[1]],
                  color='yellow', s=200, marker='*', label='Splitting point')

    if data_points:
        if isinstance(data_points, list) and len(data_points) > 0:
            if hasattr(data_points[0], '__iter__') and len(data_points[0]) == 2:
                x_coords = [point[0] for point in data_points]
                y_coords = [point[1] for point in data_points]
            else:
                x_coords = [data_points[0]]
                y_coords = [data_points[1]]

            ax.scatter(x_coords, y_coords,
                      color='black', s=80, marker='o', alpha=0.8,
                      edgecolors='white', linewidth=1, label='Data points', zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _visualize_2d_simplex(vertices: List[Tuple[float, float]], ax, color: str, alpha: float = 0.3, 
                         linewidth: int = 1, s: int = 50, label: str = None):
    n_vertices = len(vertices)
    
    if n_vertices == 0:
        return
    elif n_vertices == 1:
        x, y = vertices[0]
        ax.scatter([x], [y], color=color, s=s*2, label=label)
    elif n_vertices == 2:
        x, y = zip(*vertices)
        ax.plot(x, y, color=color, linewidth=linewidth*2, alpha=alpha, label=label)
        ax.scatter(x, y, color=color, s=s, alpha=alpha)
    elif n_vertices == 3:
        triangle = patches.Polygon(vertices, facecolor=color, alpha=alpha, 
                                 edgecolor='black', linewidth=linewidth)
        ax.add_patch(triangle)
        x, y = zip(*vertices)
        ax.scatter(x, y, color=color, s=s, alpha=alpha, label=label)
    else:
        x, y = zip(*vertices)
        ax.scatter(x, y, color=color, s=s, alpha=alpha, label=label)
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                edge_x = [vertices[i][0], vertices[j][0]]
                edge_y = [vertices[i][1], vertices[j][1]]
                ax.plot(edge_x, edge_y, color=color, linewidth=linewidth, alpha=alpha)


def _visualize_2d_children_recursive(node, ax, colors, depth: int):
    for i, child in enumerate(node.get_children()):
        child_vertices = child.get_vertices_as_tuples()
        color = colors[(depth + i) % len(colors)]
        _visualize_2d_simplex(child_vertices, ax, color, alpha=0.2, linewidth=1, s=30)
        _visualize_2d_children_recursive(child, ax, colors, depth + 1)


def visualize_subdivision_levels(base_vertices: List[Tuple[float, float]],
                                  max_level: int = 3,
                                  data_points: List[Tuple[float, float]] = None,
                                  figsize: Tuple[int, int] = (12, 12)) -> None:


    levels = list(range(max_level + 1))[:4]
    n_plots = len(levels)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, level in zip(axes, levels):
        tree = SimplexTree(base_vertices)
        tree.add_barycentric_centers_recursively(level)
        _visualize_2d_simplex(base_vertices, ax, 'red', alpha=0.3, linewidth=2, s=20)
        _visualize_2d_children_recursive(tree, ax, ['blue','green','orange','purple'], depth=0)

        if data_points:
            x_coords = [p[0] for p in data_points]
            y_coords = [p[1] for p in data_points]
            ax.scatter(x_coords, y_coords, color='black', s=60, edgecolors='white', zorder=10)

        ax.set_aspect('equal')
        ax.set_title(f"Subdivision Level {level}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

