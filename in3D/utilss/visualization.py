from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_simplex_tree(tree, splitting_point: Tuple[float, ...] = None, title: str = "Simplex Tree Visualization"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    vertices = tree.get_vertices_as_tuples()
    _visualize_3d_simplex(vertices, ax, 'red', alpha=0.3, linewidth=2, s=100, label='Original vertices')
    
    _visualize_3d_children_recursive(tree, ax, colors, depth=0)
    
    if splitting_point:
        ax.scatter([splitting_point[0]], [splitting_point[1]], [splitting_point[2]], 
                  color='yellow', s=200, marker='*', label='Splitting point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def _visualize_3d_simplex(vertices: List[Tuple[float, ...]], ax, color: str, alpha: float = 0.3, 
                         linewidth: int = 1, s: int = 50, label: str = None):
    n_vertices = len(vertices)
    
    if n_vertices == 0:
        return
    elif n_vertices == 1:
        x, y, z = vertices[0]
        ax.scatter([x], [y], [z], color=color, s=s*2, label=label)
    elif n_vertices == 2:
        x, y, z = zip(*vertices)
        ax.plot(x, y, z, color=color, linewidth=linewidth*2, alpha=alpha, label=label)
        ax.scatter(x, y, z, color=color, s=s, alpha=alpha)
    elif n_vertices == 3:
        x, y, z = zip(*vertices)
        triangle = Poly3DCollection([vertices], alpha=alpha, facecolor=color, 
                                   edgecolor='black', linewidth=linewidth)
        ax.add_collection3d(triangle)
        ax.scatter(x, y, z, color=color, s=s, alpha=alpha, label=label)
    elif n_vertices == 4:
        faces = _get_tetrahedron_faces(vertices)
        poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                                 edgecolor='black', linewidth=linewidth)
        ax.add_collection3d(poly3d)
        x, y, z = zip(*vertices)
        ax.scatter(x, y, z, color=color, s=s, alpha=alpha, label=label)
    else:
        x, y, z = zip(*vertices)
        ax.scatter(x, y, z, color=color, s=s, alpha=alpha, label=label)
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                edge_x = [vertices[i][0], vertices[j][0]]
                edge_y = [vertices[i][1], vertices[j][1]]
                edge_z = [vertices[i][2], vertices[j][2]]
                ax.plot(edge_x, edge_y, edge_z, color=color, linewidth=linewidth, alpha=alpha)


def _visualize_3d_children_recursive(node, ax, colors, depth: int):
    for i, child in enumerate(node.get_children()):
        child_vertices = child.get_vertices_as_tuples()
        color = colors[(depth + i) % len(colors)]
        _visualize_3d_simplex(child_vertices, ax, color, alpha=0.2, linewidth=1, s=30)
        _visualize_3d_children_recursive(child, ax, colors, depth + 1)


def _get_tetrahedron_faces(vertices: List[Tuple[float, ...]]) -> List[List[Tuple[float, ...]]]:
    if len(vertices) != 4:
        raise ValueError("Tetrahedron must have exactly 4 vertices")
    
    faces = [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]]
    ]
    return faces

