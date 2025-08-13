from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np


def visualize_simplex_tree(tree, splitting_point: Tuple[float, float] = None, title: str = "Simplex Tree Visualization"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    vertices = tree.get_vertices_as_tuples()
    _visualize_2d_simplex(vertices, ax, 'red', alpha=0.3, linewidth=2, s=100, label='Original vertices')
    
    _visualize_2d_children_recursive(tree, ax, colors, depth=0)
    
    if splitting_point:
        ax.scatter([splitting_point[0]], [splitting_point[1]], 
                  color='yellow', s=200, marker='*', label='Splitting point')
    
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


def visualize_triangle_with_point(triangle_vertices: List[Tuple[float, float]], 
                                 point: Tuple[float, float], 
                                 title: str = "Triangle with Point"):           
    fig, ax = plt.subplots(figsize=(10, 8))
    
    triangle = patches.Polygon(triangle_vertices, facecolor='lightblue', alpha=0.3, 
                             edgecolor='blue', linewidth=2)
    ax.add_patch(triangle)
    
    x, y = zip(*triangle_vertices)
    ax.scatter(x, y, color='red', s=100, zorder=5, label='Triangle vertices')
    
    ax.scatter([point[0]], [point[1]], color='green', s=150, marker='*', 
              zorder=6, label='Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_multiple_triangles(triangles: List[List[Tuple[float, float]]], 
                                points: List[Tuple[float, float]] = None,
                                title: str = "Multiple Triangles"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(triangles)))
    
    for i, triangle_vertices in enumerate(triangles):
        triangle = patches.Polygon(triangle_vertices, facecolor=colors[i], alpha=0.3, 
                                 edgecolor='black', linewidth=1)
        ax.add_patch(triangle)
        
        x, y = zip(*triangle_vertices)
        ax.scatter(x, y, color='red', s=50, zorder=5)
    
    if points:
        for point in points:
            ax.scatter([point[0]], [point[1]], color='green', s=100, marker='*', 
                      zorder=6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() 