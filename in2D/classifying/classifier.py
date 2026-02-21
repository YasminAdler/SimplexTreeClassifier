import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.svm import SVC
import sys
import os
from typing import Tuple, List, Dict, Optional

current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from classes.simplex_tree_classifier import SimplexTreeClassifier


def make_meshgrid(x: np.ndarray, y: np.ndarray, h: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx: np.ndarray, yy: np.ndarray, xy: np.ndarray, **params):
    Z = clf.predict(xy)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def generate_dataset(n_samples: int = 1000, 
                     dimension: int = 2,
                     polytope_order: int = 8,
                     gamma: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:

    X = np.random.normal(0, 1, (n_samples, dimension))
    ws = np.random.normal(0, 1, (polytope_order, dimension))
    ws = ws / (np.linalg.norm(ws, axis=1).reshape(-1, 1))
    
    u = np.random.uniform(0, 1, (n_samples, 1))
    X = X / (np.linalg.norm(X, axis=1).reshape(-1, 1)) * (u**(1/dimension))
    
    y = np.ones(n_samples)
    
    for index in range(n_samples):
        z = 1
        for w in ws:
            if (X[index].dot(w) - 0.5 - gamma) > 0:
                z = -1
            else:
                if (X[index].dot(w) - 0.4 - gamma) > 0:
                    z = 0
        y[index] = z
    
    # Remove zero class and rescale
    X = (X[y != 0] + [1, 1]) / 2
    y = y[y != 0]
    
    print(f"Dataset created: X shape: {X.shape}, y shape: {y.shape}, Classes: {np.unique(y)}")
    return X, y


def draw_simplex_boundaries(ax, boundaries: List[List[Tuple]], 
                           style: str = 'ro-', alpha: float = 0.5, 
                           linewidth: float = 0.5):

    for boundary in boundaries:
        if len(boundary) >= 3:
            closed_boundary = boundary + [boundary[0]]
            x_coords = [point[0] for point in closed_boundary]
            y_coords = [point[1] for point in closed_boundary]
            ax.plot(x_coords, y_coords, style, alpha=alpha, linewidth=linewidth)


def draw_svm_boundary_segments(ax, model, plane_equations: List[Dict]):
    for plane_info in plane_equations:
        vertices = plane_info['vertices']
        
        if len(vertices) >= 3:
            closed_verts = vertices + [vertices[0]]
            x_coords = [v[0] for v in closed_verts]
            y_coords = [v[1] for v in closed_verts]
            ax.fill(x_coords, y_coords, color='yellow', alpha=0.2, 
                   edgecolor='orange', linewidth=1.5)
        
        decision_values = []
        for v in vertices:
            v_transformed = model.transform(np.array([v]))
            d_val = model.classifier.decision_function(v_transformed)[0]
            decision_values.append(d_val)
        
        crossings = []
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            v1, v2 = vertices[i], vertices[j]
            d1, d2 = decision_values[i], decision_values[j]
            
            if d1 * d2 < 0:
                t = abs(d1) / (abs(d1) + abs(d2))
                crossing = (v1[0] + t * (v2[0] - v1[0]), 
                           v1[1] + t * (v2[1] - v1[1]))
                crossings.append(crossing)
        
        if len(crossings) >= 2:
            ax.plot([crossings[0][0], crossings[1][0]], 
                   [crossings[0][1], crossings[1][1]], 
                   'lime', linewidth=2, alpha=0.9)


def draw_svm_plane_lines(ax, plane_equations: List[Dict], 
                        plot_limits: Tuple[float, float] = (0, 1)):

    for plane_info in plane_equations:
        coefficients = plane_info['coefficients']
        a, b, c = coefficients
        
        if abs(b) > abs(a):
            x_line = np.array(plot_limits)
            y_line = -(a * x_line + c) / b
        else:
            y_line = np.array(plot_limits)
            x_line = -(b * y_line + c) / a
        
        ax.plot(x_line, y_line, 'lime', linewidth=2.5, alpha=0.8, zorder=10)


def visualize_subdivision_level(ax, X: np.ndarray, y: np.ndarray, 
                                subdivision_level: int, regularization: float,
                                show_segments: bool = True,
                                show_full_lines: bool = False):

    # Create and train model
    model = SimplexTreeClassifier(
        vertices=[(0, 0), (2, 0), (0, 2)],
        classifier=SVC(C=regularization),
        subdivision_levels=subdivision_level,
    )
    model.fit(X, y)
    
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    xy = model.transform(np.vstack([xx.ravel(), yy.ravel()]).T)
    
    plot_contours(ax, model.classifier, xx, yy, xy,
                 cmap=plt.cm.coolwarm, alpha=0.8)
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, 
              s=20, edgecolors='k')
    
    if hasattr(model.classifier, "decision_function"):
        Z = model.classifier.decision_function(xy).reshape(xx.shape)
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
                  linestyles=['--', '-', '--'])
    
    boundaries = model.get_simplex_boundaries()
    draw_simplex_boundaries(ax, boundaries, style='ro-', alpha=0.5, linewidth=0.5)
    
    try:
        plane_equations = model.compute_svm_plane_equations()
        
        if show_segments:
            draw_svm_boundary_segments(ax, model, plane_equations)
        
        if show_full_lines:
            draw_svm_plane_lines(ax, plane_equations)
        
        if len(plane_equations) > 0:
            ax.text(0.02, 0.88, f'SVM crosses {len(plane_equations)} simplices', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    except Exception as e:
        print(f"Could not visualize SVM planes for level {subdivision_level}: {e}")
    
    accuracy = model.classifier.score(model.transform(X), y)
    ax.text(0.02, 0.98, f'Acc: {accuracy:.3f}', transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Subdivision Level {subdivision_level}")
    ax.set_aspect('equal')
    
    return model


def visualize_all_levels_no_lines(X: np.ndarray, y: np.ndarray, 
                                  regularization: float = 1000,
                                  figsize: Tuple[int, int] = (12, 12)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for k, ax in enumerate(axes.flatten(), 1):
        print(f"Running for subdivision_levels k={k}")
        visualize_subdivision_level(ax, X, y, k, regularization,
                                   show_segments=False, show_full_lines=False)
    
    plt.tight_layout()
    return fig


def visualize_all_levels_with_plane_lines(X: np.ndarray, y: np.ndarray, 
                                          regularization: float = 1000,
                                          figsize: Tuple[int, int] = (12, 12)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for k, ax in enumerate(axes.flatten(), 1):
        print(f"Running for subdivision_levels k={k}")
        model = SimplexTreeClassifier(
            vertices=[(0, 0), (2, 0), (0, 2)],
            classifier=SVC(C=regularization),
            subdivision_levels=k,
        )
        model.fit(X, y)
        
        xx, yy = make_meshgrid(X[:, 0], X[:, 1])
        xy = model.transform(np.vstack([xx.ravel(), yy.ravel()]).T)
        
        plot_contours(ax, model.classifier, xx, yy, xy,
                     cmap=plt.cm.coolwarm, alpha=0.8)
        
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Subdivision Level {k}")
        
        if hasattr(model.classifier, "decision_function"):
            Z = model.classifier.decision_function(xy).reshape(xx.shape)
            ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
                      linestyles=['--', '-', '--'])
        
        boundaries = model.get_simplex_boundaries()
        for boundary in boundaries:
            if len(boundary) >= 3:
                closed_boundary = boundary + [boundary[0]]
                x_coords = [point[0] for point in closed_boundary]
                y_coords = [point[1] for point in closed_boundary]
                ax.plot(x_coords, y_coords, 'ro-')
        
        plane_equations = model.compute_svm_plane_equations()
        
        for plane_info in plane_equations:
            coefficients = plane_info['coefficients']
            a, b, c = coefficients
            
            if abs(b) > abs(a):
                x_line = np.array([0, 1])
                y_line = -(a * x_line + c) / b
            else:
                y_line = np.array([0, 1])
                x_line = -(b * y_line + c) / a
            
            ax.plot(x_line, y_line, 'lime', linewidth=2.5, alpha=0.8, zorder=10)
        
        accuracy = model.classifier.score(model.transform(X), y)
        ax.text(0.02, 0.98, f'Acc: {accuracy:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    X, y = generate_dataset(n_samples=1000, dimension=2, 
                           polytope_order=8, gamma=0.01)
    
    C = 1000  # Regularization parameter
    
    print("\nVisualization without plane lines")
    fig1 = visualize_all_levels_no_lines(X, y, regularization=C)
    plt.show()
    
    print("\nVisualization with full plane lines")
    fig2 = visualize_all_levels_with_plane_lines(X, y, regularization=C)
    plt.show()