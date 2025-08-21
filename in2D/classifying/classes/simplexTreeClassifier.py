import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import lil_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
from ucimlrepo import fetch_ucirepo

current_dir = os.path.dirname(os.path.abspath(__file__))
in2d_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, in2d_dir)

from embadding.classes.simplexTree import SimplexTree
from embadding.utilss.visualization import visualize_simplex_tree


class SimplexTreeClassifier:
    def __init__(self, vertices: List[Tuple[float, float]] = None, 
                 regularization=1.0, 
                 subdivision_levels=1, 
                 classifier_type='svc'):
        if vertices is None:
            vertices = [(0, 0), (1, 0), (0.5, 1)]   # default to unit triangle if no vertices provided
        
        self.tree = SimplexTree(vertices)
        self.dimension = 2  
        self.regularization = regularization
        self.subdivision_levels = subdivision_levels
        self.classifier_type = classifier_type
        self.classifier = None
        self.leaf_simplexes = []
        self.parent_simplexes = None 
        
        if self.subdivision_levels > 0:
            self.tree.add_barycentric_centers_recursively(self.subdivision_levels)
            
        self._update_leaf_simplexes()
    
    def _update_leaf_simplexes(self):
        self.leaf_simplexes = self.tree.get_leaves()
    
    def transform(self, X: np.ndarray) -> lil_matrix:
        n_samples = X.shape[0]
        n_vertices = sum(len(simplex.vertices) for simplex in self.leaf_simplexes)
        
        barycentric_matrix = lil_matrix((n_samples, n_vertices))
        self.parent_simplexes = np.zeros(n_samples, dtype=int)

        for point_idx in range(n_samples):
            point = tuple(X[point_idx])
            
            for simplex_idx, simplex in enumerate(self.leaf_simplexes):
                if simplex.point_inside_simplex(point):
                    barycentric_coords = simplex.embed_point(point)
                    if barycentric_coords is not None:
                        vertex_start_index = sum(len(self.leaf_simplexes[k].vertices) 
                                            for k in range(simplex_idx))
                        
                        for vertex_idx, barycentric_weight in enumerate(barycentric_coords):
                            barycentric_matrix[point_idx, vertex_start_index + vertex_idx] = barycentric_weight
                        
                        self.parent_simplexes[point_idx] = simplex_idx
                        break

        return barycentric_matrix
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-10)
        
        sums = normalized[:, 0] + normalized[:, 1]
        scale_factors = np.maximum(sums, 1.0)
        normalized = normalized / scale_factors[:, np.newaxis]
        
        return normalized
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_normalized = self.normalize_data(X)
        
        X_transformed = self.transform(X_normalized)
        
        if self.classifier_type == 'svc':
            self.classifier = SVC(C=self.regularization, kernel='linear')
        elif self.classifier_type == 'linear_svc':
            self.classifier = LinearSVC(C=self.regularization)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        self.classifier.fit(X_transformed, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        
        X_normalized = self.normalize_data(X)
        X_transformed = self.transform(X_normalized)
        predictions = self.classifier.predict(X_transformed)
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.classifier, 'decision_function'):
            raise AttributeError(f"{self.classifier_type} does not have decision_function")
        
        X_normalized = self.normalize_data(X)
        X_transformed = self.transform(X_normalized)
        return self.classifier.decision_function(X_transformed)
    
    def get_tree_stats(self) -> Dict:
        height = self.tree.get_depth()
        leaves = len(self.leaf_simplexes)
        total_nodes = self.tree.get_node_count()
        
        branching_factor = self.dimension + 1  
        splitting_points = (branching_factor**height - 1) / (branching_factor - 1) if height > 0 else 0
        
        return {
            'dimension': self.dimension,
            'height': height,
            'branching_factor': branching_factor,
            'splitting_points': splitting_points,
            'total_simplexes': total_nodes,
            'leaf_simplexes': leaves,
            'barycentric_features': sum(len(simplex.vertices) for simplex in self.leaf_simplexes)
        }
    
    def get_simplexes_info(self) -> List[Dict]:
        info = []
        for i, simplex in enumerate(self.leaf_simplexes):
            info.append({
                'index': i,
                'vertices': simplex.get_vertices_as_tuples(),
                'n_vertices': len(simplex.vertices)
            })
        return info
    
    def make_meshgrid(self, x, y, h=0.01):
        x_min, x_max = x.min() - 0.3, x.max() + 0.3
        y_min, y_max = y.min() - 0.3, y.max() + 0.3
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    
    def plot_decision_boundaries(self, ax, clf, xx, yy, xy, **params):
        Z = clf.predict(xy)
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    
    def visualize_classification(self, X_train, y_train, title="SimplexTree Classification"):
        if self.classifier is None:
            print("Classifier not trained yet. Call fit() first.")
            return
            
        fig, ax = plt.subplots(figsize=(12, 10))
        xx, yy = self.make_meshgrid(X_train[:, 0], X_train[:, 1])       
        mesh_points = np.vstack([xx.ravel(), yy.ravel()]).T
        mesh_transformed = self.transform(self.normalize_data(mesh_points))
        self.plot_decision_boundaries(ax, self.classifier, xx, yy, mesh_transformed,
                                    cmap=plt.cm.coolwarm, alpha=0.6)
        if hasattr(self.classifier, 'decision_function'):
            Z_decision = self.classifier.decision_function(mesh_transformed).reshape(xx.shape)            
            ax.contour(xx, yy, Z_decision, colors='k', levels=[-1, 0, 1], alpha=0.8,
                      linestyles=['--', '-', '--'])
        
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                           cmap=plt.cm.coolwarm, s=50, edgecolors='black', alpha=0.8)
        
        self._draw_simplex_boundaries(ax)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_simplex_boundaries(self, ax):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, simplex in enumerate(self.leaf_simplexes):
            vertices = simplex.get_vertices_as_tuples()
            color = colors[i % len(colors)]
            
            if len(vertices) >= 3:
                triangle = patches.Polygon(vertices, facecolor='none', 
                                         edgecolor=color, linewidth=2, alpha=0.8)
                ax.add_patch(triangle)
            x_coords, y_coords = zip(*vertices)
            ax.scatter(x_coords, y_coords, color=color, s=100, 
                      marker='o', edgecolors='black', linewidth=1, alpha=0.9,
                      zorder=5)
    
    def visualize_tree_and_classification(self, X_train, y_train):
        visualize_simplex_tree(self.tree, None, "SimplexTree Structure")
        
        self.visualize_classification(X_train, y_train, 
                                    f"Classification with {len(self.leaf_simplexes)} Simplexes")
    

def generate_points_in_triangle(vertices, n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r1 = np.random.random(n_points)
    r2 = np.random.random(n_points)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    r3 = 1 - r1 - r2
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        points[i] = (r1[i] * vertices[0][0] + r2[i] * vertices[1][0] + r3[i] * vertices[2][0],
                     r1[i] * vertices[0][1] + r2[i] * vertices[1][1] + r3[i] * vertices[2][1])
    return points

if __name__ == "__main__":
    # Test 1: my generation of data
    triangle_vertices = [(0, 0), (1, 0), (0.5, 1)]
    np.random.seed(42)
    num_samples = 1000
    boundary_margin = 0.01
    dimension = 2
    num_hyperplanes = 8
    
    X = np.random.normal(0, 1, (num_samples, dimension))
    hyperplane_normals = np.random.normal(0, 1, (num_hyperplanes, dimension))
    hyperplane_normals = hyperplane_normals / (np.linalg.norm(hyperplane_normals, axis=1).reshape(-1, 1))
    
    radial_distances = np.random.uniform(0, 1, (num_samples, 1))
    X = X / (np.linalg.norm(X, axis=1).reshape(-1, 1)) * (radial_distances**(1/dimension))
    
    y = np.ones(num_samples)
    
    for index in range(num_samples):
        class_label = 1
        for hyperplane_normal in hyperplane_normals:
            if (X[index].dot(hyperplane_normal) - 0.5 - boundary_margin) > 0:
                class_label = -1
            else:
                if (X[index].dot(hyperplane_normal) - 0.4 - boundary_margin) > 0:
                    class_label = 0
        y[index] = class_label
    
    X = X[y != 0]
    y = y[y != 0]
    
    X = (X + [1, 1]) / 2
    
    valid_points = []
    valid_labels = []
    for i, point in enumerate(X):
        if (point[0] >= 0 and point[1] >= 0 and 
            point[0] <= 1 and point[1] <= 1 and
            point[1] <= 2 * (0.5 - abs(point[0] - 0.5))):
            valid_points.append(point)
            valid_labels.append(y[i])
    
    X_synthetic = np.array(valid_points)
    y_synthetic = np.array(valid_labels)
    y_synthetic = (y_synthetic + 1) // 2
    
    # Test 2: Iris dataset
    iris = fetch_ucirepo(id=53)
    X_iris_full = iris.data.features.values
    y_iris_full = iris.data.targets.values.ravel()
    
    X_iris = X_iris_full[:, :2]  # Use first 2 features
    y_iris_binary = (y_iris_full == 'Iris-setosa').astype(int)
    
    X_iris_min, X_iris_max = X_iris.min(axis=0), X_iris.max(axis=0)
    X_iris_normalized = (X_iris - X_iris_min) / (X_iris_max - X_iris_min)
    X_iris_scaled = X_iris_normalized * 0.8 + 0.1
    
    iris_valid_points = []
    iris_valid_labels = []
    for i, point in enumerate(X_iris_scaled):
        if (point[0] >= 0 and point[1] >= 0 and 
            point[0] <= 1 and point[1] <= 1 and
            point[1] <= 2 * (0.5 - abs(point[0] - 0.5))):
            iris_valid_points.append(point)
            iris_valid_labels.append(y_iris_binary[i])
    
    X_iris_final = np.array(iris_valid_points)
    y_iris_final = np.array(iris_valid_labels)
    
    # Test both datasets
    datasets = [
        (X_synthetic, y_synthetic, "NBCSV3 Synthetic"),
        (X_iris_final, y_iris_final, "Iris Dataset")
    ]
    
    subdivision_levels = [0, 1, 2, 3]
    
    for X, y, dataset_name in datasets:
        for level in subdivision_levels:
            classifier = SimplexTreeClassifier(
                vertices=None,
                regularization=0.1,
                subdivision_levels=level,
                classifier_type='svc'
            )
            classifier.fit(X, y)
            title = f"{dataset_name} - Level {level}"
            classifier.visualize_tree_and_classification(X, y)
        

