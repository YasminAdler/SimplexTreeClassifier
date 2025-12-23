import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.sparse import lil_matrix, issparse
from sklearn.svm import LinearSVC
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
in2d_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, in2d_dir)

from embedding.classes.simplex_tree import SimplexTree
from embedding.utilss.visualization import visualize_simplex_tree
from in2D.classifying.classes.utilss.plane_equation import PlaneEquation


class SimplexTreeClassifier:
    def __init__(self, vertices: List[Tuple[float, float]] = None, 
                 regularization=0.01, 
                 subdivision_levels=1, 
                 classifier_type='linear_svc'):
        if vertices is None:
            vertices = [(0, 0), (1, 0), (0.5, 1)]
        self.tree = SimplexTree(vertices)
        self.regularization = regularization
        self.subdivision_levels = subdivision_levels
        self.classifier_type = classifier_type
        self.classifier = None
        self.leaf_simplexes = []
        self._processed_pairs_cache = set()
        self.all_nodes_lookup = {}
        self._containing_simplex_cache = {}
        
        if self.subdivision_levels > 0:
            self.tree.add_barycentric_centers_recursively(self.subdivision_levels)
        self._build_node_lookup()
    
    @property
    def vertex_registry(self):
        return self.tree.vertex_registry
    
    @property
    def all_vertices(self) -> List[Tuple[float, float]]:
        return [self.vertex_registry.get_vertex_as_tuple(i) for i in range(len(self.vertex_registry))]
    
    @property
    def vertex_to_index(self) -> Dict[Tuple, int]:
        return self.vertex_registry.vertex_to_index

    def _build_node_lookup(self):
        self.all_nodes_lookup.clear()
        self.leaf_simplexes.clear()
        self._containing_simplex_cache.clear()
        
        for node in self.tree.traverse_breadth_first():
            vertex_key = frozenset(node.vertex_indices)
            self.all_nodes_lookup[vertex_key] = node
            if node.is_leaf():
                self.leaf_simplexes.append(node)


    
    def get_vertex_decision_values(self, vertices_list, weights):
        decision_values = []
        for vertex in vertices_list:
            vertex_tuple = tuple(vertex) if not isinstance(vertex, tuple) else vertex
            idx = self.vertex_to_index[vertex_tuple]
            decision_values.append(weights[idx])
        return np.array(decision_values)

    def _get_simplex_node(self, simplex_vertices):
        indices = self._vertices_to_indices(simplex_vertices)
        vertex_key = frozenset(indices)
        return self.all_nodes_lookup.get(vertex_key, None)

    def _vertices_to_indices(self, vertices):
        indices = []
        for vertex in vertices:
            vertex_tuple = tuple(vertex) if not isinstance(vertex, tuple) else vertex
            indices.append(self.vertex_to_index[vertex_tuple])
        return indices

    def find_adjacent_simplices(self, simplex_points): ## TODO: Check if the sibling of the parent is not found is the reason for the not finding the nonconvex
        simplex_node = self._get_simplex_node(simplex_points)
        if simplex_node is None:
            return []
        adjacent_simplex_indices = []
        dimension = len(simplex_points) - 1
        max_adjacent = dimension + 1
        simplex_indices_set = set(self._vertices_to_indices(simplex_points))
        visited_nodes = set()

        if simplex_node.parent is not None:
            for sibling in simplex_node.parent.children:
                if sibling != simplex_node and sibling.is_leaf():
                    sibling_indices = list(sibling.vertex_indices)
                    adjacent_simplex_indices.append(sibling_indices)
                    visited_nodes.add(id(sibling))
                    if len(adjacent_simplex_indices) >= max_adjacent:
                        return adjacent_simplex_indices[:max_adjacent]

        current_ancestor = simplex_node.parent
        while current_ancestor is not None and len(adjacent_simplex_indices) < max_adjacent:
            for descendant in current_ancestor.traverse_breadth_first():
                if (descendant.is_leaf() and
                    descendant != simplex_node and
                    id(descendant) not in visited_nodes):
                    descendant_indices_set = set(descendant.vertex_indices)
                    shared_indices = len(simplex_indices_set.intersection(descendant_indices_set))
                    if shared_indices == dimension:
                        descendant_indices = list(descendant.vertex_indices)
                        adjacent_simplex_indices.append(descendant_indices)
                        visited_nodes.add(id(descendant))
                        if len(adjacent_simplex_indices) >= max_adjacent:
                            break
            current_ancestor = current_ancestor.parent
        return adjacent_simplex_indices[:max_adjacent]

    def transform(self, data_points) -> lil_matrix:
        max_vertices = len(self.vertex_registry)
        barycentric_matrix = lil_matrix((len(data_points), max_vertices))
        for point_index in range(len(data_points)):
            point = tuple(data_points[point_index])
            if point in self._containing_simplex_cache:
                containing_simplex = self._containing_simplex_cache[point]
            else:
                containing_simplex = self.tree.find_containing_simplex(point)
                self._containing_simplex_cache[point] = containing_simplex
            if containing_simplex is not None:
                data_point_embeddings = containing_simplex.embed_point(point)
                for local_idx, coordinate in enumerate(data_point_embeddings):
                    global_idx = containing_simplex.vertex_indices[local_idx]
                    barycentric_matrix[point_index, global_idx] = coordinate
        return barycentric_matrix

    def _compute_decision_values(self, feature_matrix) -> np.ndarray:
        if issparse(feature_matrix):
            feature_matrix = feature_matrix.toarray()
        coef = getattr(self.classifier, "coef_", None)
        if coef is None:
            raise AttributeError("Classifier does not expose coef_ .")
        coef_vector = np.asarray(coef).reshape(-1)
        decision_values = feature_matrix @ coef_vector
        return np.asarray(decision_values).reshape(-1)

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-10)
        return normalized

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_normalized = self.normalize_data(X)
        self._processed_pairs_cache.clear()
        X_transformed = self.transform(X_normalized)
        print(X_transformed.shape)
        if self.classifier_type == 'linear_svc':
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

    def visualize_with_data_points(self, data_points: np.ndarray, title: str = "Simplex Tree with Data Points", figsize: Tuple[int, int] = (8, 5)):
        data_points_list = [tuple(point) for point in data_points]
        visualize_simplex_tree(self.tree, data_points=data_points_list, title=title, figsize=figsize)

    def get_simplex_boundaries(self) -> List[List[Tuple[float, float]]]:
        boundaries = []
        for leaf in self.leaf_simplexes:
            vertices = leaf.get_vertices_as_tuples()
            if len(vertices) >= 3:
                boundaries.append(vertices)
        return boundaries

    def _simplex_crosses_boundary(self, simplex_node, weights, intercept) -> bool:
        decision_values = [weights[idx] + intercept for idx in simplex_node.vertex_indices]
        has_positive = any(val > 0 for val in decision_values)
        has_negative = any(val < 0 for val in decision_values)
        return has_positive and has_negative

    def _get_simplex_class(self, simplex_node, weights, intercept) -> bool:
        vals = [weights[i] + intercept for i in simplex_node.vertex_indices]
        is_positive = any(v > 0 for v in vals) or all(v == 0 for v in vals)
        return is_positive

    def _are_siblings_same_side(self, parent_node, weights, intercept) -> bool:
        if not parent_node.children:
            return False
        
        child0 = parent_node.children[0]
        if self._simplex_crosses_boundary(child0, weights, intercept):
            return False
        
        first_child_class = self._get_simplex_class(child0, weights, intercept)
        
        for child in parent_node.children[1:]:
            if self._simplex_crosses_boundary(child, weights, intercept):
                return False
            
            child_class = self._get_simplex_class(child, weights, intercept)
            if child_class != first_child_class:
                return False
                
        return True

    def identify_svm_crossing_simplices(self) -> List[Dict]:
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        crossing_simplices = []
        weights = self.classifier.coef_[0]
        intercept = self.classifier.intercept_[0]
        for leaf in self.leaf_simplexes:
            if self._simplex_crosses_boundary(leaf, weights, intercept):
                vertices = leaf.get_vertices_as_tuples()
                decision_values = self.get_vertex_decision_values(vertices, weights)
                crossing_simplices.append({
                    'simplex': leaf,
                    'vertices': vertices,
                    'decision_values': decision_values,
                })
        return crossing_simplices

    def compute_svm_plane_equations(self) -> List[Dict]:
        crossing_simplices = self.identify_svm_crossing_simplices()
        weights = self.classifier.coef_[0]
        plane_equations = []
        for crossing_info in crossing_simplices:
            simplex = crossing_info['simplex']
            plane_eq = PlaneEquation(simplex)
            plane_coefficients = plane_eq.compute_plane_from_weights(weights)
            plane_equations.append({
                'simplex': simplex,
                'vertices': crossing_info['vertices'],
                'plane_equation': plane_eq,
                'coefficients': plane_coefficients,
                'cartesian_form': plane_eq.get_cartesian_form()
            })
        return plane_equations

    def find_same_side_simplices(self) -> set:
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet.")
        weights = self.classifier.coef_[0]
        intercept = self.classifier.intercept_[0]
        
        same_side_keys = set()
        leaf_parents = set()
        for leaf in self.leaf_simplexes:
            if leaf.parent:
                leaf_parents.add(leaf.parent)
        
        for parent in leaf_parents:
            if self._are_siblings_same_side(parent, weights, intercept):
                for child in parent.children:
                    same_side_keys.add(frozenset(child.vertex_indices))
        
        return same_side_keys