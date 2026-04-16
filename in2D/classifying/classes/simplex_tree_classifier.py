import numpy as np
from typing import List, Tuple, Dict, Optional, Set
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
from in2D.classifying.classes.utilss.convexity_check import check_convexity

class SimplexTreeClassifier: 
    def __init__(self, vertices: List[Tuple[float, ...]] = None,
                 classifier=None,
                 subdivision_levels=1):
        """
        Initialize SimplexTreeClassifier.
        
        Args:
            vertices: Initial simplex vertices
            classifier: Any sklearn-compatible estimator instance with .fit() and .predict().
                        Examples: LinearSVC(C=1000), SVC(kernel='rbf'), OneClassSVM(kernel='linear', nu=0.1)
                        Defaults to LinearSVC(C=1.0) if None.
            subdivision_levels: Number of barycentric subdivisions
        """
        if vertices is None:
            vertices = [(0, 0), (2, 0), (0, 2)]
        self.tree = SimplexTree(vertices)
        self.subdivision_levels = subdivision_levels
        self.classifier = classifier if classifier is not None else LinearSVC(C=1.0)
        self.leaf_simplexes = []
        self.all_nodes_lookup = {}
        self._containing_simplex_cache = {}
        
        if self.subdivision_levels > 0:
            self.tree._add_barycentric_centers_recursively(self.subdivision_levels)
        self._build_node_lookup()
    
    def _build_node_lookup(self):
        self.all_nodes_lookup.clear()
        self.leaf_simplexes.clear()
        self._containing_simplex_cache.clear()
        
        for node in self.tree._traverse_breadth_first():
            vertex_key = frozenset(node.vertex_indices)
            self.all_nodes_lookup[vertex_key] = node
            if node._is_leaf():
                self.leaf_simplexes.append(node)

    
    def _get_simplex_node(self, simplex_vertices):
        indices = self._vertices_to_indices(simplex_vertices)
        vertex_key = frozenset(indices)
        return self.all_nodes_lookup.get(vertex_key, None)

    def _vertices_to_indices(self, vertices):
        indices = []
        for vertex in vertices:
            vertex_tuple = tuple(vertex) if not isinstance(vertex, tuple) else vertex
            indices.append(self.tree.registry.vertex_to_index[vertex_tuple])
        return indices

    def transform(self, data_points) -> lil_matrix:
        """
        Converts data points to barycentric coordinate embeddings.
        
        For each point, finds its containing simplex and computes barycentric
        coordinates, placing them in a sparse matrix at the simplex's vertex indices.
        
        Args:
            data_points: Array of shape (n_samples, 2) with x,y coordinates
            
        Returns:
            Sparse matrix (n_samples x n_vertices) with barycentric embeddings.
            Each row has 3 non-zero values summing to 1.
        """
        max_vertices = len(self.tree.registry)
        barycentric_matrix = lil_matrix((len(data_points), max_vertices))
        for point_index in range(len(data_points)):
            point = tuple(data_points[point_index])
            if point in self._containing_simplex_cache:
                containing_simplex = self._containing_simplex_cache[point]
            else:
                containing_simplex = self.tree.find_containing_simplex(point)
                self._containing_simplex_cache[point] = containing_simplex
            if containing_simplex is not None:
                data_point_embeddings = containing_simplex._embed_point(point)
                for local_idx, coordinate in enumerate(data_point_embeddings):
                    global_idx = containing_simplex.vertex_indices[local_idx]
                    barycentric_matrix[point_index, global_idx] = coordinate
        return barycentric_matrix

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-10)
        return normalized

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Trains the classifier by embedding data points as barycentric coordinates.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,).
               Pass None for unsupervised classifiers (e.g. OneClassSVM).
            
        Returns:
            self (fitted classifier)
        """
        X_normalized = self._normalize_data(X)
        X_transformed = self.transform(X_normalized)
        
        if y is not None:
            self.classifier.fit(X_transformed, y)
        else:
            self.classifier.fit(X_transformed)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for input data points.
        
        Args:
            X: Data points of shape (n_samples, 2)
            
        Returns:
            Array of predicted class labels
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        X_normalized = self._normalize_data(X)
        X_transformed = self.transform(X_normalized)
        predictions = self.classifier.predict(X_transformed)
        return predictions

    def find_adjacent_simplexes(self, leaf) -> list:
        """
        Finds all leaf simplexes adjacent to the given leaf.
        
        Delegates to the underlying SimplexTree.
        
        Args:
            leaf: A leaf SimplexTree node
            
        Returns:
            List of adjacent leaf SimplexTree nodes
        """
        return self.tree.find_adjacent_simplexes(leaf)

    def get_simplex_boundaries(self) -> List[List[Tuple[float, ...]]]:
        """
        Gets vertex coordinates of all leaf simplices for visualization.
        
        Returns:
            List of vertex coordinate tuples for each leaf simplex
        """
        return [leaf.get_vertices_as_tuples() for leaf in self.leaf_simplexes]

    @property
    def is_linear_classifier(self):
        """Check if the internal classifier exposes linear weights (coef_ and intercept_)."""
        return (hasattr(self.classifier, 'coef_') and
                hasattr(self.classifier, 'intercept_'))

    def get_weights_and_intercept(self):
        """
        Public accessor for the fitted classifier's weight vector and intercept.
        Only works for linear classifiers (LinearSVC, LogisticRegression, Perceptron, etc.).
        Raises AttributeError for non-linear classifiers.
        """
        return self._get_weights_and_intercept()

    def _get_weights_and_intercept(self):
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        if not self.is_linear_classifier:
            raise AttributeError(
                f"{type(self.classifier).__name__} has no coef_ attribute. "
                "This method requires a linear classifier (e.g. LinearSVC, "
                "LogisticRegression, Perceptron)."
            )
        weights = self.classifier.coef_[0]
        if hasattr(weights, 'toarray'):
            weights = weights.toarray().flatten()
        elif hasattr(weights, 'A'):
            weights = np.asarray(weights).flatten()
        intercept = self.classifier.intercept_[0]
        return weights, intercept

    def _predict_at_vertices(self, simplex_node):
        """Evaluate the classifier at each vertex using one-hot barycentric vectors."""
        n_vertices = len(self.tree.registry)
        predictions = []
        for vid in simplex_node.vertex_indices:
            one_hot = lil_matrix((1, n_vertices))
            one_hot[0, vid] = 1.0
            predictions.append(self.classifier.predict(one_hot)[0])
        return np.array(predictions)

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

    def identify_crossing_simplices(self) -> List[Dict]:
        """
        Finds leaf simplices that cross the decision boundary.
        Works with any classifier — uses weight-based check for linear classifiers,
        falls back to prediction-based check for non-linear ones.

        Returns:
            List of dicts with keys: 'simplex', 'vertices'
            (also 'decision_values' when the classifier is linear)
        """
        use_linear = self.is_linear_classifier
        if use_linear:
            weights, intercept = self._get_weights_and_intercept()

        crossing_simplices = []
        for leaf in self.leaf_simplexes:
            if use_linear:
                crosses = self._simplex_crosses_boundary(leaf, weights, intercept)
            else:
                preds = self._predict_at_vertices(leaf)
                crosses = not np.all(preds == preds[0])

            if crosses:
                info = {
                    'simplex': leaf,
                    'vertices': leaf.get_vertices_as_tuples(),
                }
                if use_linear:
                    info['decision_values'] = np.array([weights[idx] for idx in leaf.vertex_indices])
                crossing_simplices.append(info)

        return crossing_simplices

    def identify_svm_crossing_simplices(self) -> List[Dict]:
        """Backward-compatible alias. Requires a linear classifier."""
        if not self.is_linear_classifier:
            raise AttributeError(
                f"{type(self.classifier).__name__} is not a linear classifier. "
                "Use identify_crossing_simplices() instead."
            )
        return self.identify_crossing_simplices()

    def compute_svm_plane_equations(self) -> List[Dict]:
        """
        Computes the decision boundary plane equation within each crossing simplex.
        
        For each simplex that crosses the boundary, computes the linear equation
        describing where the boundary intersects that simplex.
        
        Returns:
            List of dicts with keys: 'simplex', 'vertices', 'plane_equation', 'coefficients', 'cartesian_form'
        """
        crossing_simplices = self.identify_svm_crossing_simplices()
        weights, _ = self._get_weights_and_intercept()
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
        """
        Finds leaf simplices whose siblings are all on the same side of the decision boundary.
        
        These simplices can potentially be merged back into their parent since the
        subdivision doesn't contribute to the decision boundary.
        
        Returns:
            Set of frozenset vertex keys identifying mergeable simplices
        """
        weights, intercept = self._get_weights_and_intercept()
        same_side_keys = set()
        leaf_parents = set()
        for leaf in self.leaf_simplexes:
            if leaf.parent:
                leaf_parents.add(leaf.parent)
        
        for parent in leaf_parents:
            all_children_are_leaves = all(child._is_leaf() for child in parent.children)
            if not all_children_are_leaves:
                continue
            
            if self._are_siblings_same_side(parent, weights, intercept):
                for child in parent.children:
                    same_side_keys.add(frozenset(child.vertex_indices))
        
        return same_side_keys

    def find_nonconvex_simplexes(self, epsilon: float = 0.30) -> Set[frozenset]:
        """
        Finds leaf simplices where the decision boundary is non-convex.
        
        For each pair of adjacent crossing simplices, checks whether the
        boundary between them is convex. Requires a linear classifier.
        
        Args:
            epsilon: Fraction (0-1) of distance from meeting point to external
                     crossing used for the convexity test. Default 0.30.
            
        Returns:
            Set of frozenset vertex keys identifying non-convex simplices
        """
        weights, intercept = self._get_weights_and_intercept()
        crossing_simplices = self.identify_crossing_simplices()
        crossing_set = {id(info['simplex']) for info in crossing_simplices}
        
        nonconvex_keys = set()
        for info in crossing_simplices:
            simplex1 = info['simplex']
            for simplex2 in self.find_adjacent_simplexes(simplex1):
                if id(simplex2) not in crossing_set:
                    continue
                is_convex, *_ = check_convexity(
                    simplex1, simplex2, weights, intercept,
                    global_tree=self.tree, epsilon=epsilon
                )
                if not is_convex:
                    nonconvex_keys.add(frozenset(simplex1.vertex_indices))
                    nonconvex_keys.add(frozenset(simplex2.vertex_indices))
        
        return nonconvex_keys