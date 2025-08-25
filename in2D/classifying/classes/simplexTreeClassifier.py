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
            vertices = [(0, 0), (1, 0), (0.5, 1)]   # default to triangle if no vertices provided
        
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
    
    def transform(self, simplex_tree: SimplexTree, data_points) -> lil_matrix:
        all_vertices = set()
        for node in simplex_tree.traverse_depth_first():
            for vertex in node.vertices:
                all_vertices.add(tuple(vertex))
                
        max_vertices = len(all_vertices)
        all_vertices_list = sorted(list(all_vertices))
        barycentric_matrix = lil_matrix((len(data_points), max_vertices))
        
        for point_index in range(len(data_points)): 
            point = tuple(data_points[point_index])
            containing_simplex = simplex_tree.find_containing_simplex(point)

            if containing_simplex is not None:
                data_point_embaddings = containing_simplex.embed_point(point)
                
                simplex_vertices = containing_simplex.get_vertices_as_tuples()
                
                print(f"point: {point}")
                formatted_vertices = [f"({v[0]:.2f}, {v[1]:.2f})" for v in simplex_vertices]
                print(f"containing_simplex: {formatted_vertices}")   
                             
                
                for local_idx, coordinate in enumerate(data_point_embaddings):
                    vertex = simplex_vertices[local_idx]
                    global_idx = all_vertices_list.index(vertex)
                    barycentric_matrix[point_index, global_idx] = coordinate
        
        print()            
        print("Barycentric matrix shape:", barycentric_matrix.shape)
        print("Matrix content:")
        matrix_array = barycentric_matrix.toarray()
        for i, row in enumerate(matrix_array):
            formatted_row = [f"{val:.1f}" for val in row]
            print(f"[{' '.join(formatted_row)}]")

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
        
        X_transformed = self.transform(self.tree, X_normalized)
        print(X_transformed.shape)
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
        X_transformed = self.transform(self.tree, X_normalized)
        predictions = self.classifier.predict(X_transformed)
        return predictions
    
    
if __name__ == "__main__":
    
    classifier = SimplexTreeClassifier(vertices=[(0,0), (1,0), (0,1)], subdivision_levels=1)
    
    print( "all tree vertices: ", classifier.tree.get_vertices_as_tuples()) ## Why doesnt it brings back (0.3, 0.3) as one of the tree's vertices
    print()
    
    # print("Classifier tree structure:")
    # print(f"Total nodes: {classifier.tree.get_node_count()}")
    # print(f"Leaf count: {len(classifier.tree.get_leaves())}")
    # print("Leaf simplexes:")
    # for i, leaf in enumerate(classifier.tree.get_leaves()):
    #     vertices = leaf.get_vertices_as_tuples()
    #     formatted_vertices = [f"({v[0]:.1f}, {v[1]:.1f})" for v in vertices]
    #     print(f"  Leaf {i}: {formatted_vertices}")
    


    data_points = ((0,0.3), (0.3,0), (0.2,0.2),(0.5, 0.5))
    
    classifier.transform(classifier.tree, data_points)
    visualize_simplex_tree(classifier.tree)    ## Make this project back to 2D

    
            
################################################ TO SHOW ################################################
    # print()
    # print("ORIGINAL SIMPLEX:", simplex_tree.get_vertices_as_tuples())
    # print()
    
    # data_point1 = (0.2, 0.3)
    # data_point2 = (0.4, 0.1)
    # data_point3 = (0.6, 0.3)
    
    # print("before adding centers")
    # embedding1 = simplex_tree.embed_data_point(data_point1)
    # print(f"embedding1 to point: {data_point1} are: ", embedding1)
    
    # simplex_tree.add_barycentric_centers_recursively(2)
    # print()
    # print("after adding centers")
    # embadding2 = simplex_tree.embed_data_point(data_point2)
    # print(f"embedding2 to point {data_point2} are: ", embadding2)
    
    # embadding3 = simplex_tree.embed_data_point(data_point3)
    # print(f"embedding3 to point {data_point2} are: ", embadding3)


    # if simplex_tree.children:
    #     for i, child in enumerate(simplex_tree.children):
    #         print(f"  Child {i} vertices: {child.get_vertices_as_tuples()}")
            
    # visualize_simplex_tree(simplex_tree)


####################################################################################################
    
    
    
    
################################################ TEST 2 ################################################
    # iris = fetch_ucirepo(id=53)
    # X_iris_full = iris.data.features.values
    # y_iris_full = iris.data.targets.values.ravel()
    
    # X_iris = X_iris_full[:, :2]  # Use first 2 features
    # y_iris_binary = (y_iris_full == 'Iris-setosa').astype(int)
    
    # X_iris_min, X_iris_max = X_iris.min(axis=0), X_iris.max(axis=0)
    # X_iris_normalized = (X_iris - X_iris_min) / (X_iris_max - X_iris_min)
    # X_iris_scaled = X_iris_normalized * 0.8 + 0.1
    
    # iris_valid_points = []
    # iris_valid_labels = []
    # for i, point in enumerate(X_iris_scaled):
    #     if (point[0] >= 0 and point[1] >= 0 and 
    #         point[0] <= 1 and point[1] <= 1 and
    #         point[1] <= 2 * (0.5 - abs(point[0] - 0.5))):
    #         iris_valid_points.append(point)
    #         iris_valid_labels.append(y_iris_binary[i])
    
    # X_iris_final = np.array(iris_valid_points)
    # y_iris_final = np.array(iris_valid_labels)
    
    # # Test both datasets
    # datasets = [
    #     (X_iris_final, y_iris_final, "Iris Dataset")
    # ]
    
    # subdivision_levels = [0, 1, 2, 3]
    
    # for X, y, dataset_name in datasets:
    #     for level in subdivision_levels:
    #         classifier = SimplexTreeClassifier(
    #             vertices=None,
    #             regularization=0.1,
    #             subdivision_levels=level,
    #             classifier_type='svc'
    #         )
            # classifier.fit(X, y)
            # title = f"{dataset_name} - Level {level}"
            # classifier.visualize_tree_and_classification(X, y)
            
####################################################################################################
        

