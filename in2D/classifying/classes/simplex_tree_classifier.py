import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import lil_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
from collections import defaultdict


current_dir = os.path.dirname(os.path.abspath(__file__))
in2d_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, in2d_dir)

from embedding.classes.simplex_tree import SimplexTree
from embedding.utilss.visualization import visualize_simplex_tree
from embedding.utilss.visualization import visualize_subdivision_levels
from in2D.classifying.classes.utilss.plane_equation import PlaneEquation

class SimplexTreeClassifier:
    def __init__(self, vertices: List[Tuple[float, float]] = None, 
                 regularization=0.01, 
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
    
    def transform(self, data_points) -> lil_matrix:
        simplex_tree = self.tree
        all_vertices = []
        vertex_to_index = {}
        
        for node in simplex_tree.traverse_breadth_first():
            for vertex in node.vertices:
                vertex_tuple = tuple(vertex)
                if vertex_tuple not in vertex_to_index:
                    vertex_to_index[vertex_tuple] = len(all_vertices)
                    all_vertices.append(vertex_tuple)
                
        max_vertices = len(all_vertices)
        barycentric_matrix = lil_matrix((len(data_points), max_vertices))
        
        for point_index in range(len(data_points)): 
            point = tuple(data_points[point_index])
            containing_simplex = simplex_tree.find_containing_simplex(point)

            if containing_simplex is not None:
                data_point_embeddings = containing_simplex.embed_point(point)
                
                simplex_vertices = containing_simplex.get_vertices_as_tuples()
                
                # print(f"point: {point}")
                # formatted_vertices = [f"({v[0]:.1f}, {v[1]:.1f})" for v in simplex_vertices]
                # print(f"containing_simplex: {formatted_vertices}")   
                             
                
                for local_idx, coordinate in enumerate(data_point_embeddings):
                    vertex = simplex_vertices[local_idx]
                    global_idx = vertex_to_index[vertex]
                    barycentric_matrix[point_index, global_idx] = coordinate
        return barycentric_matrix

    def get_vertex_mapping(self) -> Dict[int, Tuple[float, float]]:
        all_vertices = []
        vertex_to_index = {}
        
        for node in self.tree.traverse_breadth_first():
            for vertex in node.vertices:
                vertex_tuple = tuple(vertex)
                if vertex_tuple not in vertex_to_index:
                    vertex_to_index[vertex_tuple] = len(all_vertices)
                    all_vertices.append(vertex_tuple)

        vertex_mapping = {}
        for idx, vertex in enumerate(all_vertices):
            vertex_mapping[idx] = vertex

        return vertex_mapping

    def get_column_to_vertex_mapping(self) -> str:
        mapping = self.get_vertex_mapping()
        mapping_lines = ["Column-to-vertex mapping (in insertion order):"]
        for idx, vertex in mapping.items():
            mapping_lines.append(f"  Column {idx} -> Vertex {vertex}")
        return "\n".join(mapping_lines)

    # def get_vertex_for_column(self, column_index: int) -> Tuple[float, float]:
    #     mapping = self.get_vertex_mapping()
    #     if column_index not in mapping:
    #         raise ValueError(f"Column index {column_index} is out of range. Valid indices: 0-{len(mapping)-1}")
    #     return mapping[column_index]

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-10)
        
        return normalized
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_normalized = self.normalize_data(X)
        
        X_transformed = self.transform(X_normalized)
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
        X_transformed = self.transform(X_normalized)
        predictions = self.classifier.predict(X_transformed)
        return predictions

    def visualize_with_data_points(self, data_points: np.ndarray, title: str = "Simplex Tree with Data Points", figsize: Tuple[int, int] = (8, 5)):
        data_points_list = [tuple(point) for point in data_points]
        visualize_simplex_tree(self.tree, data_points=data_points_list, title=title, figsize=figsize)

    # def print_simplex_membership(self, data_points: List[Tuple[float, float]]) -> None:
    #     for idx, pt in enumerate(data_points):
    #         simplex = self.tree.find_containing_simplex(pt)
    #         if simplex is None:
    #             print(f"Point {pt} is outside the simplex tree")
    #         else:
    #             verts = ", ".join([f"({v[0]:.1f}, {v[1]:.1f})" for v in simplex.get_vertices_as_tuples()])
    #             print(f"Point {idx} {pt} is in simplex: [{verts}]")

    def get_simplex_boundaries(self) -> List[List[Tuple[float, float]]]:
        boundaries = []
        for leaf in self.leaf_simplexes:
            vertices = leaf.get_vertices_as_tuples()
            if len(vertices) >= 3:
                boundaries.append(vertices)
        return boundaries
    
    def identify_svm_crossing_simplices(self) -> List[Dict]:
        if self.classifier is None:
            raise ValueError("Classifier not fitted yet. Call fit() first.")
        
        crossing_simplices = []
        
        for leaf in self.leaf_simplexes:
            vertices = leaf.get_vertices_as_tuples()
            vertex_matrix = self.transform(np.array(vertices))  # Get decision function values at vertices
            decision_values = self.classifier.decision_function(vertex_matrix)
            has_positive = any(val > 0 for val in decision_values)
            has_negative = any(val < 0 for val in decision_values)
            
            if has_positive and has_negative:
                crossing_simplices.append({
                    'simplex': leaf,
                    'vertices': vertices,
                    'decision_values': decision_values,
                })
        
        return crossing_simplices
    
    def compute_svm_plane_equations(self) -> List[Dict]:
        crossing_simplices = self.identify_svm_crossing_simplices()
        plane_equations = []
        
        for crossing_info in crossing_simplices:
            simplex = crossing_info['simplex']
            decision_values = crossing_info['decision_values']
            
            plane_eq = PlaneEquation(simplex)
            
            plane_coefficients = plane_eq.compute_plane_from_weights(decision_values)
            
            plane_equations.append({
                'simplex': simplex,
                'vertices': crossing_info['vertices'],
                'plane_equation': plane_eq,
                'coefficients': plane_coefficients,
                'cartesian_form': plane_eq.get_cartesian_form()
            })
        
        return plane_equations
    
    def check_polygon_convexity(self, vertices: List[Tuple[float, float]]) -> Dict:
        if len(vertices) < 3:
            return {'is_convex': True, 'reason': 'Less than 3 vertices'}
        
        n = len(vertices)
        sign = None
        
        for i in range(n):
            v1 = np.array(vertices[i])
            v2 = np.array(vertices[(i + 1) % n])
            v3 = np.array(vertices[(i + 2) % n])
            
            edge1 = v2 - v1
            edge2 = v3 - v2
            
            cross_product_z = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            if abs(cross_product_z) < 1e-10:
                continue
            
            current_sign = np.sign(cross_product_z)
            
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return {
                    'is_convex': False,
                    'reason': f'Sign change at vertex {i+1}',
                    'vertex_index': i + 1,
                    'cross_product': cross_product_z
                }
        
        return {'is_convex': True, 'reason': 'All cross products have same sign'}
    
    def find_redundant_planes(self, angle_threshold: float = 0.9, check_convexity: bool = True) -> List[Dict]:
        plane_equations = self.compute_svm_plane_equations()
        redundant_planes = []
        
        if check_convexity:
            for i, plane in enumerate(plane_equations):
                vertices = plane['vertices']
                convexity = self.check_polygon_convexity(vertices)
                
                if not convexity['is_convex']:
                    redundant_planes.append({
                        'plane_index': i,
                        'reason': 'non_convex',
                        'convexity_details': convexity,
                        'simplex': plane['simplex'],
                        'vertices': vertices,
                        'coefficients': plane['coefficients'],
                        'equation': plane['cartesian_form']
                    })
        
        for i in range(len(plane_equations)):
            if any(r['plane_index'] == i for r in redundant_planes):
                continue
                
            for j in range(i + 1, len(plane_equations)):
                if any(r['plane_index'] == j for r in redundant_planes):
                    continue
                    
                plane1 = plane_equations[i]
                plane2 = plane_equations[j]
                
                coeffs1 = np.array(plane1['coefficients'])
                coeffs2 = np.array(plane2['coefficients'])  
                
                norm1 = coeffs1[:2] / (np.linalg.norm(coeffs1[:2]) + 1e-10)
                norm2 = coeffs2[:2] / (np.linalg.norm(coeffs2[:2]) + 1e-10)
                
                cosine_similarity = abs(np.dot(norm1, norm2))
                
                if cosine_similarity >= angle_threshold:
                    redundant_planes.append({
                        'plane_index': j,
                        'reason': 'parallel_to',
                        'parallel_to_index': i,
                        'simplex': plane2['simplex'],
                        'vertices': plane2['vertices'],
                        'coefficients': coeffs2,
                        'equation': plane2['cartesian_form'],
                        'cosine_similarity': cosine_similarity,
                        'angle_degrees': np.degrees(np.arccos(min(cosine_similarity, 1.0)))
                    })
        
        return redundant_planes
    
    def analyze_plane_redundancy(self) -> Dict:
        plane_equations = self.compute_svm_plane_equations()
        redundant_info = self.find_redundant_planes()
        
        non_convex_planes = [r for r in redundant_info if r['reason'] == 'non_convex']
        parallel_planes = [r for r in redundant_info if r['reason'] == 'parallel_to']
        
        redundant_indices = set(r['plane_index'] for r in redundant_info)
        valid_indices = set(range(len(plane_equations))) - redundant_indices
        
        analysis = {
            'total_planes': len(plane_equations),
            'redundant_count': len(redundant_indices),
            'valid_count': len(valid_indices),
            'redundant_indices': sorted(redundant_indices),
            'valid_indices': sorted(valid_indices),
            'non_convex_count': len(non_convex_planes),
            'parallel_count': len(parallel_planes),
            'non_convex_planes': non_convex_planes,
            'parallel_planes': parallel_planes,
            'summary': {
                'total': len(plane_equations),
                'keep': len(valid_indices),
                'remove': len(redundant_indices),
                'remove_non_convex': len(non_convex_planes),
                'remove_parallel': len(parallel_planes)
            }
        }
        
        return analysis
    
    def find_redundancy_groups(self, angle_threshold: float = 0.95, min_area: float = 0.001) -> Dict:

        plane_equations = self.compute_svm_plane_equations()
        
        redundant_indices = set()
        parallel_groups = {}
        degenerate_simplices = set()
        
        # Check for degenerate (very small) simplices
        for i, plane in enumerate(plane_equations):
            vertices = plane['vertices']
            # Calculate triangle area using cross product
            if len(vertices) == 3:
                v0, v1, v2 = [np.array(v) for v in vertices]
                area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
                if area < min_area:
                    degenerate_simplices.add(i)
                    redundant_indices.add(i)
        
        for i in range(len(plane_equations)):
            if i in degenerate_simplices:
                continue
                
            for j in range(i + 1, len(plane_equations)):
                if j in degenerate_simplices:
                    continue
                    
                plane1 = plane_equations[i]
                plane2 = plane_equations[j]
                
                coeffs1 = np.array(plane1['coefficients'])
                coeffs2 = np.array(plane2['coefficients'])
                
                # Normalize the normal vectors (first two components)
                norm1 = coeffs1[:2] / (np.linalg.norm(coeffs1[:2]) + 1e-10)
                norm2 = coeffs2[:2] / (np.linalg.norm(coeffs2[:2]) + 1e-10)
                
                cosine_similarity = abs(np.dot(norm1, norm2))
                
                if cosine_similarity >= angle_threshold:
                    # Keep the first one, mark the second as redundant
                    if i not in parallel_groups:
                        parallel_groups[i] = []
                    parallel_groups[i].append(j)
                    redundant_indices.add(j)
        
        all_plane_indices = set(range(len(plane_equations)))
        planes_to_keep = all_plane_indices - redundant_indices
        
        return {
            'plane_equations': plane_equations,
            'planes_to_keep': sorted(planes_to_keep),
            'planes_to_remove': sorted(redundant_indices),
            'degenerate_simplices': sorted(degenerate_simplices),
            'parallel_groups': parallel_groups,
            'total_planes': len(plane_equations),
            'kept_planes': len(planes_to_keep),
            'removed_planes': len(redundant_indices),
            'angle_threshold': angle_threshold,
            'min_area': min_area
        }
    
    def visualize_redundant_planes(self, check_convexity: bool = True) -> None:
        redundancy_result = self.find_redundancy_groups(check_convexity)
        plane_equations = redundancy_result['plane_equations']
        
        if not plane_equations:
            print("No crossing planes to visualize")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract results
        planes_to_keep = set(redundancy_result['planes_to_keep'])
        planes_to_remove = set(redundancy_result['planes_to_remove'])
        non_convex_removed = set(redundancy_result['non_convex_indices'])
        
        ax1.set_title('All Planes (Green=Valid/Convex, Red=Non-Convex)')
        ax2.set_title('After Non-Convex Removal (Valid Planes Only)')
        
        # Draw simplex boundaries
        for ax in [ax1, ax2]:
            boundaries = self.get_simplex_boundaries()
            for boundary in boundaries:
                if len(boundary) >= 3:
                    closed = boundary + [boundary[0]]
                    xs, ys = zip(*closed)
                    ax.plot(xs, ys, 'gray', linewidth=0.5, alpha=0.3)
        
        for i, plane in enumerate(plane_equations):
            coeffs = plane['coefficients']
            a, b, c = coeffs
            
            if i in planes_to_keep:
                color = 'lime'
                alpha = 0.8
                linewidth = 2.5
                label = f'Valid {i}' if i < 3 else ''
            else:  # Non-convex
                color = 'red'
                alpha = 0.5
                linewidth = 1.5
                label = f'Non-convex {i}' if i < 3 else ''
            
            # Calculate line endpoints
            if abs(b) > abs(a):
                x_line = np.array([0, 1])
                y_line = -(a * x_line + c) / b
            else:
                y_line = np.array([0, 1])
                x_line = -(b * y_line + c) / a
            
            # Draw on first subplot (all planes)
            ax1.plot(x_line, y_line, color=color, linewidth=linewidth, 
                    alpha=alpha, label=label)
            
            # Mark non-convex simplex vertices with red dots
            if i in non_convex_removed:
                vertices = plane['vertices']
                for vertex in vertices:
                    ax1.plot(vertex[0], vertex[1], 'ro', markersize=5)
            
            # Draw on second subplot (only valid planes)
            if i in planes_to_keep:
                ax2.plot(x_line, y_line, color='lime', linewidth=2.5, alpha=0.8)
        
        # Set axis properties
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text1 = (f'Total: {redundancy_result["total_planes"]}\n'
                         f'Valid (Convex): {redundancy_result["kept_planes"]}\n'
                         f'Non-convex: {len(non_convex_removed)}')
        ax1.text(0.02, 0.98, summary_text1, 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        summary_text2 = (f'Final: {redundancy_result["kept_planes"]} planes\n'
                        f'({redundancy_result["removed_planes"]} non-convex removed)')
        ax2.text(0.02, 0.98, summary_text2, 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend if not too many planes
        if len(plane_equations) < 10:
            ax1.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'Non-Convex Simplex Detection and Removal', fontsize=14)
        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":


#     classifier = SimplexTreeClassifier(vertices=[(0,0), (1,0), (0.5,0.5)], subdivision_levels=1)
#     classifier.tree.add_splitting_point((0.62, 0.3))

#     # classifier.tree.add_splitting_point((0.61, 0.28))

#     # classifier.tree.add_splitting_point((0.5, 0.4))

#     data_points = ((0.43,0.2), (0.1,0.4))
#     vertex_mapping = classifier.get_vertex_mapping()
#     transformed_matrix = classifier.transform(data_points)
    
    
#     ## PRINTS: 
#     print("Column-to-vertex mapping:")
#     for col_idx, vertex in vertex_mapping.items():
#         print(f"  Column {col_idx} -> Vertex ({vertex[0]:.1f}, {vertex[1]:.1f})")
        
#     print()
#     print("Barycentric matrix shape:", transformed_matrix.shape)
#     print("Matrix content:")
#     matrix_array = transformed_matrix.toarray()
#     for i, row in enumerate(matrix_array):
#         formatted_row = [f"{val:.2f}" for val in row]
#         print(f"[{' '.join(formatted_row)}]")
            
#     print("\n" + "="*50)
#     print("TREE STRUCTURE:")
#     print("="*50)
#     classifier.tree.print_tree()

#     print("\n" + "="*50)
#     print("VISUALIZATION:")
#     print("="*50) 
#     classifier.visualize_with_data_points(data_points)
    
#     base_vertices = [(0,0), (1,0), (0,1)]
#     data_points = [(0.5,0.2), (0.1,0.4), (0.2,0.7)] 

#####################################################################################
    # visualize_subdivision_levels(base_vertices, max_level=3, data_points=data_points)


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


