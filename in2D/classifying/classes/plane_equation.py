import numpy as np
from typing import List, Optional, Iterator, Tuple, Union
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
in2d_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, in2d_dir)

from embedding.classes.simplex import Simplex
from embedding.classes.simplex_tree import SimplexTree

class PlaneEquation:
    def __init__(self, simplex: Simplex):
        self.simplex = simplex
        self.plane_coefficients = None
        self.normalized_coefficients = None
        
    def compute_plane_from_weights(self, weight_vector: np.ndarray) -> np.ndarray:
        weight_vector = np.array(weight_vector).flatten()
        
        if len(weight_vector) != len(self.simplex.vertices):
            raise ValueError(f"Weight vector size {len(weight_vector)} doesn't match number of vertices {len(self.simplex.vertices)}")
        
        vertices = self.simplex.vertices
        x_coords = np.array([v[0] for v in vertices])
        y_coords = np.array([v[1] for v in vertices])
        
        A_matrix = np.column_stack([
            x_coords,
            y_coords,
            np.ones(len(vertices))
        ])
        
        try:
            if len(vertices) == 3:
                plane_coefficients = np.linalg.solve(A_matrix, weight_vector)
            else:
                plane_coefficients = np.linalg.lstsq(A_matrix, weight_vector, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise ValueError("Cannot compute plane equation - matrix is singular")
        
        self.plane_coefficients = plane_coefficients
        
        magnitude = np.linalg.norm(self.plane_coefficients)
        self.normalized_coefficients = self.plane_coefficients / magnitude if magnitude > 0 else self.plane_coefficients
        
        return self.plane_coefficients
    
    def get_cartesian_form(self) -> str:
        if self.plane_coefficients is None:
            raise ValueError("Plane equation not computed yet. Call compute_plane_from_weights first.")
        
        coefficient_a, coefficient_b, coefficient_c = self.plane_coefficients
        
        equation_parts = []
        
        if abs(coefficient_a) > 1e-10:
            if coefficient_a == 1.0:
                equation_parts.append("x")
            elif coefficient_a == -1.0:
                equation_parts.append("-x")
            else:
                equation_parts.append(f"{coefficient_a:.4f}x")
        
        if abs(coefficient_b) > 1e-10:
            if coefficient_b > 0:
                if equation_parts:
                    equation_parts.append("+")
                if coefficient_b == 1.0:
                    equation_parts.append("z")
                else:
                    equation_parts.append(f"{coefficient_b:.4f}z")
            else:
                if coefficient_b == -1.0:
                    equation_parts.append("- z" if equation_parts else "-z")
                else:
                    equation_parts.append(f"- {abs(coefficient_b):.4f}z" if equation_parts else f"{coefficient_b:.4f}z")
        
        if abs(coefficient_c) > 1e-10:
            if coefficient_c > 0:
                if equation_parts:
                    equation_parts.append("+")
                equation_parts.append(f"{coefficient_c:.4f}")
            else:
                equation_parts.append(f"- {abs(coefficient_c):.4f}" if equation_parts else f"{coefficient_c:.4f}")
        
        if not equation_parts:
            return "y = 0"
        
        return "y = " + " ".join(equation_parts)
    
    # def evaluate_at_point(self, x: float, y: float) -> float:
    #     if self.plane_coefficients is None:
    #         raise ValueError("Plane equation not computed yet. Call compute_plane_from_weights first.")
        
    #     coefficient_a, coefficient_b, coefficient_c = self.plane_coefficients
    #     return coefficient_a * x + coefficient_b * y + coefficient_c
    
    # def get_normal_vector(self) -> np.ndarray:
    #     if self.plane_coefficients is None:
    #         raise ValueError("Plane equation not computed yet. Call compute_plane_from_weights first.")
        
    #     coefficient_a, coefficient_b, coefficient_c = self.plane_coefficients
    #     normal = np.array([coefficient_a, coefficient_b, -1])
    #     return normal / np.linalg.norm(normal)
    

if __name__ == "__main__":
    vertices_2d = [(0, 0), (1, 0), (0.5, 1)]
    simplex = Simplex(vertices_2d)
    plane_eq = PlaneEquation(simplex)
    test_weights = np.array([0.3, 0.4, 0.3])
    
    plane_vector = plane_eq.compute_plane_from_weights(test_weights)
    print(f"Plane coefficient vector [a, b, c]: {plane_vector}")
    
    cartesian_form = plane_eq.get_cartesian_form()
    print(f"Cartesian form: {cartesian_form}")
    
    
    
    # print("\nVerification at vertices:")
    # for i, vertex in enumerate(vertices_2d):
    #     x_val, y_val = vertex
    #     w_value = plane_eq.evaluate_at_point(x_val, y_val)
    #     print(f"At vertex V{i}=({x_val}, {y_val}): w={w_value:.3f}, expected={test_weights[i]}")
    
    # normal = plane_eq.get_normal_vector()
    # print(f"\nNormal vector to plane: {normal}")
    
    
    tree = SimplexTree(vertices_2d)
    tree.add_barycentric_centers_recursively(1)
    
    leaves = tree.get_leaves()

    for i, leaf in enumerate(leaves):
        vertices = leaf.get_vertices_as_tuples()
        print(f"\nLeaf {i} vertices: {vertices}")
        
        plane_eq_leaf = PlaneEquation(leaf)
        test_weights_leaf = np.array([0.25, 0.25, 0.5])
        plane_eq_leaf.compute_plane_from_weights(test_weights_leaf)
        print(f"Plane equation: {plane_eq_leaf.get_cartesian_form()}")
 