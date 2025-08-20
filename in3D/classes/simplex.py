import numpy as np
from typing import List, Tuple, Optional


class Simplex:
    def __init__(self, vertices: List[Tuple[float, ...]], tolerance: float = 1e-10):  
        if len(vertices) < 2:
            raise ValueError("Simplex must have at least 2 vertices")
        
        self.vertices = [np.array(v) for v in vertices]
        self.n_vertices = len(vertices)
        self.dimension = len(vertices[0])
        self.tolerance = tolerance
       
        if not all(len(v) == self.dimension for v in vertices):
            raise ValueError("All vertices must have the same dimension")
        
        # Allow more vertices than dimension + 1 for higher-dimensional forms
        if self.n_vertices < self.dimension + 1:
            raise ValueError(f"For {self.dimension}D space, need at least {self.dimension + 1} vertices, got {self.n_vertices}")
        
        self._build_transformation_matrix()
    
    def _build_transformation_matrix(self):
        v0 = self.vertices[0]
        self.A = np.column_stack([v - v0 for v in self.vertices[1:]])
        
        # Handle different cases based on matrix shape
        if self.n_vertices == self.dimension + 1:
            # Standard case: square matrix
            self.det_A = np.linalg.det(self.A)
            self.is_degenerate = abs(self.det_A) < self.tolerance
            if not self.is_degenerate:
                self.A_inv = np.linalg.inv(self.A)
            else:
                self.A_inv = None
        else:
            # Overdetermined case: more vertices than dimension + 1
            # Use pseudo-inverse for least squares solution
            self.det_A = 0  # Not applicable for non-square matrices
            try:
                self.A_inv = np.linalg.pinv(self.A)  # Moore-Penrose pseudo-inverse
                self.is_degenerate = False
            except:
                self.A_inv = None
                self.is_degenerate = True
    
    def is_linearly_independent(self) -> bool:
        return not self.is_degenerate
    
    def get_determinant(self) -> float:
        return float(self.det_A)
    
    def can_perform_test(self) -> bool:
        return not self.is_degenerate
    
    @staticmethod
    def convert_2d_to_homogeneous(point_2d: Tuple[float, float]) -> Tuple[float, float, float]:
        return (point_2d[0], point_2d[1], 1.0)
    
    def embed_point(self, point: Tuple[float, ...]) -> Optional[Tuple[float, ...]]:
        if self.is_degenerate:
            return None
        P = np.array(point)
        if len(P) != self.dimension:
            raise ValueError(f"Point dimension {len(P)} doesn't match simplex dimension {self.dimension}")
        v0 = self.vertices[0]
        b = P - v0
        
        if self.n_vertices == self.dimension + 1:
            # Standard case: exact solution
            alpha_rest = self.A_inv @ b
            alpha_0 = 1 - np.sum(alpha_rest)
            return tuple([float(alpha_0)] + [float(x) for x in alpha_rest])
        else:
            # Overdetermined case: least squares solution
            # Minimize ||A * alpha_rest - b||^2 subject to sum(alpha) = 1
            alpha_rest = self.A_inv @ b
            # Normalize so all alphas sum to 1
            all_alphas = np.concatenate([[1 - np.sum(alpha_rest)], alpha_rest])
            alpha_sum = np.sum(all_alphas)
            if abs(alpha_sum) > self.tolerance:
                all_alphas = all_alphas / alpha_sum
            return tuple(float(x) for x in all_alphas)
    
    def point_inside_simplex(self, point: Tuple[float, ...]) -> bool:
        coords = self.embed_point(point)
        if coords is None:
            return False
        
        # For higher-dimensional simplexes, we check if the point is in the convex hull
        # All barycentric coordinates should be non-negative and sum to 1
        if not all(alpha >= -self.tolerance for alpha in coords):
            return False
        if abs(sum(coords) - 1.0) > self.tolerance:
            return False
        
        # For standard simplexes, also check upper bound
        if self.n_vertices == self.dimension + 1:
            if not all(alpha <= 1 + self.tolerance for alpha in coords):
                return False
        
        return True
    
    def __repr__(self):
        return f"Simplex(vertices={len(self.vertices)}, dim={self.dimension}, degenerate={self.is_degenerate})"


if __name__ == "__main__":
    simplex = Simplex([(0, 0), (1, 0), (0, 1)])
    print(simplex.embed_point((0.0, 2))) 

