import numpy as np
from typing import List, Tuple, Optional


class Simplex:
    def __init__(self, vertices: List[Tuple[float, float]], tolerance: float = 1e-10):  
        if len(vertices) < 2:
            raise ValueError("Simplex must have at least 2 vertices")
        
        self.vertices = [np.array(v) for v in vertices]
        self.n_vertices = len(vertices)
        self.dimension = len(vertices[0])
        self.tolerance = tolerance
       
        if not all(len(v) == self.dimension for v in vertices):
            raise ValueError("All vertices must have the same dimension")
        
        if self.n_vertices < self.dimension + 1:
            raise ValueError(f"For {self.dimension}D space, need at least {self.dimension + 1} vertices, got {self.n_vertices}")
        
        self._build_transformation_matrix()
    
    def _build_transformation_matrix(self):
        v0 = self.vertices[0]
        
        if self.n_vertices == self.dimension + 1:
            self.A = np.column_stack([v - v0 for v in self.vertices[1:]])
            self.det_A = np.linalg.det(self.A)
            self.is_degenerate = abs(self.det_A) < self.tolerance
            if not self.is_degenerate:
                self.A_inv = np.linalg.inv(self.A)
            else:
                self.A_inv = None
        else:
            edge_vectors = np.array([v - v0 for v in self.vertices[1:]])
            self.A = edge_vectors.T
            
            try:
                self.A_pseudo_inv = np.linalg.pinv(self.A)
                self.is_degenerate = False
                self.det_A = 1.0  
            except np.linalg.LinAlgError:
                self.is_degenerate = True
                self.A_pseudo_inv = None
                self.det_A = 0.0
    
    
    def embed_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, ...]]:
        if self.is_degenerate:
            return None
        P = np.array(point)
        if len(P) != self.dimension:
            raise ValueError(f"Point dimension {len(P)} doesn't match simplex dimension {self.dimension}")
        
        if self.n_vertices == self.dimension + 1:
            v0 = self.vertices[0]
            b = P - v0
            alpha_rest = self.A_inv @ b
            alpha_0 = 1 - np.sum(alpha_rest)
            embaddings = tuple([float(alpha_0)] + [float(x) for x in alpha_rest])
        else:
            # Find minimal containing sub-simplex and return sparse coordinates
            from itertools import combinations
            required_vertices = self.dimension + 1 
            
            # Try all combinations of required_vertices from available vertices
            best_simplex = None
            best_coords = None
            best_indices = None
            
            for vertex_indices in combinations(range(self.n_vertices), required_vertices):
                sub_vertices = [self.vertices[i] for i in vertex_indices]
                
                try:
                    # Create temporary simplex with these vertices
                    temp_simplex = Simplex([tuple(v) for v in sub_vertices], self.tolerance)
                    
                    # Check if point is inside this sub-simplex
                    if temp_simplex.point_inside_simplex(tuple(P)):
                        coords = temp_simplex.embed_point(tuple(P))
                        if coords is not None and all(c >= -self.tolerance for c in coords):
                            # This is a valid containing simplex
                            # Prefer the one with coordinates closer to interior (more balanced)
                            min_coord = min(coords)
                            if best_simplex is None or min_coord > best_simplex:
                                best_simplex = min_coord
                                best_coords = coords
                                best_indices = vertex_indices
                except (ValueError, np.linalg.LinAlgError):
                    # Skip degenerate combinations
                    continue
            
            if best_coords is not None:
                # Create sparse coordinates - zeros for all vertices, non-zero only for minimal simplex
                sparse_coords = [0.0] * self.n_vertices
                for i, vertex_idx in enumerate(best_indices):
                    sparse_coords[vertex_idx] = best_coords[i]
                embaddings = tuple(sparse_coords)
            else:
                # If no minimal simplex found, fallback to least squares
                vertex_matrix = np.array(self.vertices).T
                constraint_matrix = np.vstack([vertex_matrix, np.ones(self.n_vertices)])
                constraint_vector = np.append(P, 1.0)
                
                try:
                    weights = np.linalg.lstsq(constraint_matrix, constraint_vector, rcond=None)[0]
                    embaddings = tuple(float(w) for w in weights)
                except np.linalg.LinAlgError:
                    return None
        return embaddings
    

    def point_inside_simplex(self, point: Tuple[float, float]) -> bool:
        coords = self.embed_point(point)
        if coords is None:
            return False
        
        if not all(0 <= alpha <= 1 for alpha in coords):
            return False
        
        if abs(sum(coords) - 1.0) > self.tolerance:
            return False
    
        return True
    
    def __repr__(self):
        return f"Simplex(vertices={len(self.vertices)}, dim={self.dimension}, degenerate={self.is_degenerate})"


if __name__ == "__main__":
    simplex = Simplex([(0, 0), (1, 0), (0, 1)])
    print(simplex.embed_point((0.5, 0.5))) 