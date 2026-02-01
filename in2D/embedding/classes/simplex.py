import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vertex_registry import VertexRegistry

class Simplex:
    def __init__(self, vertex_indices: List[int], registry: 'VertexRegistry', tolerance: float = 1e-10):
        if len(vertex_indices) < 2:
            raise ValueError("Simplex must have at least 2 vertices")
        
        self.vertex_indices = vertex_indices
        self.registry = registry
        self.n_vertices = len(vertex_indices)
        self.tolerance = tolerance
        
        first_vertex = self.registry._get_vertex(vertex_indices[0])
        self.dimension = len(first_vertex)
        
        for idx in vertex_indices:
            v = self.registry._get_vertex(idx)
            if len(v) != self.dimension:
                raise ValueError("All vertices must have the same dimension")
        
        if self.n_vertices < self.dimension + 1:
            raise ValueError(f"For {self.dimension}D space, need at least {self.dimension + 1} vertices, got {self.n_vertices}")
        
        self._build_transformation_matrix()
    
    @property
    def vertices(self) -> List[np.ndarray]:
        """Returns list of vertex coordinates as numpy arrays."""
        return self.registry._get_vertices(self.vertex_indices)
    
    def _build_transformation_matrix(self):
        vertices = self.vertices
        v0 = vertices[0]
        
        if self.n_vertices == self.dimension + 1:
            self.A = np.column_stack([v - v0 for v in vertices[1:]])
            self.det_A = np.linalg.det(self.A)
            self.is_degenerate = abs(self.det_A) < self.tolerance
            if not self.is_degenerate:
                self.A_inv = np.linalg.inv(self.A)
            else:
                self.A_inv = None
        else:
            edge_vectors = np.array([v - v0 for v in vertices[1:]])
            self.A = edge_vectors.T
            
            try:
                self.A_pseudo_inv = np.linalg.pinv(self.A)
                self.is_degenerate = False
                self.det_A = 1.0  ## TODO: now its useless i can check for if its almost 0 or smaller then a certain threshold i should throw it away bc it means the volume is almost 0
            except np.linalg.LinAlgError:
                self.is_degenerate = True
                self.A_pseudo_inv = None
                self.det_A = 0.0
    
    def _embed_point(self, point: Tuple[float, ...]) -> Optional[Tuple[float, ...]]: 
        if self.is_degenerate:
            return None
        P = np.array(point)
        if len(P) != self.dimension:
            raise ValueError(f"Point dimension {len(P)} doesn't match simplex dimension {self.dimension}")
        
        vertices = self.vertices
        
        if self.n_vertices == self.dimension + 1:
            v0 = vertices[0]
            b = P - v0
            alpha_rest = self.A_inv @ b
            alpha_0 = 1 - np.sum(alpha_rest)
            embeddings = tuple([float(alpha_0)] + [float(x) for x in alpha_rest])
        else:
            from itertools import combinations
            required_vertices = self.dimension + 1 
            
            best_simplex = None
            best_coords = None
            best_indices = None
            
            for local_indices in combinations(range(self.n_vertices), required_vertices):
                sub_vertex_indices = [self.vertex_indices[i] for i in local_indices]
                
                try:
                    temp_simplex = Simplex(sub_vertex_indices, self.registry, self.tolerance)
                    
                    if temp_simplex._point_inside_simplex(tuple(P)):
                        coords = temp_simplex._embed_point(tuple(P))
                        if coords is not None and all(c >= -self.tolerance for c in coords):
                            min_coord = min(coords)
                            if best_simplex is None or min_coord > best_simplex:
                                best_simplex = min_coord
                                best_coords = coords
                                best_indices = local_indices
                except (ValueError, np.linalg.LinAlgError):
                    continue
            
            if best_coords is not None:
                sparse_coords = [0.0] * self.n_vertices
                for i, local_idx in enumerate(best_indices):
                    sparse_coords[local_idx] = best_coords[i]
                embeddings = tuple(sparse_coords)
            else:
                vertex_matrix = np.array(vertices).T
                constraint_matrix = np.vstack([vertex_matrix, np.ones(self.n_vertices)])
                constraint_vector = np.append(P, 1.0)
                
                try:
                    weights = np.linalg.lstsq(constraint_matrix, constraint_vector, rcond=None)[0]
                    embeddings = tuple(float(w) for w in weights)
                except np.linalg.LinAlgError:
                    return None
        return embeddings

    def _point_inside_simplex(self, point): ## TODO: to change the name to is__point_inside_simplex()
        coords = self._embed_point(point)
        if coords is None:
            return False
        
        if not all(0 <= alpha <= 1 for alpha in coords):
            return False
        
        if abs(sum(coords) - 1.0) > self.tolerance:
            return False
    
        return True
    
    def get_vertices_as_tuples(self) -> List[Tuple[float, ...]]:
        """Returns list of vertex coordinates as tuples for easy iteration/display."""
        return self.registry.get_vertices_as_tuples(self.vertex_indices)
    
    def __repr__(self):
        return f"Simplex(vertex_indices={self.vertex_indices}, dim={self.dimension}, degenerate={self.is_degenerate})"
