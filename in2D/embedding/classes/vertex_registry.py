import numpy as np
from typing import List, Tuple, Dict, Optional


class VertexRegistry:
    def __init__(self, tolerance: float = 1e-10):
        self.vertices: List[np.ndarray] = []
        self.vertex_to_index: Dict[Tuple, int] = {} 
        self.tolerance = tolerance
    
    def _register_vertex(self, vertex: Tuple[float, ...]) -> int:
        vertex_tuple = tuple(float(x) for x in vertex)
        if vertex_tuple in self.vertex_to_index:
            return self.vertex_to_index[vertex_tuple]
        idx = len(self.vertices)
        self.vertices.append(np.array(vertex_tuple))
        self.vertex_to_index[vertex_tuple] = idx
        return idx
    
    def register_vertices(self, vertices: List[Tuple[float, ...]]) -> List[int]:
        """
        Registers multiple vertices and returns their indices.
        
        Existing vertices return their existing index (no duplicates created).
        
        Args:
            vertices: List of coordinate tuples to register
            
        Returns:
            List of integer indices corresponding to each vertex
        """
        return [self._register_vertex(v) for v in vertices]
    
    def _get_vertex(self, idx: int) -> np.ndarray:
        return self.vertices[idx]
    
    def _get_vertices(self, indices: List[int]) -> List[np.ndarray]:
        return [self.vertices[idx] for idx in indices]
    
    def _get_vertex_as_tuple(self, idx: int) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.vertices[idx])
    
    def get_vertices_as_tuples(self, indices: List[int]) -> List[Tuple[float, ...]]:
        """
        Returns coordinates as tuples for the given vertex indices.
        
        Args:
            indices: List of vertex indices to look up
            
        Returns:
            List of coordinate tuples
        """
        return [self._get_vertex_as_tuple(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.vertices)
    
    def __repr__(self) -> str:
        return f"VertexRegistry(num_vertices={len(self.vertices)})"

