import numpy as np
from typing import List, Tuple, Dict, Optional


class VertexRegistry:
    def __init__(self, tolerance: float = 1e-10):
        self.vertices: List[np.ndarray] = []
        self.vertex_to_index: Dict[Tuple, int] = {}
        self.tolerance = tolerance
    
    def register_vertex(self, vertex: Tuple[float, ...]) -> int:
        vertex_tuple = tuple(float(x) for x in vertex)
        if vertex_tuple in self.vertex_to_index:
            return self.vertex_to_index[vertex_tuple]
        idx = len(self.vertices)
        self.vertices.append(np.array(vertex_tuple))
        self.vertex_to_index[vertex_tuple] = idx
        return idx
    
    def register_vertices(self, vertices: List[Tuple[float, ...]]) -> List[int]:
        return [self.register_vertex(v) for v in vertices]
    
    def get_vertex(self, idx: int) -> np.ndarray:
        return self.vertices[idx]
    
    def get_vertices(self, indices: List[int]) -> List[np.ndarray]:
        return [self.vertices[idx] for idx in indices]
    
    def get_vertex_as_tuple(self, idx: int) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.vertices[idx])
    
    def get_vertices_as_tuples(self, indices: List[int]) -> List[Tuple[float, ...]]:
        return [self.get_vertex_as_tuple(idx) for idx in indices]
    
    def get_index(self, vertex: Tuple[float, ...]) -> Optional[int]:
        vertex_tuple = tuple(float(x) for x in vertex)
        return self.vertex_to_index.get(vertex_tuple)
    
    def __len__(self) -> int:
        return len(self.vertices)
    
    def __repr__(self) -> str:
        return f"VertexRegistry(num_vertices={len(self.vertices)})"

