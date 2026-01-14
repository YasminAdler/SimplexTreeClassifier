import numpy as np
from typing import List, Optional, Iterator, Tuple
import sys
import os

from .simplex import Simplex
from .vertex_registry import VertexRegistry

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)


## TODO LIST: change where everever it should not be Tuple[float, float] but an array of D variables 
 
class SimplexTree(Simplex):
    def __init__(self, vertices: List[Tuple[float, float]], tolerance: float = 1e-10, 
                 registry: Optional[VertexRegistry] = None, _is_child: bool = False):
        if registry is None:
            reg = VertexRegistry(tolerance)
            self._is_root = True
            self._split_counter = [0]
        else:
            reg = registry
            self._is_root = not _is_child
            self._split_counter = None
        
        vertex_indices = reg.register_vertices(vertices)
        super().__init__(vertex_indices, reg, tolerance)
        
        self.children: List[Optional['SimplexTree']] = []
        self.parent: Optional['SimplexTree'] = None
        self.depth: int = 0
        self._node_count = 1
        self.splitting_point_index: Optional[int] = None
        self.splitting_point_vertex_index: Optional[int] = None
    
    def _get_root(self) -> 'SimplexTree':
        node = self
        while node.parent is not None:
            node = node.parent
        return node
    
    def _get_next_split_index(self) -> int:
        root = self._get_root()
        idx = root._split_counter[0]
        root._split_counter[0] += 1
        return idx
    
    def _add_child(self, child_vertices: List[Tuple[float, float]]) -> 'SimplexTree':
        child = SimplexTree(child_vertices, self.tolerance, self.registry, _is_child=True)
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child
    
    def _is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def _get_children(self) -> List['SimplexTree']:
        return self.children.copy()

    def _remove_splitting_point(self, splitting_point_index: int) -> bool: ## TODO : i need to implement a check to see if other components need the point im removing 
        for node in self._traverse_breadth_first():
            if node.splitting_point_index == splitting_point_index:
                for child in node.children:
                    if len(child.children) > 0:
                        return False
                node.children.clear()
                node.splitting_point_index = None
                node.splitting_point_vertex_index = None
                return True
        return False
    
    def remove_by_leaf_key(self, vertex_key: frozenset) -> bool:
        """
        Removes a leaf simplex by undoing its parent's split.
        
        Clears the parent's splitting point and removes all sibling children,
        restoring the parent to a leaf state.
        
        Args:
            vertex_key: Frozenset of vertex indices identifying the leaf to remove
            
        Returns:
            True if removal succeeded, False if leaf not found or has children
        """
        for node in self._traverse_breadth_first():
            if frozenset(node.vertex_indices) == vertex_key:
                parent = node.parent
                if parent is not None and parent.splitting_point_index is not None:
                    return self._remove_splitting_point(parent.splitting_point_index)
                return False
        return False
    
    def _traverse_breadth_first(self) -> Iterator['SimplexTree']: ## TODO: Change it to private writing
        from collections import deque
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            for child in node._get_children():
                queue.append(child)
    
    def get_leaves(self) -> List['SimplexTree']:
        """
        Returns all leaf nodes (simplices with no children) in the tree.
        
        Returns:
            List of SimplexTree nodes that have no children
        """
        leaves = []
        for node in self._traverse_breadth_first():
            if node._is_leaf():
                leaves.append(node)
        return leaves
    
    def find_containing_simplex(self, point: Tuple[float, float]) -> Optional['SimplexTree']:
        """
        Finds the leaf simplex containing the given point.
        
        Recursively searches the tree to find the smallest (deepest) simplex
        that contains the point.
        
        Args:
            point: Coordinates (x, y) to locate
            
        Returns:
            The leaf SimplexTree containing the point, or None if outside tree
        """
        if not self._point_inside_simplex(point):
            return None
        
        if self._is_leaf():
            return self
        
        for child in self._get_children():
            result = child.find_containing_simplex(point)
            if result is not None:
                return result
        
        return self
    
    def __repr__(self):
        vertices = self.get_vertices_as_tuples()
        vertex_str = str(vertices)
        return f"{self.__class__.__name__}(vertices={vertex_str}, children={len(self.children)}, depth={self.depth})"
    
    def _add_splitting_point(self, point: Tuple[float, float]) -> List['SimplexTree']: ## TODO: it should not be Tuple[float, float] but an array of D variables 
        if not self._is_leaf():
            for child in self.children:
                if child._point_inside_simplex(point):
                    return child._add_splitting_point(point)

        if not self._point_inside_simplex(point):
            raise ValueError(f"Splitting point {point} is not inside this simplex")

        self.splitting_point_index = self._get_next_split_index()
        vertex_indices = self.registry.register_vertices([tuple(point)])
        self.splitting_point_vertex_index = vertex_indices[0]

        vertices = self.get_vertices_as_tuples()
        
        if len(vertices) == 3:
            children = []
            for i in range(3):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 3]
                child_vertices = [v1, v2, tuple(point)]
                child = self._add_child(child_vertices)
                children.append(child)
            return children
        else:
            extended_vertices = list(vertices) + [tuple(point)]
            extended_child = self._add_child(extended_vertices)
            return [extended_child]
    
    def _compute_barycentric_center(self) -> Tuple[float, float]: ## TODO: private writing
        vertices = self.vertices
        center = np.mean(vertices, axis=0)
        return tuple(float(x) for x in center)
    
    
    def _add_barycentric_centers_recursively(self, levels: int) -> None: 
        if levels <= 0:
            return
        
        if self._is_leaf():
            barycenter = self._compute_barycentric_center()
            children = self._add_splitting_point(barycenter)
        
        for child in self.children:
            child._add_barycentric_centers_recursively(levels - 1)

    def print_tree(self, show_only_splitting_points: bool = False) -> None:
        """
        Prints the tree structure to console.
        
        Args:
            show_only_splitting_points: If True, shows only splitting point indices.
                                   If False, shows all nodes with vertex indices.
        """
        def _print(node, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            if show_only_splitting_points:
                if node.splitting_point_index is not None:
                    print(f"{prefix}{connector}[{node.splitting_point_index}] (vertex {node.splitting_point_vertex_index})")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    child_count = len(node.children)
                    for idx, child in enumerate(node.children):
                        _print(child, new_prefix, idx == child_count - 1)
            else:
                if node.splitting_point_index is not None:
                    label = f"[{node.splitting_point_index}] vertices: {node.vertex_indices}"
                else:
                    label = f"vertices: {node.vertex_indices}"
                print(f"{prefix}{connector}{label}")
                new_prefix = prefix + ("    " if is_last else "│   ")
                child_count = len(node.children)
                for idx, child in enumerate(node.children):
                    _print(child, new_prefix, idx == child_count - 1)

        _print(self)
    
    def get_splitting_points(self) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Returns all splitting points currently in the tree.
        
        Each splitting point is the center used to subdivide a simplex into children.
        
        Returns:
            List of (split_index, (x, y)) tuples for each splitting point
        """
        splitting_points = []
        for node in self._traverse_breadth_first():
            if node.splitting_point_index is not None:
                coords = tuple(self.registry._get_vertex(node.splitting_point_vertex_index))
                splitting_points.append((node.splitting_point_index, coords))
        return splitting_points
    
    