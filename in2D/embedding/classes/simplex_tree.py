import numpy as np
from typing import List, Optional, Iterator, Tuple
import sys
import os
from embedding.utilss.visualization import visualize_simplex_tree
from .simplex import Simplex

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)



class SimplexTree(Simplex):
    def __init__(self, vertices: List[Tuple[float, float]], tolerance: float = 1e-10):
        super().__init__(vertices, tolerance)
        self.children: List[Optional['SimplexTree']] = []
        self.parent: Optional['SimplexTree'] = None
        self.depth: int = 0
        self._node_count = 1
    
    def get_node_count(self) -> int:
        count = 1
        for child in self.children:
            count += child.get_node_count()
        return count
    
    def get_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(child.get_depth() for child in self.children)
    
    def add_child(self, child_vertices: List[Tuple[float, float]]) -> 'SimplexTree':
        child = SimplexTree(child_vertices)
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_children(self) -> List['SimplexTree']:
        return self.children.copy()
    
    def get_child_count(self) -> int:
        return len(self.children)
    
    def traverse_breadth_first(self) -> Iterator['SimplexTree']:
        from collections import deque
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            for child in node.get_children():
                queue.append(child)
    
    def get_leaves(self) -> List['SimplexTree']:
        leaves = []
        for node in self.traverse_breadth_first():
            if node.is_leaf():
                leaves.append(node)
        return leaves
    
    def get_vertices_as_tuples(self) -> List[Tuple[float, float]]:
        return [tuple(float(x) for x in v) for v in self.vertices]
    
    def find_containing_simplex(self, point: Tuple[float, float]) -> Optional['SimplexTree']:     
        if not self.point_inside_simplex(point):
            return None
        
        if self.is_leaf():
            return self
        
        for child in self.get_children():
            result = child.find_containing_simplex(point)
            if result is not None:
                return result
        
        return self
    
    def __repr__(self):
        vertex_str = str([tuple(v) for v in self.vertices])
        return f"{self.__class__.__name__}(vertices={vertex_str}, children={len(self.children)}, depth={self.depth})"
    
    def add_splitting_point(self, point: Tuple[float, float]) -> List['SimplexTree']:
        if not self.is_leaf():
            for child in self.children:
                if child.point_inside_simplex(point):
                    return child.add_splitting_point(point)

        if not self.point_inside_simplex(point):
            raise ValueError(f"Splitting point {point} is not inside this simplex")

        barycentric_coords = self.embed_point(point)
        if barycentric_coords is None:
            raise ValueError(f"Could not compute barycentric coordinates for splitting point {point}")

        if len(self.vertices) == 3:
            children = []
            for i in range(3):
                v1 = tuple(self.vertices[i])
                v2 = tuple(self.vertices[(i + 1) % 3])
                child_vertices = [v1, v2, tuple(point)]
                child = self.add_child(child_vertices)
                children.append(child)
            return children
        else:
            extended_vertices = list(self.vertices) + [tuple(point)]
            extended_child = self.add_child(extended_vertices)
            return [extended_child]

    def subdivide_with_point(self, point: Tuple[float, float]) -> List['SimplexTree']:
        containing_simplex = self.find_containing_simplex(point)
    
        if containing_simplex is None:
            raise ValueError(f"Splitting point {point} is not inside any simplex in this tree")
        return containing_simplex.add_splitting_point(point)
    
    def embed_data_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, ...]]:
        containing_simplex = self.find_containing_simplex(point)
        if containing_simplex is None:
            print(f"DATA POINT: {point} is outside all simplexes")
            return None
        
        embedding = containing_simplex.embed_point(point)
        return embedding
    
    def compute_barycentric_center(self) -> Tuple[float, float]:
        center = np.mean(self.vertices, axis=0)
        return tuple(float(x) for x in center)
    
    def add_barycentric_centers_to_all_leaves(self) -> int:
        leaves = self.get_leaves()
        subdivided_count = 0
        
        for leaf in leaves:
            barycenter = leaf.compute_barycentric_center()
            children = leaf.add_splitting_point(barycenter)
            subdivided_count += len(children)
        
        return subdivided_count
    
    def add_barycentric_centers_recursively(self, levels: int) -> None:
        if levels <= 0:
            return
        
        if self.is_leaf():
            barycenter = self.compute_barycentric_center()
            children = self.add_splitting_point(barycenter)
        
        for child in self.children:
            child.add_barycentric_centers_recursively(levels - 1)

    def print_tree(self) -> None:
        def _print(node, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            vertices = ", ".join([f"({v[0]:.1f}, {v[1]:.1f})" for v in node.get_vertices_as_tuples()])
            print(f"{prefix}{connector}{vertices}")
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(node.children)
            for idx, child in enumerate(node.children):
                _print(child, new_prefix, idx == child_count - 1)

        _print(self)

if __name__ == "__main__":
    vertices = [(0, 0), (1, 0), (0.5, 1)] 
    tree = SimplexTree(vertices)
    
    barycenter = tree.compute_barycentric_center()
    print(f"Barycentric point of the initial triangle is: {barycenter}")
    visualize_simplex_tree(tree, None, "2D Triangle") 

    tree_barycentric = SimplexTree(vertices)

    tree_barycentric.add_barycentric_centers_recursively(1)
    visualize_simplex_tree(tree_barycentric, None, "tree_barycentric") 
    
    print("\nTree structure after barycentric subdivision:")
    print(f"Total nodes: {tree_barycentric.get_node_count()}")
    print(f"Tree depth: {tree_barycentric.get_depth()}")
    print(f"Leaf nodes: {len(tree_barycentric.get_leaves())}")
    
    visualize_simplex_tree(tree_barycentric, None, "tree_barycentric_after_subdivision") 

    tree_mixed = SimplexTree(vertices)
    test_point = (0.343, 0.2)
    
    print(f"Adding custom point: {test_point}")
    children = tree_mixed.subdivide_with_point(test_point)
    
    visualize_simplex_tree(tree_mixed, test_point, "tree with manually added point") 

    print("\nNow adding barycentric centers to all leaves...")
    count = tree_mixed.add_barycentric_centers_to_all_leaves()
    print(f"Subdivided {count} leaf simplexes")
    
    
    tree_mixed.add_barycentric_centers_recursively(2)

    visualize_simplex_tree(tree_mixed, None, "mixed") 
    
    children = tree_mixed.subdivide_with_point(test_point)

