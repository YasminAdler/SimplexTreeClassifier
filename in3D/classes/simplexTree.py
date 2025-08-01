import numpy as np
from typing import List, Optional, Iterator, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilss.visualization import visualize_simplex_tree
from simplex import Simplex

VISUALIZATION_AVAILABLE = True


class SimplexTree(Simplex):
    def __init__(self, vertices: List[Tuple[float, ...]], tolerance: float = 1e-10):
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
    
    def add_child(self, child_vertices: List[Tuple[float, ...]]) -> 'SimplexTree':
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
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.get_children())
    
    def traverse_depth_first(self) -> Iterator['SimplexTree']:
        yield from self._dfs_traverse()
    
    def _dfs_traverse(self) -> Iterator['SimplexTree']:
        yield self
        for child in self.get_children():
            yield from child._dfs_traverse()
    
    def get_leaves(self) -> List['SimplexTree']:
        leaves = []
        for node in self.traverse_depth_first():
            if node.is_leaf():
                leaves.append(node)
        return leaves
    
    def get_nodes_at_depth(self, target_depth: int) -> List['SimplexTree']:
        nodes = []
        for node in self.traverse_depth_first():
            if node.depth == target_depth:
                nodes.append(node)
        return nodes
    
    def get_vertices_as_tuples(self) -> List[Tuple[float, ...]]:
        return [tuple(float(x) for x in v) for v in self.vertices]
    
    def find_containing_simplex(self, point: Tuple[float, ...]) -> Optional['SimplexTree']:     
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
    
    def add_splitting_point(self, point: Tuple[float, ...]) -> List['SimplexTree']:
        if not self.point_inside_simplex(point):
            raise ValueError(f"Point {point} is not inside this simplex")
        
        barycentric_coords = self.embed_point(point)
        if barycentric_coords is None:
            raise ValueError(f"Could not compute barycentric coordinates for point {point}")
        
        child_simplexes = []
        for i in range(self.n_vertices):
            new_vertices = list(self.vertices)
            new_vertices[i] = tuple(point)
            child_simplexes.append(new_vertices)
        
        children = []
        for child_vertices in child_simplexes:
            child = self.add_child(child_vertices)
            children.append(child)
        
        return children

    def add_point_to_the_most_specific_simplex(self, point: Tuple[float, ...]) -> List['SimplexTree']:
        containing_simplex = self.find_containing_simplex(point)
        
        if containing_simplex is None:
            raise ValueError(f"Point {point} is not inside any simplex in this tree")
        
        return containing_simplex.add_splitting_point(point)
    
    def print_tree(self) -> None:
        def format_vertices(vertices):
            return str([tuple(v) for v in vertices])
        
        def print_node(node, indent=0):
            spaces = "  " * indent
            vertices_str = format_vertices(node.get_vertices_as_tuples())
            print(f"{spaces}SimplexTree(")
            print(f"{spaces}  vertices={vertices_str},")
            print(f"{spaces}  depth={node.depth},")
            print(f"{spaces}  is_leaf={node.is_leaf()},")
            print(f"{spaces}  children={node.get_child_count()}")
            if node.children:
                print(f"{spaces}  children_details=[")
                for child in node.children:
                    print_node(child, indent + 2)
                print(f"{spaces}  ]")
            print(f"{spaces})")
        
        print("TREE FORMAT:")
        print_node(self)


if __name__ == "__main__":
    print(" " * 60)
    print("TESTING SIMPLEX TREE")
    print("-" * 60)
    
    vertices_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)] 
    tree_3d = SimplexTree(vertices_3d)
    test_point = (0.3, 0.4, 0.3)

    print(f"Point {test_point} inside simplex {vertices_3d} : {tree_3d.point_inside_simplex(test_point)}")
    
    embedded = tree_3d.embed_point(test_point)
    print(f"Embedded point coordinates: {embedded}")

    print("-" * 60)
    print("Adding splitting points to the most specific simplex recursively")
    tree_3d.add_point_to_the_most_specific_simplex(test_point)

    print("\n" + "-" * 60)
    tree_3d.print_tree()
    
    print("\n" + "-" * 60)
    print("Visualizing the simplex tree...")
    visualize_simplex_tree(tree_3d, test_point, "3D Tetrahedron Splitting")