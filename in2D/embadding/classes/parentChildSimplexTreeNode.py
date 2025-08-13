from typing import List, Optional
from .simplexTree import SimplexTree


class ParentChildSimplexTreeNode:
    def __init__(self, simplex_tree: SimplexTree):
        self.simplex_tree = simplex_tree
        self.children: List['ParentChildSimplexTreeNode'] = []
        self.parent: Optional['ParentChildSimplexTreeNode'] = None
        self.depth: int = 0
    
    def add_child(self, child_node: 'ParentChildSimplexTreeNode') -> None:
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)
    
    def get_children(self) -> List['ParentChildSimplexTreeNode']:
        return self.children.copy()
    
    def remove_child(self, child_node: 'ParentChildSimplexTreeNode') -> bool:
        if child_node in self.children:
            self.children.remove(child_node)
            child_node.parent = None
            return True
        return False
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_child_count(self) -> int:
        return len(self.children)
    
    def is_simplex_tree(self) -> bool:
        return not self.simplex_tree.is_leaf()
    
    def is_simplex(self) -> bool:
        return self.simplex_tree.is_leaf()
    
    def point_inside_node(self, point) -> bool:
        return self.simplex_tree.point_inside_simplex(point)
    
    def find_containing_simplex(self, point) -> Optional[SimplexTree]:
        if not self.point_inside_node(point):
            return None
        
        if self.is_simplex():
            return self.simplex_tree
        
        for child in self.get_children():
            result = child.find_containing_simplex(point)
            if result is not None:
                return result
                
        return self.simplex_tree
    
    def __repr__(self):
        node_type = "SimplexTree" if self.is_simplex_tree() else "Simplex"
        return f"ParentChildNode({node_type}, depth={self.depth}, children={self.get_child_count()})" 