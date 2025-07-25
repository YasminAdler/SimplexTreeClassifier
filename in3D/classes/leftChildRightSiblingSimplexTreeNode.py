from typing import List, Optional
from .simplexTree import SimplexTree


class LeftChildRightSiblingSimplexTreeNode:
    def __init__(self, simplex_tree: SimplexTree):
        self.simplex_tree = simplex_tree
        self.left_child: Optional['LeftChildRightSiblingSimplexTreeNode'] = None
        self.right_sibling: Optional['LeftChildRightSiblingSimplexTreeNode'] = None
        self.parent: Optional['LeftChildRightSiblingSimplexTreeNode'] = None
        self.depth: int = 0
    
    def add_child(self, child_node: 'LeftChildRightSiblingSimplexTreeNode') -> None:
        child_node.parent = self
        child_node.depth = self.depth + 1
        
        if self.left_child is None:
            self.left_child = child_node
        else:
            current = self.left_child
            while current.right_sibling is not None:
                current = current.right_sibling
            current.right_sibling = child_node
    
    def get_children(self) -> List['LeftChildRightSiblingSimplexTreeNode']:
        children = []
        current = self.left_child
        while current is not None:
            children.append(current)
            current = current.right_sibling
        return children
    
    def remove_child(self, child_node: 'LeftChildRightSiblingSimplexTreeNode') -> bool:
        if self.left_child is None:
            return False
        
        if self.left_child == child_node:
            self.left_child = child_node.right_sibling
            child_node.parent = None
            child_node.right_sibling = None
            return True
        
        current = self.left_child
        while current.right_sibling is not None:
            if current.right_sibling == child_node:
                current.right_sibling = child_node.right_sibling
                child_node.parent = None
                child_node.right_sibling = None
                return True
            current = current.right_sibling
        
        return False
    
    def is_leaf(self) -> bool:
        return self.left_child is None
    
    def get_child_count(self) -> int:
        count = 0
        current = self.left_child
        while current is not None:
            count += 1
            current = current.right_sibling
        return count
    
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
        return f"LeftChildRightSiblingNode({node_type}, depth={self.depth}, children={self.get_child_count()})" 