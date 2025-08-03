#!/usr/bin/env python3
"""
Simple example demonstrating 2D Simplex Tree usage
"""

from classes.simplexTree import SimplexTree
from classes.parentChildSimplexTreeNode import ParentChildSimplexTreeNode
from utilss.visualization import visualize_simplex_tree


def main():
    print("2D Simplex Tree Example")
    print("=" * 40)
    
    # Create a triangle
    triangle_vertices = [(0, 0), (4, 0), (2, 3)]
    print(f"Creating triangle with vertices: {triangle_vertices}")
    
    # Create SimplexTree
    tree = SimplexTree(triangle_vertices)
    
    # Add a splitting point to create children
    splitting_point = (1.5, 1.0)
    print(f"Adding splitting point: {splitting_point}")
    
    children = tree.add_point_to_the_most_specific_simplex(splitting_point)
    print(f"Created {len(children)} child triangles")
    
    # Test point location
    test_point = (1.0, 0.5)
    print(f"Testing point location for: {test_point}")
    
    containing_simplex = tree.find_containing_simplex(test_point)
    if containing_simplex:
        print(f"Point found in simplex: {containing_simplex.get_vertices_as_tuples()}")
        
        # Get barycentric coordinates
        coords = containing_simplex.embed_point(test_point)
        print(f"Barycentric coordinates: {coords}")
    else:
        print("Point not found in any simplex")
    
    # Wrap in tree node for advanced operations
    root_node = ParentChildSimplexTreeNode(tree)
    
    # Create child nodes
    for child_tree in tree.get_children():
        child_node = ParentChildSimplexTreeNode(child_tree)
        root_node.add_child(child_node)
    
    print(f"\nTree structure:")
    print(f"- Root node type: {'SimplexTree' if root_node.is_simplex_tree() else 'Simplex'}")
    print(f"- Number of children: {root_node.get_child_count()}")
    print(f"- Tree depth: {tree.get_depth()}")
    print(f"- Total nodes: {tree.get_node_count()}")
    
    # Visualize the tree
    print("\nVisualizing the tree...")
    visualize_simplex_tree(tree, splitting_point, "2D Triangle Example")


if __name__ == "__main__":
    main() 