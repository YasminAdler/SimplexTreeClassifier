#!/usr/bin/env python3
"""
Test script for 2D Simplex Tree functionality
Demonstrates the hierarchical triangle subdivision and point location
"""

from classes.simplexTree import SimplexTree
from classes.parentChildSimplexTreeNode import ParentChildSimplexTreeNode
from classes.leftChildRightSiblingSimplexTreeNode import LeftChildRightSiblingSimplexTreeNode
from utilss.visualization import visualize_simplex_tree, visualize_triangle_with_point


def test_basic_2d_simplex():
    """Test basic 2D simplex (triangle) functionality"""
    print("=" * 60)
    print("TESTING BASIC 2D SIMPLEX (TRIANGLE)")
    print("=" * 60)
    
    # Create a triangle
    vertices = [(0, 0), (4, 0), (2, 3)]
    simplex = SimplexTree(vertices)
    
    print(f"Triangle vertices: {vertices}")
    print(f"Triangle dimension: {simplex.dimension}")
    print(f"Number of vertices: {simplex.n_vertices}")
    print(f"Is degenerate: {simplex.is_degenerate}")
    
    # Test point inside
    test_point = (2, 1)
    is_inside = simplex.point_inside_simplex(test_point)
    print(f"Point {test_point} inside triangle: {is_inside}")
    
    if is_inside:
        coords = simplex.embed_point(test_point)
        print(f"Barycentric coordinates: {coords}")
    
    # Visualize the triangle with point
    visualize_triangle_with_point(vertices, test_point, "Basic 2D Triangle Test")


def test_2d_simplex_tree():
    """Test 2D simplex tree with hierarchical subdivision"""
    print("\n" + "=" * 60)
    print("TESTING 2D SIMPLEX TREE WITH SUBDIVISION")
    print("=" * 60)
    
    # Create root triangle
    vertices_2d = [(0, 0), (4, 0), (2, 3)]
    tree_2d = SimplexTree(vertices_2d)
    
    print(f"Root triangle: {vertices_2d}")
    print(f"Initial tree depth: {tree_2d.get_depth()}")
    print(f"Initial node count: {tree_2d.get_node_count()}")
    
    # Add splitting points
    splitting_points = [
        (1.5, 1.0),
        (1.0, 0.5),
        (2.5, 1.5)
    ]
    
    for i, point in enumerate(splitting_points):
        print(f"\nAdding splitting point {i+1}: {point}")
        try:
            children = tree_2d.add_point_to_the_most_specific_simplex(point)
            print(f"Created {len(children)} child triangles")
        except ValueError as e:
            print(f"Error: {e}")
    
    print(f"\nFinal tree depth: {tree_2d.get_depth()}")
    print(f"Final node count: {tree_2d.get_node_count()}")
    
    # Print tree structure
    print("\n" + "-" * 40)
    tree_2d.print_tree()
    
    # Visualize the tree
    print("\n" + "-" * 40)
    print("Visualizing the simplex tree...")
    visualize_simplex_tree(tree_2d, splitting_points[0], "2D Triangle Tree Subdivision")


def test_tree_node_wrappers():
    """Test the tree node wrapper classes"""
    print("\n" + "=" * 60)
    print("TESTING TREE NODE WRAPPERS")
    print("=" * 60)
    
    # Create a simple tree
    vertices = [(0, 0), (4, 0), (2, 3)]
    tree = SimplexTree(vertices)
    tree.add_point_to_the_most_specific_simplex((1.5, 1.0))
    
    # Test ParentChildSimplexTreeNode
    print("Testing ParentChildSimplexTreeNode:")
    parent_child_node = ParentChildSimplexTreeNode(tree)
    
    # Create child nodes for the children of the tree
    for child_tree in tree.get_children():
        child_node = ParentChildSimplexTreeNode(child_tree)
        parent_child_node.add_child(child_node)
    
    print(f"Node type: {'SimplexTree' if parent_child_node.is_simplex_tree() else 'Simplex'}")
    print(f"Number of children: {parent_child_node.get_child_count()}")
    print(f"Is leaf: {parent_child_node.is_leaf()}")
    
    # Test point location
    test_point = (1.0, 0.5)
    containing_simplex = parent_child_node.find_containing_simplex(test_point)
    if containing_simplex:
        print(f"Point {test_point} found in simplex: {containing_simplex.get_vertices_as_tuples()}")
    
    # Test LeftChildRightSiblingSimplexTreeNode
    print("\nTesting LeftChildRightSiblingSimplexTreeNode:")
    lcrs_node = LeftChildRightSiblingSimplexTreeNode(tree)
    
    # Create child nodes for the children of the tree
    for child_tree in tree.get_children():
        child_node = LeftChildRightSiblingSimplexTreeNode(child_tree)
        lcrs_node.add_child(child_node)
    
    print(f"Node type: {'SimplexTree' if lcrs_node.is_simplex_tree() else 'Simplex'}")
    print(f"Number of children: {lcrs_node.get_child_count()}")
    print(f"Is leaf: {lcrs_node.is_leaf()}")
    
    # Test point location
    containing_simplex = lcrs_node.find_containing_simplex(test_point)
    if containing_simplex:
        print(f"Point {test_point} found in simplex: {containing_simplex.get_vertices_as_tuples()}")


def test_point_location_algorithm():
    """Test the point location algorithm with multiple points"""
    print("\n" + "=" * 60)
    print("TESTING POINT LOCATION ALGORITHM")
    print("=" * 60)
    
    # Create a complex tree
    vertices = [(0, 0), (6, 0), (3, 4)]
    tree = SimplexTree(vertices)
    
    # Add multiple splitting points to create a complex hierarchy
    splitting_points = [
        (2, 1.5),
        (4, 1.5),
        (1, 0.5),
        (5, 0.5),
        (3, 2.5)
    ]
    
    for point in splitting_points:
        try:
            tree.add_point_to_the_most_specific_simplex(point)
        except ValueError as e:
            print(f"Error adding point {point}: {e}")
    
    # Test points in different regions
    test_points = [
        (1, 0.5),   # Should be in a leaf simplex
        (3, 2),     # Should be in the root or a child
        (5, 1),     # Should be in a specific child
        (0.5, 0.2), # Should be in a leaf simplex
        (4.5, 0.8)  # Should be in a specific child
    ]
    
    print("Point location results:")
    for point in test_points:
        containing_simplex = tree.find_containing_simplex(point)
        if containing_simplex:
            print(f"Point {point} → Simplex {containing_simplex.get_vertices_as_tuples()}")
        else:
            print(f"Point {point} → Not found in any simplex")
    
    # Visualize the final tree
    print("\n" + "-" * 40)
    print("Visualizing complex tree...")
    visualize_simplex_tree(tree, splitting_points[0], "Complex 2D Triangle Tree")


if __name__ == "__main__":
    print("2D SIMPLEX TREE CLASSIFIER TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_2d_simplex()
        test_2d_simplex_tree()
        test_tree_node_wrappers()
        test_point_location_algorithm()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc() 