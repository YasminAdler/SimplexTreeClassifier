#!/usr/bin/env python3
"""
Comprehensive Example Usage of SimplexTreeClassifier

This script demonstrates all the key features of the SimplexTreeClassifier project:
- 2D and 3D simplex point location
- Hierarchical tree structures
- Multiple tree node implementations
- Visualization capabilities
- Performance characteristics
"""

import sys
import os
import numpy as np
from typing import List, Tuple

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'in2D'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'in3D', 'classes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'in3D', 'utilss'))

# Import 2D components
from simplex_point_location import Triangle, SimplexPointLocator

# Import 3D components
from simplex import Simplex
from simplexTree import SimplexTree
from parentChildSimplexTreeNode import ParentChildSimplexTreeNode
from leftChildRightSiblingSimplexTreeNode import LeftChildRightSiblingSimplexTreeNode

# Import visualization
try:
    from visualization import visualize_simplex_tree
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Visualization not available. Install matplotlib for 3D plots.")
    VISUALIZATION_AVAILABLE = False


def demo_2d_point_location():
    """Demonstrate 2D simplex point location with triangles."""
    print("\n" + "="*60)
    print("2D SIMPLEX POINT LOCATION DEMO")
    print("="*60)
    
    # Create triangles
    triangle1 = Triangle((0, 0), (4, 0), (2, 2))
    triangle2 = Triangle((0, 0), (2, 2), (0, 4))
    
    # Create locator
    locator = SimplexPointLocator([triangle1, triangle2])
    
    # Test points
    test_points = [(2, 0.5), (1, 1), (0.5, 2), (3, 1), (5, 5)]
    
    print(f"Triangles:")
    print(f"  Triangle 1: {triangle1}")
    print(f"  Triangle 2: {triangle2}")
    print()
    
    for point in test_points:
        result = locator.find_containing_simplex_with_coords(point)
        if result:
            index, triangle, coords = result
            print(f"Point {point} is in Triangle {index + 1}")
            print(f"  Barycentric coordinates: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")
            print(f"  Sum: {sum(coords):.3f}")
        else:
            print(f"Point {point} is not in any triangle")


def demo_3d_simplex_basic():
    """Demonstrate basic 3D simplex operations."""
    print("\n" + "="*60)
    print("3D SIMPLEX BASIC OPERATIONS DEMO")
    print("="*60)
    
    # Create 3D tetrahedron
    vertices_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    simplex_3d = Simplex(vertices_3d)
    
    print(f"3D Tetrahedron vertices: {vertices_3d}")
    print(f"Dimension: {simplex_3d.dimension}")
    print(f"Number of vertices: {simplex_3d.n_vertices}")
    print(f"Is degenerate: {simplex_3d.is_degenerate}")
    print(f"Determinant: {simplex_3d.get_determinant():.6f}")
    
    # Test points
    test_points = [(0.3, 0.4, 0.3), (0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.1, 0.1, 0.1)]
    
    for point in test_points:
        is_inside = simplex_3d.point_inside_simplex(point)
        coords = simplex_3d.embed_point(point)
        print(f"\nPoint {point}:")
        print(f"  Inside simplex: {is_inside}")
        if coords:
            print(f"  Barycentric coordinates: {[f'{c:.3f}' for c in coords]}")
            print(f"  Sum: {sum(coords):.3f}")


def demo_3d_tree_hierarchy():
    """Demonstrate 3D hierarchical tree structure."""
    print("\n" + "="*60)
    print("3D HIERARCHICAL TREE DEMO")
    print("="*60)
    
    # Create root tetrahedron
    vertices_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tree_3d = SimplexTree(vertices_3d)
    
    print(f"Root tetrahedron: {vertices_3d}")
    print(f"Initial tree depth: {tree_3d.get_depth()}")
    print(f"Initial node count: {tree_3d.get_node_count()}")
    
    # Add splitting points to create hierarchy
    splitting_points = [(0.3, 0.4, 0.3), (0.7, 0.2, 0.1)]
    
    for i, point in enumerate(splitting_points, 1):
        print(f"\nAdding splitting point {i}: {point}")
        children = tree_3d.add_point_to_the_most_specific_simplex(point)
        print(f"  Created {len(children)} children")
        print(f"  New tree depth: {tree_3d.get_depth()}")
        print(f"  New node count: {tree_3d.get_node_count()}")
    
    # Get tree statistics
    leaves = tree_3d.get_leaves()
    print(f"\nTree Statistics:")
    print(f"  Total leaves: {len(leaves)}")
    print(f"  Max depth: {tree_3d.get_depth()}")
    print(f"  Total nodes: {tree_3d.get_node_count()}")
    
    # Test point location
    test_point = (0.2, 0.3, 0.2)
    containing_simplex = tree_3d.find_containing_simplex(test_point)
    if containing_simplex:
        print(f"\nPoint {test_point} found in simplex:")
        print(f"  Vertices: {[tuple(v) for v in containing_simplex.vertices]}")
        print(f"  Depth: {containing_simplex.depth}")
        print(f"  Is leaf: {containing_simplex.is_leaf()}")


def demo_tree_node_wrappers():
    """Demonstrate tree node wrapper implementations."""
    print("\n" + "="*60)
    print("TREE NODE WRAPPERS DEMO")
    print("="*60)
    
    # Create a simple tree
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tree = SimplexTree(vertices)
    tree.add_point_to_the_most_specific_simplex((0.3, 0.4, 0.3))
    
    # Test ParentChildSimplexTreeNode
    print("ParentChildSimplexTreeNode:")
    parent_child_root = ParentChildSimplexTreeNode(tree)
    
    # Add children to the wrapper
    for child_tree in tree.get_children():
        child_node = ParentChildSimplexTreeNode(child_tree)
        parent_child_root.add_child(child_node)
    
    print(f"  Root node type: {'SimplexTree' if parent_child_root.is_simplex_tree() else 'Simplex'}")
    print(f"  Child count: {parent_child_root.get_child_count()}")
    print(f"  Is leaf: {parent_child_root.is_leaf()}")
    
    # Test point location
    test_point = (0.2, 0.3, 0.2)
    result = parent_child_root.find_containing_simplex(test_point)
    if result:
        print(f"  Point {test_point} found in simplex at depth {result.depth}")
    
    # Test LeftChildRightSiblingSimplexTreeNode
    print("\nLeftChildRightSiblingSimplexTreeNode:")
    left_right_root = LeftChildRightSiblingSimplexTreeNode(tree)
    
    # Add children to the wrapper
    for child_tree in tree.get_children():
        child_node = LeftChildRightSiblingSimplexTreeNode(child_tree)
        left_right_root.add_child(child_node)
    
    print(f"  Root node type: {'SimplexTree' if left_right_root.is_simplex_tree() else 'Simplex'}")
    print(f"  Child count: {left_right_root.get_child_count()}")
    print(f"  Is leaf: {left_right_root.is_leaf()}")
    
    # Test point location
    result = left_right_root.find_containing_simplex(test_point)
    if result:
        print(f"  Point {test_point} found in simplex at depth {result.depth}")


def demo_visualization():
    """Demonstrate 3D visualization capabilities."""
    if not VISUALIZATION_AVAILABLE:
        print("\n" + "="*60)
        print("VISUALIZATION DEMO (SKIPPED)")
        print("="*60)
        print("Install matplotlib to enable 3D visualization")
        return
    
    print("\n" + "="*60)
    print("3D VISUALIZATION DEMO")
    print("="*60)
    
    # Create a complex tree for visualization
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tree = SimplexTree(vertices)
    
    # Add multiple splitting points
    splitting_points = [(0.3, 0.4, 0.3), (0.7, 0.2, 0.1), (0.2, 0.6, 0.2)]
    
    for point in splitting_points:
        tree.add_point_to_the_most_specific_simplex(point)
    
    print(f"Created tree with {tree.get_node_count()} nodes")
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {len(tree.get_leaves())}")
    
    # Visualize the tree
    print("Opening 3D visualization window...")
    visualize_simplex_tree(tree, splitting_points[0], "Complex 3D Simplex Tree")
    print("Visualization complete!")


def demo_performance_comparison():
    """Demonstrate performance characteristics."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON DEMO")
    print("="*60)
    
    import time
    
    # Test different tree sizes
    tree_sizes = [1, 5, 10, 20]
    
    for size in tree_sizes:
        print(f"\nTesting tree with {size} splitting points:")
        
        # Create tree
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        tree = SimplexTree(vertices)
        
        # Add splitting points
        start_time = time.time()
        for i in range(size):
            point = (0.1 + 0.8 * i / size, 0.1 + 0.8 * (i % 3) / 3, 0.1 + 0.8 * (i % 2) / 2)
            tree.add_point_to_the_most_specific_simplex(point)
        build_time = time.time() - start_time
        
        # Test point location performance
        test_points = [(0.2, 0.3, 0.2), (0.5, 0.5, 0.5), (0.8, 0.2, 0.8)]
        start_time = time.time()
        for point in test_points:
            tree.find_containing_simplex(point)
        query_time = time.time() - start_time
        
        print(f"  Build time: {build_time:.4f}s")
        print(f"  Query time: {query_time:.4f}s")
        print(f"  Node count: {tree.get_node_count()}")
        print(f"  Tree depth: {tree.get_depth()}")


def main():
    """Run all demonstrations."""
    print("SimplexTreeClassifier - Comprehensive Demo")
    print("="*60)
    
    try:
        demo_2d_point_location()
        demo_3d_simplex_basic()
        demo_3d_tree_hierarchy()
        demo_tree_node_wrappers()
        demo_visualization()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("All demonstrations completed successfully.")
        print("Check the documentation for more detailed information.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please check your installation and dependencies.")


if __name__ == "__main__":
    main() 