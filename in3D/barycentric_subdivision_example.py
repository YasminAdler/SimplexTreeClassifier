"""
Example script demonstrating barycentric subdivision of 3D SimplexTree.

This shows how to automatically subdivide tetrahedra by adding their
barycentric centers (centroids) as splitting points.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'classes'))

from simplexTree import SimplexTree
from utilss.visualization import visualize_simplex_tree


def demonstrate_3d_barycentric_subdivision():
    """Demonstrate barycentric subdivision in 3D."""
    
    print("=" * 80)
    print("3D BARYCENTRIC SUBDIVISION DEMONSTRATION")
    print("=" * 80)
    
    # Create a regular tetrahedron
    vertices = [(0, 0, 0), (1, 0, 0), (0.5, 0.866, 0), (0.5, 0.289, 0.816)]
    
    # Example 1: Single level barycentric subdivision
    print("\n1. SINGLE LEVEL BARYCENTRIC SUBDIVISION")
    print("-" * 40)
    
    tree1 = SimplexTree(vertices)
    barycenter = tree1.compute_barycentric_center()
    print(f"Original tetrahedron vertices: {[tuple(round(x, 3) for x in v) for v in vertices]}")
    print(f"Barycentric center: {tuple(round(x, 3) for x in barycenter)}")
    
    # Add barycentric center to root
    tree1.add_splitting_point(barycenter)
    print(f"After subdivision: {tree1.get_child_count()} children created")
    print(f"(In 3D, each tetrahedron splits into 4 sub-tetrahedra)")
    
    # Visualize
    visualize_simplex_tree(tree1, barycenter, "3D Single Level Barycentric Subdivision")
    
    
    # Example 2: Recursive barycentric subdivision
    print("\n2. RECURSIVE BARYCENTRIC SUBDIVISION")
    print("-" * 40)
    
    tree2 = SimplexTree(vertices)
    print("Adding barycentric centers recursively for 2 levels...")
    tree2.add_barycentric_centers_recursively(2)
    
    print(f"\nFinal tree statistics:")
    print(f"  Total nodes: {tree2.get_node_count()}")
    print(f"  Tree depth: {tree2.get_depth()}")
    print(f"  Leaf nodes: {len(tree2.get_leaves())}")
    print(f"  Branching factor: 4 (dimension + 1)")
    
    # Visualize
    visualize_simplex_tree(tree2, None, "3D 2-Level Recursive Barycentric Subdivision")
    
    
    # Example 3: Volume analysis
    print("\n3. VOLUME ANALYSIS WITH SUBDIVISION")
    print("-" * 40)
    
    tree3 = SimplexTree(vertices)
    
    # Function to compute tetrahedron volume
    def compute_tetrahedron_volume(verts):
        import numpy as np
        v0, v1, v2, v3 = [np.array(v) for v in verts]
        # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        matrix = np.column_stack([v1-v0, v2-v0, v3-v0])
        volume = abs(np.linalg.det(matrix)) / 6
        return volume
    
    original_volume = compute_tetrahedron_volume(vertices)
    print(f"Original tetrahedron volume: {original_volume:.6f}")
    
    # Subdivide and analyze
    for level in range(3):
        tree3.add_barycentric_centers_to_all_leaves()
        leaves = tree3.get_leaves()
        
        # Compute total volume of leaves (should equal original)
        total_volume = sum(compute_tetrahedron_volume(leaf.get_vertices_as_tuples()) 
                          for leaf in leaves)
        avg_volume = total_volume / len(leaves)
        
        print(f"\nLevel {level + 1}:")
        print(f"  Number of sub-tetrahedra: {len(leaves)}")
        print(f"  Average volume: {avg_volume:.6f}")
        print(f"  Total volume: {total_volume:.6f} (should equal {original_volume:.6f})")
        print(f"  Volume ratio: {avg_volume/original_volume:.6f}")
    
    # Visualize
    visualize_simplex_tree(tree3, None, "3D Volume-Preserving Subdivision")
    
    
    # Example 4: Mixed subdivision with analysis
    print("\n4. MIXED SUBDIVISION WITH POINT LOCATION")
    print("-" * 40)
    
    tree4 = SimplexTree(vertices)
    
    # Add custom point
    custom_point = (0.4, 0.3, 0.2)
    print(f"Adding custom point: {custom_point}")
    tree4.add_point_to_the_most_specific_simplex(custom_point)
    
    # Add barycentric centers
    print("Adding barycentric centers to all leaves...")
    tree4.add_barycentric_centers_to_all_leaves()
    
    # Test point location
    test_points = [
        (0.25, 0.25, 0.25),  # Near center
        (0.1, 0.1, 0.1),      # Near vertex
        (0.8, 0.1, 0.1),      # Near another vertex
    ]
    
    print("\nTesting point location:")
    for point in test_points:
        containing = tree4.find_containing_simplex(point)
        if containing:
            print(f"  Point {point} is in simplex at depth {containing.depth}")
        else:
            print(f"  Point {point} is outside the tetrahedron")
    
    # Visualize
    visualize_simplex_tree(tree4, custom_point, "3D Mixed Subdivision with Point Location")
    
    
    # Example 5: Performance analysis
    print("\n5. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    tree5 = SimplexTree(vertices)
    
    print("Subdivision growth analysis:")
    print("Level | Nodes | Leaves | Growth Factor")
    print("------|-------|--------|---------------")
    
    for level in range(5):
        if level > 0:
            prev_leaves = len(tree5.get_leaves())
            tree5.add_barycentric_centers_to_all_leaves()
            new_leaves = len(tree5.get_leaves())
            growth = new_leaves / prev_leaves
        else:
            tree5.add_barycentric_centers_to_all_leaves()
            new_leaves = len(tree5.get_leaves())
            growth = new_leaves
        
        nodes = tree5.get_node_count()
        print(f"  {level+1:2d}  |  {nodes:4d} | {new_leaves:6d} | {growth:13.1f}")
    
    print(f"\nMemory scaling: O(4^n) where n is the number of levels")
    print(f"Time complexity for point location: O(log N) where N is number of nodes")


if __name__ == "__main__":
    demonstrate_3d_barycentric_subdivision()