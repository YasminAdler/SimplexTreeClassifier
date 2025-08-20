"""
Example showing how to handle more than 3 points in 2D SimplexTree
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'classes'))

from simplexTree import SimplexTree
from utilss.visualization import visualize_simplex_tree
import matplotlib.pyplot as plt


def create_tree_from_multiple_points():
    """Demonstrate different ways to handle more than 3 points."""
    
    # Example: We have 5 points but SimplexTree needs exactly 3
    all_points = [(0, 0), (2, 0), (1, 2), (0.5, 0.5), (1.5, 0.8)]
    
    print("=" * 60)
    print("HANDLING MULTIPLE POINTS IN 2D")
    print("=" * 60)
    
    # Method 1: Use first 3 points as base triangle, add others as splitting points
    print("\nMethod 1: Base triangle + splitting points")
    print("-" * 40)
    
    # Create base triangle from first 3 points
    base_vertices = all_points[:3]
    tree1 = SimplexTree(base_vertices)
    print(f"Base triangle: {base_vertices}")
    
    # Add remaining points as splitting points
    remaining_points = all_points[3:]
    for point in remaining_points:
        try:
            tree1.add_point_to_the_most_specific_simplex(point)
            print(f"Added point {point} successfully")
        except ValueError as e:
            print(f"Could not add point {point}: {e}")
    
    print(f"\nFinal tree statistics:")
    print(f"  Total nodes: {tree1.get_node_count()}")
    print(f"  Leaf nodes: {len(tree1.get_leaves())}")
    
    visualize_simplex_tree(tree1, None, "Method 1: Base Triangle + Splitting Points")
    
    
    # Method 2: Create a large triangle that contains all points
    print("\n\nMethod 2: Bounding triangle")
    print("-" * 40)
    
    # Find bounding box
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Create a triangle that bounds all points with some margin
    margin = 0.5
    bounding_vertices = [
        (min_x - margin, min_y - margin),
        (max_x + margin, min_y - margin),
        ((min_x + max_x) / 2, max_y + margin)
    ]
    
    tree2 = SimplexTree(bounding_vertices)
    print(f"Bounding triangle: {bounding_vertices}")
    
    # Add all original points
    for point in all_points:
        tree2.add_point_to_the_most_specific_simplex(point)
        print(f"Added point {point}")
    
    print(f"\nFinal tree statistics:")
    print(f"  Total nodes: {tree2.get_node_count()}")
    print(f"  Leaf nodes: {len(tree2.get_leaves())}")
    
    visualize_simplex_tree(tree2, None, "Method 2: Bounding Triangle with All Points")
    
    
    # Method 3: Choose best 3 points (largest area)
    print("\n\nMethod 3: Choose best triangle")
    print("-" * 40)
    
    # Function to compute triangle area
    def triangle_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
    
    # Find the 3 points that form the largest triangle
    best_area = 0
    best_vertices = None
    
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            for k in range(j + 1, len(all_points)):
                area = triangle_area(all_points[i], all_points[j], all_points[k])
                if area > best_area:
                    best_area = area
                    best_vertices = [all_points[i], all_points[j], all_points[k]]
    
    tree3 = SimplexTree(best_vertices)
    print(f"Best triangle (area={best_area:.2f}): {best_vertices}")
    
    # Add remaining points
    for point in all_points:
        if point not in best_vertices:
            try:
                tree3.add_point_to_the_most_specific_simplex(point)
                print(f"Added point {point}")
            except ValueError as e:
                print(f"Point {point} is outside the triangle")
    
    visualize_simplex_tree(tree3, None, "Method 3: Largest Triangle + Remaining Points")
    
    
    # Show why we can't create SimplexTree with 4+ points directly
    print("\n\nDemonstration: Why 4+ points don't work")
    print("-" * 40)
    
    try:
        invalid_tree = SimplexTree(all_points[:4])  # Try with 4 points
    except ValueError as e:
        print(f"Error creating SimplexTree with 4 points: {e}")
        print("This is expected! SimplexTree requires exactly dimension + 1 vertices.")


if __name__ == "__main__":
    create_tree_from_multiple_points()