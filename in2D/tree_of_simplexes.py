import numpy as np
from simplex_point_location import Triangle, SimplexPointLocator

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def subdivide_triangle(v0, v1, v2):
    m01 = midpoint(v0, v1)
    m12 = midpoint(v1, v2)
    m20 = midpoint(v2, v0)
    
    return [
        Triangle(v0, m01, m20),
        Triangle(m01, v1, m12),
        Triangle(m20, m12, v2),
        Triangle(m01, m12, m20)
    ]

def create_tree_of_simplexes():
    triangles = []
    
    main_v0 = (0, 0)
    main_v1 = (8, 0)
    main_v2 = (4, 6)
    
    level1_triangles = subdivide_triangle(main_v0, main_v1, main_v2)
    
    level2_triangles = subdivide_triangle(
        level1_triangles[0].v0, 
        level1_triangles[0].v1, 
        level1_triangles[0].v2
    )
    
    level3_triangles = subdivide_triangle(
        level2_triangles[0].v0,
        level2_triangles[0].v1,
        level2_triangles[0].v2
    )
    
    level4_triangles = subdivide_triangle(
        level3_triangles[0].v0,
        level3_triangles[0].v1,
        level3_triangles[0].v2
    )
    
    triangles.extend(level1_triangles[1:])  
    triangles.extend(level2_triangles[1:])  
    triangles.extend(level3_triangles[1:])  
    triangles.extend(level4_triangles)
    
    return triangles

def test_tree_of_simplexes():
    triangles = create_tree_of_simplexes()
    locator = SimplexPointLocator(triangles)
    
    print(f"Created tree with {len(triangles)} triangles")
    print("=" * 50)
    
    test_points = [
        (0.5, 0.5),    
        (0.25, 0.25),  
        (0.125, 0.125), 
        (2, 1),        
        (6, 1),        
        (4, 4),        
        (1, 3),        
        (10, 10),      
    ]
    
    for point in test_points:
        print(f"\nTesting point {point}:")
        result = locator.find_containing_simplex_with_coords(point)
        
        if result:
            index, triangle, coords = result
            print(f"  ✓ Found in triangle {index}")
            print(f"  Vertices: {triangle.v0}, {triangle.v1}, {triangle.v2}")
            print(f"  Barycentric: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")
            
            area = float(abs(np.cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0)) / 2)
            if area > 10:
                level = "Level 1"
            elif area > 2.5:
                level = "Level 2"
            elif area > 0.6:
                level = "Level 3"
            else:
                level = "Level 4"
            print(f"  Subdivision level: {level} (area={area:.3f})")
        else:
            print(f"  ✗ Not found in any triangle")

def visualize_tree():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        triangles = create_tree_of_simplexes()
        
        fig, ax = plt.subplots(1, 1)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        for i, triangle in enumerate(triangles):
            color = colors[i % len(colors)]
            patch = patches.Polygon(
                [triangle.v0, triangle.v1, triangle.v2],
                alpha=0.6, 
                fc=color,
                ec='black',
                linewidth=1
            )
            ax.add_patch(patch)
            
            centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3
            ax.text(centroid[0], centroid[1], str(i), ha='center', va='center', size=8)
        
        ax.axis([-1, 9, -1, 7])
        ax.set_aspect('equal')
        ax.set_title('Tree of Simplexes - Hierarchical Triangle Subdivision')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")

if __name__ == "__main__":
    test_tree_of_simplexes()
    print("\n" + "=" * 50)
    print("Attempting to visualize...")
    visualize_tree() 