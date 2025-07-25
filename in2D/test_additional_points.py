from simplex_point_location import Triangle, SimplexPointLocator

def test_multiple_points():
    V0 = (0, 0)
    V1 = (4, 0)
    V2 = (0, 4)
    M = (2, 2)
    
    triangle1 = Triangle(V0, V1, M)
    triangle2 = Triangle(V0, M, V2)
    
    locator = SimplexPointLocator([triangle1, triangle2])
    
    test_points = [
        (2, 0.5),
        (0.5, 1.5),
        (1, 1),
        (3, 1),
        (0.5, 3),
        (5, 0),
        (0, 5),
        (-1, 1),
        (2, 2),
        (0, 0),
    ]
    
    print("Testing multiple points:")
    print("=" * 60)
    
    for point in test_points:
        print(f"\nPoint P = {point}")
        result = locator.find_containing_simplex_with_coords(point)
        
        if result:
            index, triangle, coords = result
            print(f"  ✓ Inside Triangle {index + 1}")
            print(f"  Barycentric coordinates: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")
        else:
            print(f"  ✗ Not contained in any triangle")
            
            for i, triangle in enumerate([triangle1, triangle2], 1):
                coords = locator.compute_barycentric_coordinates(point, triangle)
                print(f"    Triangle {i}: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")

def test_edge_cases():
    print("\n\nTesting edge cases:")
    print("=" * 60)
    
    degenerate = Triangle((0, 0), (1, 0), (2, 0))
    normal = Triangle((0, 0), (2, 0), (1, 1))
    
    locator = SimplexPointLocator([degenerate, normal])
    
    test_point = (1, 0.5)
    print(f"\nTesting point {test_point} with degenerate triangle:")
    
    result = locator.find_containing_simplex_with_coords(test_point)
    if result:
        index, triangle, coords = result
        print(f"  ✓ Inside Triangle {index + 1}")
        print(f"  Barycentric coordinates: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")
    else:
        print(f"  ✗ Not contained in any triangle")

if __name__ == "__main__":
    test_multiple_points()
    test_edge_cases() 