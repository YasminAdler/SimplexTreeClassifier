import numpy as np
from typing import List, Tuple, Optional

class Triangle:
    def __init__(self, v0: Tuple[float, float], v1: Tuple[float, float], v2: Tuple[float, float]):
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
    
    def __repr__(self):
        return f"Triangle({self.v0}, {self.v1}, {self.v2})"

class SimplexPointLocator:
    def __init__(self, triangles: List[Triangle], tolerance: float = 1e-10):
        self.triangles = triangles
        self.tolerance = tolerance
    
    def compute_barycentric_coordinates(self, point: Tuple[float, float], triangle: Triangle) -> Tuple[float, float, float]:
        P = np.array(point)
        v0, v1, v2 = triangle.v0, triangle.v1, triangle.v2
        A = np.column_stack([v1 - v0, v2 - v0])
        b = P - v0
        try:
            alpha_12 = np.linalg.solve(A, b)
            alpha_1, alpha_2 = alpha_12[0], alpha_12[1]
        except np.linalg.LinAlgError:
            return -1, -1, -1
        alpha_0 = 1 - alpha_1 - alpha_2
        return alpha_0, alpha_1, alpha_2
    
    def point_in_triangle(self, point: Tuple[float, float], triangle: Triangle) -> bool:
        alpha_0, alpha_1, alpha_2 = self.compute_barycentric_coordinates(point, triangle)
        return (0 <= alpha_0 <= 1 and 
                0 <= alpha_1 <= 1 and 
                0 <= alpha_2 <= 1)
    
    def find_containing_simplex(self, point: Tuple[float, float]) -> Optional[Tuple[int, Triangle]]:
        for i, triangle in enumerate(self.triangles):
            if self.point_in_triangle(point, triangle):
                return i, triangle
        return None
    
    def find_containing_simplex_with_coords(self, point: Tuple[float, float]) -> Optional[Tuple[int, Triangle, Tuple[float, float, float]]]:
        for i, triangle in enumerate(self.triangles):
            coords = self.compute_barycentric_coordinates(point, triangle)
            
            if coords[0] == -1:
                continue
            
            if not (0 <= coords[0] <= 1 and 
                    0 <= coords[1] <= 1 and 
                    0 <= coords[2] <= 1):
                continue
            
            coord_sum = sum(coords)
            if abs(coord_sum - 1.0) > self.tolerance:
                continue
            
            return i, triangle, coords
        return None

def example_usage():
    V0 = (0, 0)
    V1 = (4, 0)
    V2 = (0, 4)
    M = ((V1[0] + V2[0]) / 2, (V1[1] + V2[1]) / 2)
    triangle1 = Triangle(V0, V1, M)
    triangle2 = Triangle(V0, M, V2)
    locator = SimplexPointLocator([triangle1, triangle2])
    P = (2, 0.5)
    
    print(f"Testing point P = {P}")
    print(f"Triangle 1: {triangle1}")
    print(f"Triangle 2: {triangle2}")
    print()
    
    for i, triangle in enumerate([triangle1, triangle2], 1):
        coords = locator.compute_barycentric_coordinates(P, triangle)
        alpha_0, alpha_1, alpha_2 = coords
        
        print(f"Triangle {i}:")
        print(f"  Barycentric coordinates: α₀={alpha_0:.3f}, α₁={alpha_1:.3f}, α₂={alpha_2:.3f}")
        print(f"  Sum: {sum(coords):.3f}")
        print(f"  All in [0,1]? {0 <= alpha_0 <= 1 and 0 <= alpha_1 <= 1 and 0 <= alpha_2 <= 1}")
        print(f"  Point inside? {locator.point_in_triangle(P, triangle)}")
        print()
    
    result = locator.find_containing_simplex_with_coords(P)
    if result:
        index, triangle, coords = result
        print(f"Point P = {P} is contained in Triangle {index + 1}")
        print(f"Barycentric coordinates: α₀={coords[0]:.3f}, α₁={coords[1]:.3f}, α₂={coords[2]:.3f}")
        reconstructed = (coords[0] * triangle.v0 + 
                        coords[1] * triangle.v1 + 
                        coords[2] * triangle.v2)
        print(f"Verification: {coords[0]:.3f}*{triangle.v0} + {coords[1]:.3f}*{triangle.v1} + {coords[2]:.3f}*{triangle.v2} = {reconstructed}")
    else:
        print(f"Point P = {P} is not contained in any triangle")

if __name__ == "__main__":
    example_usage() 