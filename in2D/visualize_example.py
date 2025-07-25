import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simplex_point_location import Triangle, SimplexPointLocator

def visualize_example():
    V0 = (0, 0)
    V1 = (4, 0)
    V2 = (0, 4)
    M = (2, 2)
    
    triangle1 = Triangle(V0, V1, M)
    triangle2 = Triangle(V0, M, V2)
    
    locator = SimplexPointLocator([triangle1, triangle2])
    
    fig, ax = plt.subplots(1, 1)
    
    triangle1_patch = patches.Polygon([V0, V1, M], alpha=0.3, color='blue', label='Triangle 1')
    triangle2_patch = patches.Polygon([V0, M, V2], alpha=0.3, color='red', label='Triangle 2')
    
    ax.add_patch(triangle1_patch)
    ax.add_patch(triangle2_patch)
    
    ax.plot([V0[0], V1[0]], [V0[1], V1[1]], 'k-', linewidth=2)
    ax.plot([V1[0], M[0]], [V1[1], M[1]], 'k-', linewidth=2)
    ax.plot([M[0], V0[0]], [M[1], V0[1]], 'k-', linewidth=2)
    ax.plot([V0[0], M[0]], [V0[1], M[1]], 'k-', linewidth=2)
    ax.plot([M[0], V2[0]], [M[1], V2[1]], 'k-', linewidth=2)
    ax.plot([V2[0], V0[0]], [V2[1], V0[1]], 'k-', linewidth=2)
    
    ax.plot(*V0, 'ko', ms=8)
    ax.plot(*V1, 'ko', ms=8)
    ax.plot(*V2, 'ko', ms=8)
    ax.plot(*M, 'ko', ms=8)
    
    ax.text(V0[0]-0.2, V0[1]-0.2, 'V₀(0,0)', size=12, ha='center')
    ax.text(V1[0]+0.2, V1[1]-0.2, 'V₁(4,0)', size=12, ha='center')
    ax.text(V2[0]-0.2, V2[1]+0.2, 'V₂(0,4)', size=12, ha='center')
    ax.text(M[0]+0.2, M[1]+0.2, 'M(2,2)', size=12, ha='center')
    
    test_points = [
        (2, 0.5),
        (0.5, 1.5),
        (1, 1),
        (3, 1),
        (0.5, 3),
        (5, 0),
        (0, 5),
    ]
    
    colors = ['green', 'orange', 'purple', 'cyan', 'yellow', 'gray', 'pink']
    
    for i, point in enumerate(test_points):
        result = locator.find_containing_simplex_with_coords(point)
        
        if result:
            index, triangle, coords = result
            color = colors[i % len(colors)]
            marker = 'o' if index == 0 else 's'
            ax.plot(*point, marker, color=color, ms=10, 
                   label=f'P{i+1}{point} ∈ Δ{index+1}')
            
            ax.text(point[0]+0.1, point[1]+0.1, 
                   f'({coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f})',
                   size=8, color=color)
        else:
            ax.plot(*point, 'x', color='red', ms=12, 
                   label=f'P{i+1}{point} ∉ any Δ')
    
    ax.axis([-1, 6, -1, 5])
    ax.set_title('Simplex Point Location Example\n(Numbers show barycentric coordinates)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        visualize_example()
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        print("Skipping visualization...") 