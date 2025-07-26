# GitHub Visualization Test

This file tests GitHub's ability to visualize different types of content.

## 🖼️ Test 1: Standard Image

![Test Image](images/test.png "This is a test image")

## 📊 Test 2: Mermaid Class Diagram

```mermaid
classDiagram
    class Simplex {
        +vertices: List[np.ndarray]
        +dimension: int
        +point_inside_simplex(point)
        +embed_point(point)
    }
    
    class SimplexTree {
        +children: List[SimplexTree]
        +depth: int
        +add_child(vertices)
        +find_containing_simplex(point)
    }
    
    Simplex <|-- SimplexTree
```

## 🔄 Test 3: Mermaid Flowchart

```mermaid
flowchart TD
    A[Start: Point P] --> B{Point in simplex?}
    B -->|No| C[Return None]
    B -->|Yes| D{Is leaf?}
    D -->|Yes| E[Return simplex]
    D -->|No| F[Check children]
    F --> G{Point in child?}
    G -->|No| F
    G -->|Yes| H[Recurse]
    H --> D
```

## 📈 Test 4: Mermaid Graph

```mermaid
graph TD
    Root[Root SimplexTree] --> Child1[Child 1]
    Root --> Child2[Child 2]
    Root --> Child3[Child 3]
    Root --> Child4[Child 4]
    
    style Root fill:#ff9999
    style Child1 fill:#99ccff
    style Child2 fill:#99ff99
    style Child3 fill:#ffcc99
    style Child4 fill:#cc99ff
```

## 📋 Test 5: Table

| Feature | 2D | 3D | Tree |
|---------|----|----|----|
| Point Location | ✅ | ✅ | ✅ |
| Barycentric Coords | ✅ | ✅ | ✅ |
| Tree Structure | ❌ | ✅ | ✅ |
| Visualization | 2D | 3D | N/A |

## 🎯 Test 6: Code Block

```python
# Test code block
from in3D.classes.simplexTree import SimplexTree

vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
tree = SimplexTree(vertices)
print(f"Tree created with {tree.get_node_count()} nodes")
```

## ✅ Test Results

If you can see:
- [x] The mermaid diagrams rendered as visual diagrams
- [x] The table formatted properly
- [x] The code block with syntax highlighting
- [x] The image placeholder (if you add an actual image)

Then GitHub visualization is working perfectly!

## 🚀 Next Steps

1. Add actual images to the `docs/images/` directory
2. Replace the image placeholder with real screenshots
3. Create more detailed mermaid diagrams
4. Test on GitHub.com to see the final result

GitHub will render all of these elements beautifully when you push to your repository! 