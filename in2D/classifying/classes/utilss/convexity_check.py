import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
in2d_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, in2d_dir)

from in2D.classifying.classes.utilss.plane_equation import PlaneEquation


def get_shared_vertices(simplex1_node, simplex2_node):
    set1 = set(tuple(v) for v in simplex1_node.vertices)
    set2 = set(tuple(v) for v in simplex2_node.vertices)
    shared_tuples = set1.intersection(set2)
    return [np.array(v) for v in shared_tuples]


def find_intersection_on_edge(line_coeffs, point_a, point_b):
    # line_coeffs is [w_0, ..., w_n, b]
    w = line_coeffs[:-1]
    b = line_coeffs[-1]
    
    val_a = np.dot(w, point_a) + b
    val_b = np.dot(w, point_b) + b
    
    if abs(val_a - val_b) < 1e-10:
        return None
    
    t = val_a / (val_a - val_b)
    if t < 0 or t > 1:
        return None
    
    return point_a + t * (point_b - point_a)


def find_point_on_svm_line(simplex_node, line_coeffs, shared_vertices):
    vertices = [np.array(v) for v in simplex_node.vertices]
    shared_set = set(tuple(v) for v in shared_vertices)
    
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            vi_shared = tuple(vertices[i]) in shared_set
            vj_shared = tuple(vertices[j]) in shared_set
            
            # Skip shared edge
            if vi_shared and vj_shared:
                continue
            
            point = find_intersection_on_edge(line_coeffs, vertices[i], vertices[j])
            if point is not None:
                return point
    
    return None


def find_point_on_shared_boundary(shared_vertices, line_coeffs):
    vertices = [np.array(v) for v in shared_vertices]
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            point = find_intersection_on_edge(line_coeffs, vertices[i], vertices[j])
            if point is not None:
                return point
    return None


def check_convexity_between_simplices(simplex1_node, simplex2_node, weights, global_tree=None) -> dict:
    shared_vertices = get_shared_vertices(simplex1_node, simplex2_node)
    if len(shared_vertices) < 2:
        return {'is_convex': True, 'test_point': None}
    
    svm_plane1 = PlaneEquation(simplex1_node)
    svm_plane2 = PlaneEquation(simplex2_node)
    
    line_coeffs1 = svm_plane1.compute_plane_from_weights(weights)
    line_coeffs2 = svm_plane2.compute_plane_from_weights(weights)
    
    meeting_point = find_point_on_shared_boundary(shared_vertices, line_coeffs1)
    
    if meeting_point is None:
        return {'is_convex': True, 'test_point': None}
        
    point_on_line1 = find_point_on_svm_line(simplex1_node, line_coeffs1, shared_vertices)
    point_on_line2 = find_point_on_svm_line(simplex2_node, line_coeffs2, shared_vertices)
    
    if point_on_line1 is None or point_on_line2 is None:
        return {'is_convex': True, 'test_point': None}
    
    average_point = (meeting_point + point_on_line1 + point_on_line2) / 3
    
    containing_simplex = None
    if simplex1_node.point_inside_simplex(tuple(average_point)):
        containing_simplex = simplex1_node
    elif simplex2_node.point_inside_simplex(tuple(average_point)):
        containing_simplex = simplex2_node
    
    if containing_simplex is None and global_tree is not None:
        containing_simplex = global_tree.find_containing_simplex(tuple(average_point))
    
    if containing_simplex is None:
        return {'is_convex': True, 'test_point': average_point}
        
    barycentric_coords = containing_simplex.embed_point(tuple(average_point))
    if barycentric_coords is None:
        return {'is_convex': True, 'test_point': average_point}
        
    vertex_decision_values = weights[containing_simplex.vertex_indices]
    
    value_at_average_point = np.dot(vertex_decision_values, barycentric_coords)
    
    is_inside_red_area = value_at_average_point >= 0
    
    return {'is_convex': is_inside_red_area, 'test_point': average_point}
