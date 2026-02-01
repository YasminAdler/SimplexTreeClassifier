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


def find_crossing_on_edge(line_coeffs, point_a, point_b):
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


def find_crossing_on_shared_face(shared_vertices, line_coeffs):
    for i in range(len(shared_vertices)):
        for j in range(i + 1, len(shared_vertices)):
            point_a = np.array(shared_vertices[i])
            point_b = np.array(shared_vertices[j])
            crossing = find_crossing_on_edge(line_coeffs, point_a, point_b)
            if crossing is not None:
                return crossing
    return None


def find_svm_meeting_point(simplex1_node, simplex2_node, weights, intercept):
    shared_vertices = get_shared_vertices(simplex1_node, simplex2_node)
    
    if len(shared_vertices) < 2:
        return None
    
    svm_plane1 = PlaneEquation(simplex1_node)
    line_coeffs1 = svm_plane1.compute_plane_from_weights(weights, intercept)
    
    meeting_point = find_crossing_on_shared_face(shared_vertices, line_coeffs1)
    
    return meeting_point


def find_external_crossing(simplex_node, line_coeffs, shared_vertices):
    vertices = [np.array(v) for v in simplex_node.vertices]
    shared_set = set(tuple(v) for v in shared_vertices)
    
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            vi_shared = tuple(vertices[i]) in shared_set
            vj_shared = tuple(vertices[j]) in shared_set
            
            if vi_shared and vj_shared:
                continue
            
            crossing = find_crossing_on_edge(line_coeffs, vertices[i], vertices[j])
            if crossing is not None:
                return crossing
    
    return None


def find_epsilon_points(simplex1_node, simplex2_node, weights, intercept, epsilon):
    """
    Find test points along the SVM boundary for convexity checking.
    
    Args:
        simplex1_node: First adjacent simplex
        simplex2_node: Second adjacent simplex  
        weights: SVM hyperplane weights
        intercept: SVM hyperplane intercept
        epsilon: Fraction (0-1) of distance from meeting point to external crossing.
                 e.g., 0.3 = 30% of the way toward the external crossing.
    
    Returns:
        meeting_point: Where SVM crosses the shared edge
        point1: Test point in simplex1 direction
        point2: Test point in simplex2 direction
    """
    shared_vertices = get_shared_vertices(simplex1_node, simplex2_node)
    
    if len(shared_vertices) < 2:
        return None, None, None
    
    svm_plane1 = PlaneEquation(simplex1_node)
    svm_plane2 = PlaneEquation(simplex2_node)
    
    line_coeffs1 = svm_plane1.compute_plane_from_weights(weights, intercept)
    line_coeffs2 = svm_plane2.compute_plane_from_weights(weights, intercept)
    
    meeting_point = find_crossing_on_shared_face(shared_vertices, line_coeffs1)
    
    if meeting_point is None:
        return None, None, None
    
    external1 = find_external_crossing(simplex1_node, line_coeffs1, shared_vertices)
    external2 = find_external_crossing(simplex2_node, line_coeffs2, shared_vertices)
    
    if external1 is None or external2 is None:
        return meeting_point, None, None
    
    dir1 = external1 - meeting_point
    dir2 = external2 - meeting_point
    
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return meeting_point, None, None
    
    # Move epsilon fraction (0-1) along the direction toward external crossing
    # epsilon=0.3 means 30% of the way from meeting point to external crossing
    point1 = meeting_point + epsilon * dir1
    point2 = meeting_point + epsilon * dir2
    
    return meeting_point, point1, point2


def find_average_point(simplex1_node, simplex2_node, weights, intercept, epsilon):
    meeting_point, point1, point2 = find_epsilon_points(
        simplex1_node, simplex2_node, weights, intercept, epsilon
    )
    
    if meeting_point is None or point1 is None or point2 is None:
        return None, meeting_point, point1, point2
    
    average_point = (meeting_point + point1 + point2) / 3
    
    return average_point, meeting_point, point1, point2


def is_point_in_red_area(point, containing_simplex, weights, intercept):
    barycentric = containing_simplex._embed_point(tuple(point))
    if barycentric is None:
        return None
    
    vertex_decisions = [weights[idx] + intercept for idx in containing_simplex.vertex_indices]
    
    decision_value = np.dot(vertex_decisions, barycentric)
    
    return decision_value >= 0


def check_convexity(simplex1_node, simplex2_node, weights, intercept, global_tree=None, epsilon=0.3):
    """
    Check if the SVM boundary between two adjacent simplices is convex.
    
    Creates test points along the SVM boundary and checks if the average point
    falls on the expected side of the hyperplane.
    
    Args:
        simplex1_node: First adjacent simplex
        simplex2_node: Second adjacent simplex
        weights: SVM hyperplane weights
        intercept: SVM hyperplane intercept
        global_tree: Optional tree to search for containing simplex if average
                     point falls outside both input simplices
        epsilon: Fraction (0-1) of distance from meeting point to external crossing.
                 Default 0.3 = 30% of the way. Higher values sample further out.
    
    Returns:
        is_convex: True if convex (or undetermined), False if non-convex
        average_point: The test point used for checking
        meeting: Where SVM crosses the shared edge
        pt1, pt2: The epsilon test points in each simplex direction
    """
    average_point, meeting, pt1, pt2 = find_average_point(
        simplex1_node, simplex2_node, weights, intercept, epsilon
    )
    
    if average_point is None:
        return True, None, meeting, pt1, pt2
    
    containing_simplex = None
    
    if simplex1_node._point_inside_simplex(tuple(average_point)):
        containing_simplex = simplex1_node
    elif simplex2_node._point_inside_simplex(tuple(average_point)):
        containing_simplex = simplex2_node
    elif global_tree is not None:
        containing_simplex = global_tree.find_containing_simplex(tuple(average_point))
    
    if containing_simplex is None:
        return True, average_point, meeting, pt1, pt2
    
    in_red = is_point_in_red_area(average_point, containing_simplex, weights, intercept)
    
    if in_red is None:
        return True, average_point, meeting, pt1, pt2
    
    return in_red, average_point, meeting, pt1, pt2
