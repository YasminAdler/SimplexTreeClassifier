"""Recover the linear decision boundary inside a single simplex.

Given a linear classifier's per-vertex weights, ``PlaneEquation`` turns them
into the hyperplane (in the original feature space) where the decision function
is zero within one simplex. Used by the convexity checks.
"""

import numpy as np

from .simplex import Simplex


class PlaneEquation:
    def __init__(self, simplex: Simplex):
        self.simplex = simplex
        self.plane_coefficients = None
        self.normalized_coefficients = None

    def compute_plane_from_weights(self, weight_vector: np.ndarray,
                                   intercept: float = 0.0) -> np.ndarray:
        # Convert sparse matrix to dense array if needed (OneClassSVM returns sparse).
        if hasattr(weight_vector, 'toarray'):
            weight_vector = weight_vector.toarray().flatten()
        elif hasattr(weight_vector, 'A'):
            weight_vector = np.asarray(weight_vector).flatten()
        simplex_weights = np.asarray(weight_vector)[self.simplex.vertex_indices]
        plane_eq = self.simplex.A_inv.T @ (simplex_weights[1:] - simplex_weights[0])
        constant = (simplex_weights[0] + intercept) - plane_eq @ self.simplex.vertices[0]
        self.plane_coefficients = np.append(plane_eq, constant)
        return self.plane_coefficients

    def get_cartesian_form(self) -> str:
        if self.plane_coefficients is None:
            raise ValueError(
                "Plane equation not computed yet. Call compute_plane_from_weights first."
            )

        coeffs = self.plane_coefficients
        var_names = [f"x{i+1}" for i in range(len(coeffs) - 1)]
        constant = coeffs[-1]

        equation_parts = []
        for coeff, var in zip(coeffs[:-1], var_names):
            if abs(coeff) < 1e-10:
                continue
            if coeff == 1.0:
                term = var
            elif coeff == -1.0:
                term = f"-{var}"
            else:
                term = f"{coeff:.4f}{var}"

            if equation_parts and coeff > 0:
                equation_parts.append("+")
            equation_parts.append(term)

        if abs(constant) > 1e-10:
            if constant > 0 and equation_parts:
                equation_parts.append("+")
            equation_parts.append(f"{constant:.4f}")

        if not equation_parts:
            return "0 = 0"

        return " ".join(equation_parts) + " = 0"
