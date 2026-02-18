"""Objective functions for linkage mechanism optimization.

Each objective class implements __call__(linkage) -> float,
returning a scalar value to be minimized.
"""

import numpy as np
from .linkage import FourBarLinkage


class PathError:
    """Measures how closely the coupler curve matches a target path.

    The error is the sum of minimum distances from each target point
    to the nearest point on the coupler curve.
    """

    def __init__(self, target_points: np.ndarray, n_curve_points: int = 360):
        """
        Parameters
        ----------
        target_points : (m, 2) array of desired path points
        n_curve_points : number of points to sample on the coupler curve
        """
        self.target_points = np.asarray(target_points)
        self.n_curve_points = n_curve_points

    def __call__(self, linkage: FourBarLinkage, branch: int = 1) -> float:
        curve = linkage.trace_coupler_curve(self.n_curve_points, branch)
        if curve is None:
            return 1e6

        # Remove NaN points
        valid = ~np.isnan(curve[:, 0])
        curve = curve[valid]
        if len(curve) < 10:
            return 1e6

        # For each target point, find the minimum distance to the curve
        total_error = 0.0
        for pt in self.target_points:
            dists = np.linalg.norm(curve - pt, axis=1)
            total_error += np.min(dists)

        return total_error / len(self.target_points)


class TransmissionAngle:
    """Penalizes deviation of transmission angle from the ideal 90 degrees.

    Returns the maximum deviation from 90 degrees over the full cycle.
    A lower value means better force transmission.
    """

    def __init__(self, n_points: int = 360):
        self.n_points = n_points

    def __call__(self, linkage: FourBarLinkage, branch: int = 1) -> float:
        analysis = linkage.full_cycle_analysis(self.n_points, branch=branch)
        ta = analysis["transmission_angles"]
        valid = ~np.isnan(ta)
        if np.sum(valid) < 10:
            return np.pi / 2  # worst case

        # Deviation from ideal (pi/4 = 45 deg means the angle itself is
        # between 45-90, we want it close to pi/2)
        deviations = np.abs(ta[valid] - np.pi / 2)
        return np.max(deviations)


class MechanicalAdvantage:
    """Evaluates mechanical advantage uniformity.

    Returns the variance of the mechanical advantage (omega4/omega2)
    over the cycle. Lower variance = more uniform output motion.
    """

    def __init__(self, n_points: int = 360, omega2: float = 1.0):
        self.n_points = n_points
        self.omega2 = omega2

    def __call__(self, linkage: FourBarLinkage, branch: int = 1) -> float:
        analysis = linkage.full_cycle_analysis(
            self.n_points, self.omega2, branch=branch
        )
        omega4 = analysis["omega4"]
        valid = ~np.isnan(omega4)
        if np.sum(valid) < 10:
            return 1e6

        ma = omega4[valid] / self.omega2
        return np.var(ma)


class CouplerCurveSmoothness:
    """Penalizes sharp changes in the coupler curve direction.

    Returns the maximum curvature of the coupler curve.
    Lower value means a smoother curve.
    """

    def __init__(self, n_points: int = 360):
        self.n_points = n_points

    def __call__(self, linkage: FourBarLinkage, branch: int = 1) -> float:
        curve = linkage.trace_coupler_curve(self.n_points, branch)
        if curve is None:
            return 1e6

        valid = ~np.isnan(curve[:, 0])
        curve = curve[valid]
        if len(curve) < 20:
            return 1e6

        # Compute discrete curvature using finite differences
        dx = np.gradient(curve[:, 0])
        dy = np.gradient(curve[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = (dx**2 + dy**2)**1.5
        denom = np.where(denom < 1e-12, 1e-12, denom)
        curvature = np.abs(dx * ddy - dy * ddx) / denom

        return np.max(curvature)
