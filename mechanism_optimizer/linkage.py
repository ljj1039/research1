"""Four-bar linkage mechanism kinematics analysis.

Convention:
    - Link 1 (L1): Ground link (fixed), from O2 to O4
    - Link 2 (L2): Crank (input), from O2
    - Link 3 (L3): Coupler, connects crank to follower
    - Link 4 (L4): Follower (output), from O4

    O2 is at the origin, O4 is at (L1, 0).
    theta2 is the input crank angle measured from +x axis.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CouplerPoint:
    """Defines a coupler point relative to the coupler link.

    The coupler point P is defined by distance `a` from joint A (crank-coupler)
    and angle `alpha` measured from line AB (coupler link direction).
    """
    a: float  # distance from joint A along coupler
    alpha: float  # angle offset from coupler line (rad)


class FourBarLinkage:
    """Four-bar linkage mechanism with kinematics solver.

    Parameters
    ----------
    L1 : float  Ground link length
    L2 : float  Crank link length
    L3 : float  Coupler link length
    L4 : float  Follower link length
    coupler_point : CouplerPoint, optional
        Coupler point definition. Default is midpoint of coupler.
    """

    def __init__(self, L1: float, L2: float, L3: float, L4: float,
                 coupler_point: CouplerPoint | None = None):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.coupler_point = coupler_point or CouplerPoint(L3 / 2, 0.0)

    @property
    def link_lengths(self) -> np.ndarray:
        return np.array([self.L1, self.L2, self.L3, self.L4])

    def grashof_condition(self) -> str:
        """Check the Grashof condition for the linkage.

        Returns 'grashof' if shortest + longest <= sum of other two,
        'non-grashof' otherwise, 'change-point' if equal.
        """
        links = sorted(self.link_lengths)
        s, p, q, l = links
        diff = s + l - p - q
        if diff < -1e-10:
            return "grashof"
        elif diff > 1e-10:
            return "non-grashof"
        else:
            return "change-point"

    def solve_position(self, theta2: float, branch: int = 1) -> dict | None:
        """Solve position kinematics for given crank angle.

        Parameters
        ----------
        theta2 : float  Crank angle in radians
        branch : int  Assembly mode (+1 or -1)

        Returns
        -------
        dict with keys: theta3, theta4, A, B, P (joint and coupler positions)
        or None if no valid configuration exists.
        """
        L1, L2, L3, L4 = self.L1, self.L2, self.L3, self.L4

        # Joint A position (crank tip)
        Ax = L2 * np.cos(theta2)
        Ay = L2 * np.sin(theta2)

        # Distance from O4 to A
        dx = Ax - L1
        dy = Ay
        d = np.sqrt(dx**2 + dy**2)

        # Check if configuration is possible
        if d > L3 + L4 or d < abs(L3 - L4) or d < 1e-12:
            return None

        # Solve for theta4 using cosine law
        cos_beta = (d**2 + L4**2 - L3**2) / (2 * d * L4)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)

        gamma = np.arctan2(dy, dx)
        theta4 = gamma + branch * beta

        # Joint B position (follower tip = coupler-follower joint)
        Bx = L1 + L4 * np.cos(theta4)
        By = L4 * np.sin(theta4)

        # Solve theta3 from A to B
        theta3 = np.arctan2(By - Ay, Bx - Ax)

        # Coupler point
        cp = self.coupler_point
        Px = Ax + cp.a * np.cos(theta3 + cp.alpha)
        Py = Ay + cp.a * np.sin(theta3 + cp.alpha)

        return {
            "theta3": theta3,
            "theta4": theta4,
            "A": np.array([Ax, Ay]),
            "B": np.array([Bx, By]),
            "P": np.array([Px, Py]),
        }

    def solve_velocity(self, theta2: float, omega2: float,
                       theta3: float, theta4: float) -> dict:
        """Solve velocity kinematics.

        Parameters
        ----------
        theta2 : float  Crank angle (rad)
        omega2 : float  Crank angular velocity (rad/s)
        theta3, theta4 : float  From position solution

        Returns
        -------
        dict with omega3, omega4, VA, VB, VP
        """
        L2, L3, L4 = self.L2, self.L3, self.L4

        # Velocity coefficient matrix
        # L2*omega2*sin(theta2) + L3*omega3*sin(theta3) = L4*omega4*sin(theta4)
        # L2*omega2*cos(theta2) + L3*omega3*cos(theta3) = L4*omega4*cos(theta4)
        A_mat = np.array([
            [L3 * np.sin(theta3), -L4 * np.sin(theta4)],
            [L3 * np.cos(theta3), -L4 * np.cos(theta4)],
        ])
        b_vec = np.array([
            -L2 * omega2 * np.sin(theta2),
            -L2 * omega2 * np.cos(theta2),
        ])

        det = np.linalg.det(A_mat)
        if abs(det) < 1e-12:
            return {"omega3": 0.0, "omega4": 0.0,
                    "VA": np.zeros(2), "VB": np.zeros(2), "VP": np.zeros(2)}

        result = np.linalg.solve(A_mat, b_vec)
        omega3, omega4 = result

        # Joint velocities
        VA = np.array([-L2 * omega2 * np.sin(theta2),
                        L2 * omega2 * np.cos(theta2)])
        VB = np.array([-L4 * omega4 * np.sin(theta4),
                        L4 * omega4 * np.cos(theta4)])

        # Coupler point velocity
        cp = self.coupler_point
        VP = VA + np.array([
            -cp.a * omega3 * np.sin(theta3 + cp.alpha),
             cp.a * omega3 * np.cos(theta3 + cp.alpha),
        ])

        return {
            "omega3": omega3,
            "omega4": omega4,
            "VA": VA,
            "VB": VB,
            "VP": VP,
        }

    def transmission_angle(self, theta2: float, branch: int = 1) -> float | None:
        """Calculate the transmission angle (angle at joint B between L3 and L4).

        The ideal transmission angle is 90 degrees.
        Returns angle in radians, or None if no valid config.
        """
        result = self.solve_position(theta2, branch)
        if result is None:
            return None
        mu = abs(result["theta3"] - result["theta4"])
        # Normalize to [0, pi]
        mu = mu % np.pi
        if mu > np.pi / 2:
            mu = np.pi - mu
        return mu

    def trace_coupler_curve(self, n_points: int = 360,
                            branch: int = 1) -> np.ndarray | None:
        """Trace the coupler curve over a full crank rotation.

        Returns (n, 2) array of coupler point positions, or None for
        points where no valid configuration exists.
        """
        theta2_range = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = []
        for theta2 in theta2_range:
            result = self.solve_position(theta2, branch)
            if result is not None:
                points.append(result["P"])
            else:
                points.append(np.array([np.nan, np.nan]))
        return np.array(points)

    def full_cycle_analysis(self, n_points: int = 360, omega2: float = 1.0,
                            branch: int = 1) -> dict:
        """Perform full-cycle kinematic analysis.

        Returns dict with arrays for theta2, theta3, theta4,
        coupler_positions, transmission_angles, omega3, omega4.
        """
        theta2_arr = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        theta3_arr = np.full(n_points, np.nan)
        theta4_arr = np.full(n_points, np.nan)
        positions = np.full((n_points, 2), np.nan)
        trans_angles = np.full(n_points, np.nan)
        omega3_arr = np.full(n_points, np.nan)
        omega4_arr = np.full(n_points, np.nan)

        for i, theta2 in enumerate(theta2_arr):
            result = self.solve_position(theta2, branch)
            if result is None:
                continue
            theta3_arr[i] = result["theta3"]
            theta4_arr[i] = result["theta4"]
            positions[i] = result["P"]

            mu = self.transmission_angle(theta2, branch)
            if mu is not None:
                trans_angles[i] = mu

            vel = self.solve_velocity(theta2, omega2,
                                      result["theta3"], result["theta4"])
            omega3_arr[i] = vel["omega3"]
            omega4_arr[i] = vel["omega4"]

        return {
            "theta2": theta2_arr,
            "theta3": theta3_arr,
            "theta4": theta4_arr,
            "coupler_positions": positions,
            "transmission_angles": trans_angles,
            "omega3": omega3_arr,
            "omega4": omega4_arr,
        }
