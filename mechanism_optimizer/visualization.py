"""Visualization utilities for linkage mechanisms and optimization results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .linkage import FourBarLinkage


def plot_linkage(linkage: FourBarLinkage, theta2: float, branch: int = 1,
                 ax=None, show_coupler_point: bool = True):
    """Plot the linkage at a given crank angle.

    Parameters
    ----------
    linkage : FourBarLinkage
    theta2 : float  Crank angle in radians
    branch : int  Assembly mode
    ax : matplotlib Axes, optional
    show_coupler_point : bool  Whether to highlight the coupler point
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    result = linkage.solve_position(theta2, branch)
    if result is None:
        ax.text(0.5, 0.5, "No valid configuration",
                transform=ax.transAxes, ha="center")
        return ax

    O2 = np.array([0, 0])
    O4 = np.array([linkage.L1, 0])
    A = result["A"]
    B = result["B"]
    P = result["P"]

    # Draw links
    for start, end, color, lw in [
        (O2, A, "#2196F3", 2.5),    # crank (blue)
        (A, B, "#4CAF50", 2.5),     # coupler (green)
        (O4, B, "#FF9800", 2.5),    # follower (orange)
        (O2, O4, "#757575", 1.5),   # ground (gray)
    ]:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=lw, solid_capstyle="round")

    # Draw joints
    for pt, label in [(O2, "$O_2$"), (O4, "$O_4$"), (A, "$A$"), (B, "$B$")]:
        ax.plot(pt[0], pt[1], "ko", markersize=6, zorder=5)
        ax.annotate(label, pt, textcoords="offset points",
                    xytext=(8, 8), fontsize=10)

    # Ground hatching
    ax.plot([O2[0] - 0.3, O4[0] + 0.3], [0, 0], "k-", linewidth=1)
    for x in np.linspace(O2[0] - 0.2, O4[0] + 0.2, 8):
        ax.plot([x, x - 0.15], [0, -0.15], "k-", linewidth=0.5)

    # Fixed pivots
    for pt in [O2, O4]:
        circle = Circle(pt, 0.08, fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(circle)

    if show_coupler_point:
        ax.plot(P[0], P[1], "r*", markersize=12, zorder=5, label="Coupler point")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_coupler_curve(linkage: FourBarLinkage, n_points: int = 360,
                       branch: int = 1, ax=None,
                       target_points: np.ndarray | None = None):
    """Plot the coupler curve traced by the coupler point.

    Parameters
    ----------
    linkage : FourBarLinkage
    n_points : int
    branch : int
    ax : matplotlib Axes, optional
    target_points : (m, 2) array, optional  Target path to overlay
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    curve = linkage.trace_coupler_curve(n_points, branch)
    if curve is not None:
        ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=1.5,
                label="Coupler curve")

    if target_points is not None:
        ax.plot(target_points[:, 0], target_points[:, 1], "r--o",
                markersize=4, linewidth=1, label="Target path")

    # Show fixed pivots
    ax.plot(0, 0, "ks", markersize=8, label="$O_2$")
    ax.plot(linkage.L1, 0, "k^", markersize=8, label="$O_4$")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Coupler Curve")
    return ax


def plot_pareto_front(F: np.ndarray, obj_names: list[str] | None = None,
                      ax=None):
    """Plot the Pareto front from optimization results.

    Parameters
    ----------
    F : (n, m) array of objective values
    obj_names : list of objective names
    ax : matplotlib Axes, optional
    """
    if F.ndim != 2:
        raise ValueError("F must be 2D array")

    n_obj = F.shape[1]
    if obj_names is None:
        obj_names = [f"Objective {i+1}" for i in range(n_obj)]

    if n_obj == 2:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(F[:, 0], F[:, 1], c="#2196F3", s=30, edgecolors="navy",
                   alpha=0.7)
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_title("Pareto Front")
        ax.grid(True, alpha=0.3)
        return ax

    elif n_obj == 3:
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c="#2196F3", s=30,
                   edgecolors="navy", alpha=0.7)
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_zlabel(obj_names[2])
        ax.set_title("Pareto Front")
        return ax

    else:
        # For >3 objectives, use parallel coordinates
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Normalize objectives to [0, 1]
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range < 1e-12] = 1.0
        F_norm = (F - F_min) / F_range

        x = np.arange(n_obj)
        for row in F_norm:
            ax.plot(x, row, alpha=0.3, color="#2196F3")
        ax.set_xticks(x)
        ax.set_xticklabels(obj_names, rotation=30, ha="right")
        ax.set_ylabel("Normalized value")
        ax.set_title("Pareto Front (Parallel Coordinates)")
        ax.grid(True, alpha=0.3)
        return ax


def plot_analysis_summary(linkage: FourBarLinkage, branch: int = 1):
    """Plot a comprehensive analysis summary with 4 subplots.

    Subplots: linkage at 0 deg, coupler curve,
              transmission angle vs theta2, angular velocity ratio vs theta2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Linkage at theta2 = 0
    plot_linkage(linkage, 0, branch, ax=axes[0, 0])
    axes[0, 0].set_title("Linkage at $\\theta_2 = 0$")

    # 2. Coupler curve
    plot_coupler_curve(linkage, branch=branch, ax=axes[0, 1])

    # 3. Full cycle analysis
    analysis = linkage.full_cycle_analysis(branch=branch)
    theta2_deg = np.degrees(analysis["theta2"])

    ta_deg = np.degrees(analysis["transmission_angles"])
    axes[1, 0].plot(theta2_deg, ta_deg, "b-", linewidth=1.5)
    axes[1, 0].axhline(y=45, color="r", linestyle="--", alpha=0.5, label="45°")
    axes[1, 0].axhline(y=90, color="g", linestyle="--", alpha=0.5, label="90° (ideal)")
    axes[1, 0].set_xlabel("$\\theta_2$ (deg)")
    axes[1, 0].set_ylabel("Transmission angle (deg)")
    axes[1, 0].set_title("Transmission Angle")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Angular velocity ratio
    omega4 = analysis["omega4"]
    axes[1, 1].plot(theta2_deg, omega4, "b-", linewidth=1.5)
    axes[1, 1].set_xlabel("$\\theta_2$ (deg)")
    axes[1, 1].set_ylabel("$\\omega_4 / \\omega_2$")
    axes[1, 1].set_title("Angular Velocity Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes
