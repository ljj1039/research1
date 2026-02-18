"""Example: Multi-objective optimization of a four-bar linkage.

This example optimizes a four-bar linkage to:
1. Minimize path error (coupler curve vs. target circle)
2. Minimize transmission angle deviation (maximize force transmission)

Target: circular coupler curve centered at (3, 4) with radius 1.5
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from mechanism_optimizer.linkage import FourBarLinkage, CouplerPoint
from mechanism_optimizer.objectives import PathError, TransmissionAngle
from mechanism_optimizer.optimizer import LinkageOptimizer
from mechanism_optimizer.visualization import (
    plot_coupler_curve, plot_pareto_front, plot_analysis_summary,
)


def generate_circle_target(cx: float, cy: float, r: float,
                           n: int = 36) -> np.ndarray:
    """Generate target points along a circle."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return np.column_stack([x, y])


def main():
    # --- 1. Define target path ---
    target = generate_circle_target(cx=3.0, cy=4.0, r=1.5, n=36)
    print(f"Target path: circle at (3, 4) with radius 1.5, {len(target)} points")

    # --- 2. Define objectives ---
    obj_path = PathError(target, n_curve_points=360)
    obj_trans = TransmissionAngle(n_points=360)

    objectives = [obj_path, obj_trans]
    obj_names = ["Path Error", "Transmission Angle Deviation"]

    # --- 3. Set design variable bounds ---
    bounds = {
        "L_min": 1.0,
        "L_max": 10.0,
        "a_min": 0.5,
        "a_max": 8.0,
        "alpha_min": -np.pi,
        "alpha_max": np.pi,
    }

    # --- 4. Run optimization ---
    optimizer = LinkageOptimizer(
        objectives=objectives,
        bounds=bounds,
        require_grashof=True,
        require_full_rotation=True,
        branch=1,
    )

    print("Running NSGA-II optimization...")
    result = optimizer.run(pop_size=80, n_gen=100, seed=42, verbose=False)

    n_solutions = len(result["linkages"])
    print(f"Found {n_solutions} Pareto-optimal solutions")

    if n_solutions == 0:
        print("No feasible solutions found. Try adjusting bounds or constraints.")
        return

    # --- 5. Print top solutions ---
    F = result["F"]
    # Sort by path error
    sort_idx = np.argsort(F[:, 0])
    print("\nTop 5 solutions (sorted by path error):")
    print(f"{'#':>3} {'Path Error':>12} {'Trans. Angle Dev':>18} "
          f"{'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6}")
    print("-" * 70)
    for rank, i in enumerate(sort_idx[:5]):
        x = result["X"][i]
        f = F[i]
        print(f"{rank+1:>3} {f[0]:>12.4f} {np.degrees(f[1]):>15.2f}° "
              f"{x[0]:>6.2f} {x[1]:>6.2f} {x[2]:>6.2f} {x[3]:>6.2f}")

    # --- 6. Visualize results ---
    # Pareto front
    fig_pf, ax_pf = plt.subplots(figsize=(8, 6))
    plot_pareto_front(F, obj_names, ax=ax_pf)
    fig_pf.savefig("pareto_front.png", dpi=150, bbox_inches="tight")
    print("\nSaved: pareto_front.png")

    # Best path solution - coupler curve
    best_idx = sort_idx[0]
    best_linkage = result["linkages"][best_idx]
    print(f"\nBest path solution: L1={best_linkage.L1:.2f}, L2={best_linkage.L2:.2f}, "
          f"L3={best_linkage.L3:.2f}, L4={best_linkage.L4:.2f}")
    print(f"  Grashof: {best_linkage.grashof_condition()}")

    fig_cc, ax_cc = plt.subplots(figsize=(8, 6))
    plot_coupler_curve(best_linkage, target_points=target, ax=ax_cc)
    ax_cc.set_title("Best Path Solution - Coupler Curve vs Target")
    fig_cc.savefig("coupler_curve_best.png", dpi=150, bbox_inches="tight")
    print("Saved: coupler_curve_best.png")

    # Analysis summary for best solution
    fig_summary, _ = plot_analysis_summary(best_linkage)
    fig_summary.savefig("analysis_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: analysis_summary.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
