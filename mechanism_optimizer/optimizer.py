"""Multi-objective optimization for linkage mechanisms using NSGA-II."""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .linkage import FourBarLinkage, CouplerPoint


class LinkageProblem(Problem):
    """pymoo Problem definition for linkage optimization.

    Design variables: [L1, L2, L3, L4, coupler_a, coupler_alpha]
    - L1..L4: link lengths
    - coupler_a: coupler point distance from joint A
    - coupler_alpha: coupler point angle offset (rad)

    Constraints:
    - Grashof condition (optional)
    - Full rotatability of crank
    """

    def __init__(self, objectives, bounds,
                 require_grashof: bool = True,
                 require_full_rotation: bool = True,
                 branch: int = 1):
        """
        Parameters
        ----------
        objectives : list of callable
            Objective functions, each takes (FourBarLinkage, branch) -> float
        bounds : dict
            Keys: 'L_min', 'L_max' for link lengths,
                  'a_min', 'a_max' for coupler distance,
                  'alpha_min', 'alpha_max' for coupler angle
        require_grashof : bool
            If True, penalize non-Grashof linkages
        require_full_rotation : bool
            If True, penalize linkages where crank can't fully rotate
        branch : int
            Assembly mode (+1 or -1)
        """
        self.objectives = objectives
        self.require_grashof = require_grashof
        self.require_full_rotation = require_full_rotation
        self.branch = branch

        L_min = bounds.get("L_min", 1.0)
        L_max = bounds.get("L_max", 10.0)
        a_min = bounds.get("a_min", 0.1)
        a_max = bounds.get("a_max", L_max)
        alpha_min = bounds.get("alpha_min", -np.pi)
        alpha_max = bounds.get("alpha_max", np.pi)

        xl = np.array([L_min, L_min, L_min, L_min, a_min, alpha_min])
        xu = np.array([L_max, L_max, L_max, L_max, a_max, alpha_max])

        n_constraints = 0
        if require_grashof:
            n_constraints += 1
        if require_full_rotation:
            n_constraints += 1

        super().__init__(
            n_var=6,
            n_obj=len(objectives),
            n_ieq_constr=n_constraints,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        G = np.zeros((X.shape[0], self.n_ieq_constr))

        for i, x in enumerate(X):
            L1, L2, L3, L4, cp_a, cp_alpha = x
            cp = CouplerPoint(a=cp_a, alpha=cp_alpha)
            linkage = FourBarLinkage(L1, L2, L3, L4, cp)

            # Evaluate objectives
            for j, obj_fn in enumerate(self.objectives):
                F[i, j] = obj_fn(linkage, self.branch)

            # Constraints (g <= 0 means feasible)
            g_idx = 0
            if self.require_grashof:
                links = sorted([L1, L2, L3, L4])
                # Grashof: s + l <= p + q  =>  s + l - p - q <= 0
                G[i, g_idx] = links[0] + links[3] - links[1] - links[2]
                g_idx += 1

            if self.require_full_rotation:
                # Check if crank can fully rotate by sampling
                n_check = 36
                angles = np.linspace(0, 2 * np.pi, n_check, endpoint=False)
                fail_count = sum(
                    1 for a in angles
                    if linkage.solve_position(a, self.branch) is None
                )
                G[i, g_idx] = fail_count  # 0 = feasible
                g_idx += 1

        out["F"] = F
        if self.n_ieq_constr > 0:
            out["G"] = G


class LinkageOptimizer:
    """High-level optimizer for four-bar linkage mechanisms.

    Uses NSGA-II for multi-objective optimization.
    """

    def __init__(self, objectives: list, bounds: dict,
                 require_grashof: bool = True,
                 require_full_rotation: bool = True,
                 branch: int = 1):
        self.problem = LinkageProblem(
            objectives, bounds,
            require_grashof=require_grashof,
            require_full_rotation=require_full_rotation,
            branch=branch,
        )

    def run(self, pop_size: int = 100, n_gen: int = 200,
            seed: int | None = None, verbose: bool = True) -> dict:
        """Run the NSGA-II optimization.

        Parameters
        ----------
        pop_size : int  Population size
        n_gen : int  Number of generations
        seed : int, optional  Random seed for reproducibility
        verbose : bool  Print progress

        Returns
        -------
        dict with keys:
            'X': design variables of Pareto-optimal solutions
            'F': objective values of Pareto-optimal solutions
            'linkages': list of FourBarLinkage objects on Pareto front
            'result': raw pymoo result object
        """
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", n_gen)

        result = minimize(
            self.problem,
            algorithm,
            termination,
            seed=seed,
            verbose=verbose,
        )

        # Build linkage objects for Pareto front
        linkages = []
        if result.X is not None:
            X = result.X if result.X.ndim == 2 else result.X.reshape(1, -1)
            for x in X:
                L1, L2, L3, L4, cp_a, cp_alpha = x
                cp = CouplerPoint(a=cp_a, alpha=cp_alpha)
                linkages.append(FourBarLinkage(L1, L2, L3, L4, cp))

            return {
                "X": X,
                "F": result.F if result.F.ndim == 2 else result.F.reshape(1, -1),
                "linkages": linkages,
                "result": result,
            }

        return {
            "X": np.array([]),
            "F": np.array([]),
            "linkages": [],
            "result": result,
        }
