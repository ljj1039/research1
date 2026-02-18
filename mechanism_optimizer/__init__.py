"""Linkage mechanism multi-objective optimization tool."""

from .linkage import FourBarLinkage
from .objectives import PathError, TransmissionAngle, MechanicalAdvantage
from .optimizer import LinkageOptimizer
from .visualization import plot_linkage, plot_coupler_curve, plot_pareto_front
