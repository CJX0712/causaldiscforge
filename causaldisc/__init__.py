"""CausalDiscForge — Causal Discovery (Causal Structure Learning).

Learn the causal graph (DAG / CPDAG) from observational data.

NOTE: this is NOT causal *inference* (estimating treatment effects of a
given variable, which is the domain of the sibling repo ``causalforge``).
CausalDiscForge learns the *graph structure* itself.

Author: 晨星 (CJX0712)
"""

__version__ = "1.0.0"
__author__ = "晨星"

from causaldisc.core.types import CPDAG, NONE, TAIL, ARROW, CIRCLE
from causaldisc.core.config import Config

__all__ = ["CPDAG", "NONE", "TAIL", "ARROW", "CIRCLE", "Config", "__version__"]
