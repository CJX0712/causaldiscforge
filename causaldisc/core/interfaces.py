"""Abstract interface all discoverers must satisfy."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from causaldisc.core.types import CPDAG


@runtime_checkable
class BaseDiscoverer(Protocol):
    name: str

    def fit(self, X: np.ndarray, **kw) -> CPDAG:
        """Estimate a CPDAG from data X of shape [n, d].

        The returned CPDAG MUST already be normalised via dag_to_cpdag /
        meek rules so that `graph` uses only {NONE, TAIL, ARROW} (PAG backends
        may add CIRCLE).
        """
        ...
