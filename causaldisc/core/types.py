"""Core data types for causal discovery.

Edge endpoint codes follow the causal-learn convention so that backend
adapters can be translated with zero transformation:

    NONE   (0): no edge
    TAIL   (-1): tail endpoint  ( o--  , an undirected endpoint)
    ARROW  (1): arrowhead      ( -->  )
    CIRCLE (2): circle         ( o-o  , PAG only)

For a directed edge i -> j the matrix stores G[i][j]=ARROW, G[j][i]=TAIL.
For an undirected edge i -- j the matrix stores G[i][j]=TAIL, G[j][i]=TAIL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NONE = 0
TAIL = -1
ARROW = 1
CIRCLE = 2


@dataclass
class CPDAG:
    """A CPDAG (or PAG) represented as an endpoint-code matrix.

    ``graph`` is a [d, d] int8 matrix using the endpoint codes above.
    For a CPDAG only {NONE, TAIL, ARROW} appear; PAG outputs may use CIRCLE.
    """

    d: int
    graph: np.ndarray
    edge_freq: Optional[np.ndarray] = None  # [d,d] confidence in [0,1]
    name: str = "cpdag"
    pag: bool = False  # True when circle endpoints (latent/hidden-confounding) present

    def __post_init__(self) -> None:
        self.graph = np.asarray(self.graph, dtype=np.int8).reshape(self.d, self.d)
        if self.edge_freq is not None:
            self.edge_freq = np.asarray(self.edge_freq, dtype=float).reshape(self.d, self.d)

    def copy(self) -> "CPDAG":
        return CPDAG(
            self.d,
            self.graph.copy(),
            None if self.edge_freq is None else self.edge_freq.copy(),
            self.name,
            self.pag,
        )

    @property
    def adj(self) -> np.ndarray:
        """Undirected adjacency: 1 if any edge (of any orientation) connects i,j."""
        return (np.abs(self.graph) > 0).astype(int)

    @property
    def skeleton_mask(self) -> np.ndarray:
        return np.abs(self.graph) > 0

    def edge_list(self):
        return [(i, j) for i in range(self.d) for j in range(i + 1, self.d)
                if self.graph[i, j] != NONE or self.graph[j, i] != NONE]

    def n_edges(self) -> int:
        return int(np.sum(np.abs(self.graph) > 0) // 2)


@dataclass
class DiscoveryResult:
    """Bundle of an estimated CPDAG and optional metadata returned by discoverers."""

    cpdag: CPDAG
    method: str
    runtime_s: float = 0.0
    bic: Optional[float] = None
    extra: dict = field(default_factory=dict)
