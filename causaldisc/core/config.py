"""Global configuration with ENV_CAUSALDISC_* overrides and sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None and v.strip() else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None and v.strip() else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    random_seed: int = field(default_factory=lambda: _env_int("CAUSALDISC_SEED", 42))
    n_jobs: int = field(default_factory=lambda: _env_int("CAUSALDISC_NJOBS", 1))  # Windows sandbox: keep 1
    pc_alpha: float = field(default_factory=lambda: _env_float("CAUSALDISC_PC_ALPHA", 0.01))
    pc_max_level: int = field(default_factory=lambda: _env_int("CAUSALDISC_PC_MAX_LEVEL", 3))
    hc_restarts: int = field(default_factory=lambda: _env_int("CAUSALDISC_HC_RESTARTS", 3))
    hc_max_parents: int = field(default_factory=lambda: _env_int("CAUSALDISC_HC_MAX_PARENTS", 3))
    notears_lambdas: tuple = (0.1, 0.2)
    notears_w_thresholds: tuple = (0.3,)
    consensus_bootstraps: int = field(default_factory=lambda: _env_int("CAUSALDISC_BOOTSTRAPS", 30))
    consensus_tau: float = field(default_factory=lambda: _env_float("CAUSALDISC_TAU", 0.6))
    consensus_allow_lingam: bool = field(default_factory=lambda: _env_bool("CAUSALDISC_USE_LINGAM", False))
    hpo_trials: int = field(default_factory=lambda: _env_int("CAUSALDISC_HPO_TRIALS", 20))
    verbose: bool = field(default_factory=lambda: _env_bool("CAUSALDISC_VERBOSE", False))
    # scipy was pinned <=1.16.3 by lingam; keep numpy/scipy-pure code within that API.
    numpy_tolerance: float = 1e-12

    def replace(self, **kw) -> "Config":
        d = dict(self.__dict__)
        d.update(kw)
        return Config(**d)
