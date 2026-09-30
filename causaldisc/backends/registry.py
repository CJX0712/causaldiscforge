"""Backend registry: probe availability and resolve a spec to callables.

Usage:
    registry.resolve_backends(("auto",))            -> all available SOTA backends
    registry.resolve_backends(("cl_pc", "cl_ges"))  -> only those
NOTE: always wrap spec in list() before comparing; a tuple ("auto",) != ["auto"].
"""
from __future__ import annotations

from causaldisc.backends import causallearn_backend, pgmpy_backend, lingam_backend


def _all_backends():
    return {
        "cl_pc": causallearn_backend.pc,
        "cl_ges": causallearn_backend.ges,
        "cl_cdnod": causallearn_backend.cdnod,
        "cl_fci": causallearn_backend.fci,
        "pgmpy_hc": pgmpy_backend.hc,
        "pgmpy_pc": pgmpy_backend.pc,
        "lingam": lingam_backend.direct_lingam,
    }


def available_backends():
    out = {}
    if causallearn_backend.available():
        out.update({k: f for k, f in _all_backends().items() if k.startswith("cl_")})
    if pgmpy_backend.available():
        out.update({k: f for k, f in _all_backends().items() if k.startswith("pgmpy_")})
    if lingam_backend.available():
        out["lingam"] = _all_backends()["lingam"]
    return out


def resolve_backends(spec):
    spec_list = list(spec) if spec else []
    if len(spec_list) == 1 and spec_list[0] == "auto":
        return available_backends()
    av = available_backends()
    return {k: av[k] for k in spec_list if k in av}
