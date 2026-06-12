"""Internal package for the dual-link perf harness (scripts/gen_dual_harnesses.py).

Re-exports the shared catalog/shape surface the generator consumes.
"""
from .core import CATALOG, routine_shape

__all__ = ["CATALOG", "routine_shape"]
