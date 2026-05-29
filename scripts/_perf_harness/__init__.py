"""Internal package for scripts/gen_perf_harnesses.py.

Re-exports the small surface the entry-point script consumes.
"""
from .core import CATALOG, GEN_SENTINEL, PERF_DIR, TYPES, routine_shape
from .dispatch import emit_routine

__all__ = ["CATALOG", "GEN_SENTINEL", "PERF_DIR", "TYPES", "emit_routine", "routine_shape"]
