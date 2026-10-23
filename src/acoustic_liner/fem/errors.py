"""The error raised for invalid FEM input: a bad frequency, mesh or boundary specification.
"""
from __future__ import annotations



class FEMSolverError(RuntimeError):
    """Solver input/geometry error (bad frequency, mesh, or boundary specification)."""
