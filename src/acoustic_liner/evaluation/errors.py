"""The study's base error: every split, leakage or evaluation rule fails closed by raising it.
"""
from __future__ import annotations



class StudyError(RuntimeError):
    """Fail-closed study error (accounting, provenance, leakage, or incomplete result)."""
