"""Leakage protection: the sealed external-test guard and the nearest-neighbour distance used by the
near-duplicate checks.
"""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import numpy as np
from . import errors as ERR


# The sealed token is assembled so this source file itself never needs to be opened by name-matching tools.
SEALED_TOKENS = ("at" + "las",)
SEALED_PATH_TOKENS = SEALED_TOKENS + ("cold-sat", "19990018022")
NEAR_DUP_FRAC = 0.01                      # approved default 6: 1% of the training median NN distance


class LeakageError(ERR.StudyError):
    """A split/leakage/sealed-test rule was violated (fail-closed)."""


# --------------------------------------------------------------------------- #
#  Sealed-test guard
# --------------------------------------------------------------------------- #
def contains_sealed_token(text) -> bool:
    t = str(text).lower()
    return any(tok in t for tok in SEALED_PATH_TOKENS)


def assert_no_sealed(obj, where="input"):
    """Refuse any object whose serialized form mentions the sealed external test."""
    txt = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    if contains_sealed_token(txt):
        raise LeakageError(f"{where}: sealed external-test identifier present (refused)")


@contextlib.contextmanager
def sealed_guard():
    """While active, any open() of a path carrying a sealed token raises, and the in-memory vehicle table cannot
    serve a sealed entry. Installed around every diagnose/compare/verify command and in every worker."""
    from ..physics import environments as LE
    from ..physics import spectra as SP
    real_open, real_io_open = builtins.open, io.open
    real_spl, real_refs = LE.spl_array, SP.reference_environments

    def _check(path):
        if isinstance(path, (str, bytes, os.PathLike)) and contains_sealed_token(os.fsdecode(path)):
            raise LeakageError(f"refused to open sealed external-test file: {os.path.basename(os.fsdecode(path))}")

    def g_open(file, *a, **k):
        _check(file); return real_open(file, *a, **k)

    def g_io_open(file, *a, **k):
        _check(file); return real_io_open(file, *a, **k)

    def g_spl(name, *a, **k):
        if contains_sealed_token(name):
            raise LeakageError("refused: sealed vehicle spectrum requested")
        return real_spl(name, *a, **k)

    def g_refs(*a, **k):
        raise LeakageError("refused: reference-vehicle table is not used by the study")

    builtins.open, io.open, LE.spl_array, SP.reference_environments = g_open, g_io_open, g_spl, g_refs
    try:
        yield
    finally:
        builtins.open, io.open, LE.spl_array, SP.reference_environments = real_open, real_io_open, real_spl, real_refs


def nn_dist(A, B, exclude_self=False):
    """Nearest-neighbour Euclidean distance from each row of A to the rows of B."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        d = np.sqrt(((B - A[i]) ** 2).sum(1))
        if exclude_self:
            d = d[d > 0] if np.any(d > 0) else d
        out[i] = d.min()
    return out
