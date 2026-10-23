"""Canonical form of a liner design (banks sorted by f0) and the duplicate key used for exact budget
accounting: the 18-D normalized geometry vector, rounded.
"""
from __future__ import annotations

import numpy as np
from . import schema as DS


KEY_DECIMALS = 6


# --------------------------------------------------------------------------- #
#  Geometry canonicalization + duplicate key
# --------------------------------------------------------------------------- #
def canonical_design(design):
    """Groups sorted by ascending f0 (the permutation-invariant canonical order the cVAE is trained on)."""
    groups = sorted((dict(g) for g in design["groups"]), key=lambda g: g["f0"])
    for g in groups:
        g.pop("cavity_volume", None)
        g.setdefault("Q", DS.Q_FIXED)
    return dict(D=float(design["D"]), face_t=float(design["face_t"]), wall_t=float(design["wall_t"]),
                groups=groups)


def geometry_vector(design):
    """18-D normalized canonical GEOMETRY vector (decoder layout without coverage): box-normalized D/face/wall and
    per-bank f0/r/L, plus the 3 area log-ratios. Used for duplicate keys and all geometry distances."""
    d = canonical_design(design)
    x = np.asarray(DS.decoder_vector(d, 0.0)[:-1], float)[None, :]
    z = _normalize_targets(np.concatenate([x, np.zeros((1, 1))], 1))[0, :-1]
    return z


def geometry_key(design):
    return tuple(np.round(geometry_vector(design), KEY_DECIMALS).tolist())


def key_hash(key):
    """Compact 16-hex digest of a geometry key (stored in traces; equality <=> identical key)."""
    import hashlib
    return hashlib.sha256(repr(tuple(key)).encode()).hexdigest()[:16]


# continuous decoder coordinates (index in the 19-D vector) and their box bounds
_CONT = [(0, 0.01, 0.05), (1, 0.0005, 0.002), (2, 0.0004, 0.0015)]          # D, face_t, wall_t
for _g in range(4):
    b = 3 + 3 * _g
    _CONT += [(b + 1, 0.001, 0.01), (b + 2, 0.005, 0.04)]                    # r, L (f0 handled apart)
F0_IDX = [3 + 3 * g for g in range(4)]                                       # f0 positions
F0_LO, F0_HI = 50.0, 280.0


def _normalize_targets(X):
    Z = X.copy()
    for idx, lo, hi in _CONT:
        Z[:, idx] = (X[:, idx] - lo) / (hi - lo)
    for idx in F0_IDX:
        Z[:, idx] = (X[:, idx] - F0_LO) / (F0_HI - F0_LO)
    return Z
