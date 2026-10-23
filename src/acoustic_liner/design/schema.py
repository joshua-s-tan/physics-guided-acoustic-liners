# -*- coding: utf-8 -*-
r"""Case-F fixed-length design vector <-> structured liner design.

MIGRATED verbatim from `fairing_case/design_space.py` (frozen SHA-256 recorded in the historical migration record,
which is not distributed) for the §14 step-8 self-containment migration. Pure NumPy leaf; behaviour is byte-for-byte with the reference
(fixture-verified). Only change: the reference `__main__` demo imported `fairing_oracle` (out of the release
package); it is replaced by a self-contained round-trip check.

Bridges the GENERATOR (cVAE / optimizer, which emit flat vectors) to the ORACLE (`oracle.label_v2`, which wants
a structured design dict). The generator never scores itself; it only emits vectors.

Design vector (19 dims), all continuous, box-bounded:
    [ D, face_t, wall_t,
      f0_1, r_1, L_1, a_1,   f0_2, r_2, L_2, a_2,
      f0_3, r_3, L_3, a_3,   f0_4, r_4, L_4, a_4 ]
where (f0, r=neck radius, L=neck length) tune each of N_GROUPS=4 resonator bands and the four a_* are
UNNORMALISED area logits -> softmax -> area fractions summing to 1 (area-budget satisfied by construction).
Damping Q is a FIXED assumed constant (Q_FIXED), flagged as the model's largest uncertainty; NOT a design var.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

N_GROUPS = 4
Q_FIXED = 15.0            # assumed damping (see physics.resonator caveat)

# (name, lo, hi) -- SI units (m, Hz). Order is the vector order.
_GLOBAL_AXES: List[Tuple[str, float, float]] = [
    ("D",      0.010, 0.050),     # backing depth [m], <= 50 mm envelope
    ("face_t", 0.0005, 0.0020),   # facesheet thickness [m]
    ("wall_t", 0.0004, 0.0015),   # partition wall thickness [m]
]
_GROUP_AXES: List[Tuple[str, float, float]] = [
    ("f0", 50.0, 280.0),          # tuning frequency [Hz] (covers the 50-250 band)
    ("r",  0.0010, 0.0100),       # neck radius [m] (>= 0.5 mm floor; box starts 1 mm)
    ("L",  0.0050, 0.0400),       # neck length [m] (extended necks -> compact low-f)
    ("a", -3.0, 3.0),             # area logit (softmax across groups)
]

DIM = len(_GLOBAL_AXES) + N_GROUPS * len(_GROUP_AXES)   # 3 + 4*4 = 19


def bounds() -> Tuple[np.ndarray, np.ndarray]:
    los, his = [], []
    for _, lo, hi in _GLOBAL_AXES:
        los.append(lo); his.append(hi)
    for _ in range(N_GROUPS):
        for _, lo, hi in _GROUP_AXES:
            los.append(lo); his.append(hi)
    return np.array(los), np.array(his)


LO, HI = bounds()


def clip_to_box(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, float), LO), HI)


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - np.max(v))
    return e / e.sum()


def vector_to_design(x: np.ndarray) -> Dict:
    """Flat 19-vector -> structured design dict for oracle.label_v2."""
    x = clip_to_box(x)
    D, face_t, wall_t = float(x[0]), float(x[1]), float(x[2])
    logits = []
    raw = []
    off = len(_GLOBAL_AXES)
    for g in range(N_GROUPS):
        f0, r, L, a = x[off + 4 * g: off + 4 * g + 4]
        raw.append((float(f0), float(r), float(L)))
        logits.append(float(a))
    fracs = _softmax(np.array(logits))
    groups = []
    for (f0, r, L), frac in zip(raw, fracs):
        groups.append(dict(f0=f0, r_neck=r, L_neck=L,
                           area_frac=float(frac), Q=Q_FIXED))
    return dict(D=D, groups=groups, face_t=face_t, wall_t=wall_t)


def design_to_vector(design: Dict) -> np.ndarray:
    """Structured design -> flat vector (area_frac -> log used as logit)."""
    x = [design["D"], design["face_t"], design["wall_t"]]
    for g in design["groups"]:
        x += [g["f0"], g["r_neck"], g["L_neck"],
              math.log(max(g["area_frac"], 1e-6))]
    return clip_to_box(np.array(x, float))


def sample_random(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample in the box. Sort group tunings ascending for identifiability (breaks the permutation
    symmetry that would confuse a generator)."""
    x = LO + rng.random(DIM) * (HI - LO)
    off = len(_GLOBAL_AXES)
    f0s = sorted(x[off + 4 * g] for g in range(N_GROUPS))
    for g in range(N_GROUPS):
        x[off + 4 * g] = f0s[g]
    return x


def normalize01(x: np.ndarray) -> np.ndarray:
    """Box -> [0,1]^DIM (for the cVAE to train on scale-free inputs)."""
    return (clip_to_box(x) - LO) / (HI - LO)


def denormalize01(u: np.ndarray) -> np.ndarray:
    """[0,1]^DIM -> box."""
    return LO + np.clip(np.asarray(u, float), 0.0, 1.0) * (HI - LO)


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("design", "schema"):
        raise SystemExit(0)
    rng = np.random.default_rng(0)
    print(f"DIM={DIM}  N_GROUPS={N_GROUPS}  Q_FIXED={Q_FIXED}")
    x = sample_random(rng)
    d = vector_to_design(x)
    x2 = design_to_vector(d)
    print("round-trip max |dx| (norm):", float(np.max(np.abs(normalize01(x) - normalize01(x2)))))


# --------------------------------------------------------------------------- #
#  Area gauge-fix + decoder vector (+ its exact inverse, for label recomputation)
# --------------------------------------------------------------------------- #
def _area_u(design):
    a = np.clip(np.array([g["area_frac"] for g in design["groups"]], float), 1e-6, None)
    return np.log(a[:3] / a[3])


def _u_to_fracs(u):
    z = np.array([u[0], u[1], u[2], 0.0]); e = np.exp(z - z.max()); return e / e.sum()


def decoder_vector(design, f_cov):
    v = [design["D"], design["face_t"], design["wall_t"]]
    for g in design["groups"]:
        v += [g["f0"], g["r_neck"], g["L_neck"]]
    v += list(_area_u(design)) + [f_cov]
    return [float(x) for x in v]


def design_from_decoder_vector(dvec):
    """EXACT inverse of `decoder_vector` (area fractions reconstructed via softmax([u1,u2,u3,0])). Returns
    (design, f_cov). Lets validation recompute a label from STORED inputs without trusting the stored label."""
    D, face_t, wall_t = float(dvec[0]), float(dvec[1]), float(dvec[2])
    raw = []
    off = 3
    for g in range(N_GROUPS):
        f0, r, L = dvec[off + 3 * g: off + 3 * g + 3]
        raw.append((float(f0), float(r), float(L)))
    u = dvec[off + 3 * N_GROUPS: off + 3 * N_GROUPS + 3]
    fracs = _u_to_fracs([float(x) for x in u])
    f_cov = float(dvec[-1])
    groups = [dict(f0=f0, r_neck=r, L_neck=L, area_frac=float(fr), Q=Q_FIXED)
              for (f0, r, L), fr in zip(raw, fracs)]
    return dict(D=D, groups=groups, face_t=face_t, wall_t=wall_t), f_cov
