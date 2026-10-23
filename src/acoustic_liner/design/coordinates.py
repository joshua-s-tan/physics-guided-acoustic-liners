# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 feasibility-by-construction design coordinates (pre-registration sec. 4).

A deterministic, differentiable BIJECTION between an unconstrained coordinate vector u in R^18 and the canonical
GEOMETRY domain (coverage is NOT a generator coordinate -- `physics.quality.quality_eval` solves coverage internally
under the mass budget, so it is optimized inside the training objective and recomputed canonically at evaluation):

    DOMAIN = { geometry representable EXACTLY by the locked plain quality-CMA (design.schema layout: four area logits
               in [-3, 3]  <=>  pairwise area log-ratio span <= 6) }
             ∩ { canonical CMA/design box }  ∩ { canonical geometric feasibility (feasibility.check_feasibility) }
             ∩ { banks strictly ordered by f0 }

Why an EXACT construction is possible. With S = pi r^2, L_eff = L + 2*0.85*r and V = S / (L_eff k^2), k = 2 pi f0 / c
(physics.resonator), the open-area ratio is sigma = S / (V / D) = D * L_eff * k^2 -- independent of r except through the
end correction. Inside the box the ONLY binding geometric constraints are
    (a) sigma >= sigma_min  <=>  L + 1.7 r >= Lambda(D, f0) := sigma_min / (D k^2)
    (b) L_neck <= D
plus the box. sigma <= sigma_max, neck+ligament-inside-cell and bank activity can never bind in the box;
`_prove_inactive_constraints` checks this at import against the live FeasibilityLimits and RAISES (not `assert`, which
`python -O` strips) on any drift, so the guarantee fails closed.

Construction (u layout; effective dimension 18):
    u[0]      D      in (D_MIN, D_HI)           D_MIN = smallest depth at which ANY bank is feasible at f0 <= F0_HI
    u[1]      face_t in (box)
    u[2]      wall_t in (box)
    u[3:7]    f0_1 < f0_2 < f0_3 < f0_4 in (f0_min(D), F0_HI): positive cumulative gaps, 5-part softmax (gauge p5)
    u[7:15]   per bank (r, L): r in (r_min(D, f0), R_HI), then L in (L_min(r), L_hi(D))
    u[15:18]  area log-ratios w_i = log(frac_i / frac_4), sequentially:  w_j in (max(0, w_<j) - S, min(0, w_<j) + S)
              => span(w_1, w_2, w_3, 0) < S = SPAN_MAX - margin, SPAN_MAX = 2 * 3.0 (the CMA logit box width)
Every conditional range is an open interval mapped by a sigmoid, so the image is exactly DOMAIN up to nested ~1e-9
relative safety margins on the sigma_min boundary and the span boundary (they keep every interval non-degenerate, so
float round-off cannot invert an interval at a corner of the feasible set). The CMA logits are the CENTERED gauge
a = v - (max v + min v) / 2 of v = (w_1, w_2, w_3, 0), which lies strictly inside [-3, 3].

The decoder never clips or projects after decoding; non-finite / wrong-shape / boolean inputs raise. The canonical
feasibility gate (dataset.is_feasible) remains authoritative -- this transform narrows the search space, it does not
certify. New module: no frozen (cvae.SOURCE_MODULES) module is edited.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import schema as DS
from . import constraints as FEAS
from ..physics import resonator as RM

DTYPE = torch.float64
NG = DS.N_GROUPS                                  # 4 banks
DIM = 18                                          # effective unconstrained geometry dimension
_LIM = FEAS.FeasibilityLimits()


class CoordinateDomainError(ValueError):
    """A design is outside the PHYS-GEN coordinate domain (encode), or an input is malformed (decode)."""


class CoordinateProofError(RuntimeError):
    """An import-time invariant of the construction does not hold for the live box / feasibility limits."""


def _require(ok, msg):
    if not ok:
        raise CoordinateProofError(msg)


# --- box: the locked CMA's design.schema box (bounds are read, never redefined) --------------------------------------
_lo, _hi = DS.bounds()
_require(len(_lo) == 3 + 4 * NG, f"design_space layout changed: DIM={len(_lo)}")
D_LO_BOX, D_HI = float(_lo[0]), float(_hi[0])
FACE_LO, FACE_HI = float(_lo[1]), float(_hi[1])
WALL_LO, WALL_HI = float(_lo[2]), float(_hi[2])
F0_LO, F0_HI = float(_lo[3]), float(_hi[3])
R_LO, R_HI = float(_lo[4]), float(_hi[4])
L_LO, L_HI_BOX = float(_lo[5]), float(_hi[5])
LOGIT_LO, LOGIT_HI = float(_lo[6]), float(_hi[6])
for _g in range(NG):                               # every bank shares the same per-bank box
    _require(tuple(_lo[3 + 4 * _g: 7 + 4 * _g]) == tuple(_lo[3:7]) and
             tuple(_hi[3 + 4 * _g: 7 + 4 * _g]) == tuple(_hi[3:7]), "per-bank box differs across banks")
_require(LOGIT_LO == -LOGIT_HI, "area-logit box must be symmetric for the centered gauge")

EC = RM.end_correction(1.0)                       # end-correction coefficient per metre of radius (2*0.85 = 1.7)
C_SOUND = float(RM.C)
SIGMA_MARGIN = 1e-9                               # relative margins (float round-off guards; see module docstring)
SIGMA_MIN_EFF = _LIM.sigma_min * (1.0 + SIGMA_MARGIN)
SIGMA_FLOOR_F0 = SIGMA_MIN_EFF * (1.0 + 2.0 * SIGMA_MARGIN)
R_FLOOR_FACTOR = 1.0 + SIGMA_MARGIN
SPAN_MAX = LOGIT_HI - LOGIT_LO                    # 6.0: largest area log-ratio span the CMA logits can express
SPAN = SPAN_MAX * (1.0 - SIGMA_MARGIN)            # strict interior -> centered logits never touch the box edge


def _k2(f0):
    return (2.0 * math.pi * f0 / C_SOUND) ** 2


def _L_hi(D):
    return min(L_HI_BOX, D)


def f0_min_np(D: float) -> float:
    """Lowest f0 at which a bank is feasible at depth D (best case r=R_HI, L=L_hi(D)), with the f0 margin."""
    return (C_SOUND / (2.0 * math.pi)) * math.sqrt(SIGMA_FLOOR_F0 / (D * (_L_hi(D) + EC * R_HI)))


def _solve_D_min() -> float:
    """Smallest D with f0_min(D) < F0_HI (bisection; f0_min is strictly decreasing in D)."""
    lo, hi = D_LO_BOX, D_HI
    _require(f0_min_np(hi) < F0_HI, "no feasible depth in the box: the coordinate domain is empty")
    if f0_min_np(lo) < F0_HI:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f0_min_np(mid) >= F0_HI:
            lo = mid
        else:
            hi = mid
    return hi


D_MIN = _solve_D_min()
F0_FLOOR_AT_D_HI = f0_min_np(D_HI)               # frozen design-space limitation: no bank tunes below this (~102 Hz)


def _prove_inactive_constraints():
    """The constraints NOT built into the transform cannot bind anywhere in the box (raises on drift)."""
    sig_max_box = D_HI * (_L_hi(D_HI) + EC * R_HI) * _k2(F0_HI)
    # neck + ligament inside cell:  2r + 2*lig <= r*sqrt(pi/sigma)  <=>  sigma <= pi / (2 + 2*lig/r)^2 (tightest at R_LO)
    sig_cell_min = math.pi / (2.0 + 2.0 * _LIM.ligament_min_m / R_LO) ** 2
    _require(sig_max_box < _LIM.sigma_max, f"sigma_max can bind: {sig_max_box} >= {_LIM.sigma_max}")
    _require(sig_max_box < sig_cell_min, f"neck-in-cell can bind: {sig_max_box} >= {sig_cell_min}")
    _require(_LIM.min_neck_r_m <= R_LO and R_HI <= _LIM.max_neck_r_m, "neck-radius box exceeds feasibility limits")
    _require(_LIM.min_face_t_m <= FACE_LO and FACE_HI <= _LIM.max_face_t_m, "facesheet box exceeds limits")
    _require(_LIM.min_wall_t_m <= WALL_LO and WALL_HI <= _LIM.max_wall_t_m, "wall box exceeds limits")
    _require(D_HI <= _LIM.max_depth_m and L_LO > 0.0, "depth/neck box exceeds limits")
    # every bank is ACTIVE: with span(log frac) <= SPAN_MAX the smallest fraction is 1 / (1 + (NG-1) e^SPAN_MAX)
    min_frac = 1.0 / (1.0 + (NG - 1) * math.exp(SPAN_MAX))
    _require(min_frac > _LIM.inactive_area_frac_eps, f"a bank can become inactive: min frac {min_frac}")
    return dict(sigma_max_in_box=sig_max_box, sigma_cell_floor=sig_cell_min, min_area_frac=min_frac)


INACTIVE_PROOF = _prove_inactive_constraints()


# =================================================================================================================
#  decode (torch, differentiable, batched)
# =================================================================================================================
def _interval(lo, hi, u):
    """lo + (hi - lo) * sigmoid(u), anchored at the NEARER end (bottom for u <= 0, top for u > 0 via
    sigmoid(-u) = 1 - sigmoid(u)): identical map, but lo <= x <= hi holds EXACTLY in floating point -- the plain
    bottom-anchored form can round one ulp above hi when sigmoid saturates to 1.0."""
    lo = torch.as_tensor(lo, dtype=DTYPE); hi = torch.as_tensor(hi, dtype=DTYPE)
    return torch.where(u <= 0, lo + (hi - lo) * torch.sigmoid(u), hi - (hi - lo) * torch.sigmoid(-u))


def _check_u(u):
    if not torch.is_tensor(u):
        raise CoordinateDomainError("u must be a torch tensor")
    if u.dtype == torch.bool or not u.is_floating_point():
        raise CoordinateDomainError(f"u must be floating point; got {u.dtype}")
    if u.dim() != 2 or u.shape[1] != DIM:
        raise CoordinateDomainError(f"u must have shape [B, {DIM}]; got {tuple(u.shape)}")
    if not bool(torch.isfinite(u).all()):
        raise CoordinateDomainError("u contains non-finite values")


def _area_logratios_t(ua):
    """ua [B,3] -> w [B,3] with span(w_1, w_2, w_3, 0) < SPAN, each w_j an exact conditional interval."""
    cols, hi_prev, lo_prev = [], torch.zeros_like(ua[:, 0]), torch.zeros_like(ua[:, 0])
    for j in range(NG - 1):
        lo = hi_prev - SPAN                     # max(0, w_<j) - SPAN
        hi = lo_prev + SPAN                     # min(0, w_<j) + SPAN
        w = _interval(lo, hi, ua[:, j])
        cols.append(w)
        hi_prev, lo_prev = torch.maximum(hi_prev, w), torch.minimum(lo_prev, w)
    return torch.stack(cols, dim=1)


def decode_t(u):
    """u [B,18] -> dict of physical tensors: D, face_t, wall_t [B]; f0, r, L, frac, cma_logit [B,4]; logratio [B,3];
    cma_vec [B,19] in the locked CMA's design.schema layout [D, face, wall, (f0, r, L, a) x 4] (physical units)."""
    _check_u(u)
    u = u.to(DTYPE)
    D = _interval(D_MIN, D_HI, u[:, 0])
    face_t = _interval(FACE_LO, FACE_HI, u[:, 1])
    wall_t = _interval(WALL_LO, WALL_HI, u[:, 2])

    L_hi = torch.clamp(D, max=L_HI_BOX)                                  # min(L_HI_BOX, D)
    f0_lo = (C_SOUND / (2.0 * math.pi)) * torch.sqrt(SIGMA_FLOOR_F0 / (D * (L_hi + EC * R_HI)))
    z = torch.cat([u[:, 3:7], torch.zeros_like(u[:, :1])], dim=1)     # 5-part softmax, gauge on the last part
    p = torch.softmax(z, dim=1)
    cum = torch.cumsum(p[:, :NG], dim=1)                                 # strictly increasing, in (0,1)
    tail = torch.flip(torch.cumsum(torch.flip(p[:, 1:], [1]), dim=1), [1])   # 1 - cum, summed directly (no cancellation)
    span_f0 = (F0_HI - f0_lo).unsqueeze(1)
    # nearer-end anchoring again: exact f0_lo <= f0 <= F0_HI even when cum saturates to 1.0
    f0 = torch.where(cum <= tail, f0_lo.unsqueeze(1) + span_f0 * cum, F0_HI - span_f0 * tail)

    Lam = SIGMA_MIN_EFF / (D.unsqueeze(1) * (2.0 * math.pi * f0 / C_SOUND) ** 2)   # need L + EC r >= Lam
    r_lo = torch.clamp((Lam * R_FLOOR_FACTOR - L_hi.unsqueeze(1)) / EC, min=R_LO)
    r = _interval(r_lo, R_HI, u[:, 7:15:2])
    L_lo = torch.clamp(Lam - EC * r, min=L_LO)
    L = _interval(L_lo, L_hi.unsqueeze(1), u[:, 8:15:2])                  # exact upper end => L_neck <= D exactly

    w = _area_logratios_t(u[:, 15:18])
    v = torch.cat([w, torch.zeros_like(w[:, :1])], dim=1)
    frac = torch.softmax(v, dim=1)
    cma_logit = v - 0.5 * (v.max(dim=1, keepdim=True).values + v.min(dim=1, keepdim=True).values)

    banks = torch.stack([f0, r, L, cma_logit], dim=2).reshape(u.shape[0], 4 * NG)
    cma_vec = torch.cat([D.unsqueeze(1), face_t.unsqueeze(1), wall_t.unsqueeze(1), banks], dim=1)
    return dict(D=D, face_t=face_t, wall_t=wall_t, f0=f0, r=r, L=L, logratio=w, frac=frac,
                cma_logit=cma_logit, cma_vec=cma_vec)


def decode_np(u):
    """Single design: u [18] -> (design dict, cma_vec np[19]). No clipping, no projection."""
    a = np.asarray(u)
    if a.dtype == bool:
        raise CoordinateDomainError("u must be floating point; got bool")
    out = decode_t(torch.as_tensor(np.asarray(a, float)).reshape(1, DIM))
    groups = [dict(f0=float(out["f0"][0, g]), r_neck=float(out["r"][0, g]), L_neck=float(out["L"][0, g]),
                   area_frac=float(out["frac"][0, g]), Q=DS.Q_FIXED) for g in range(NG)]
    design = dict(D=float(out["D"][0]), groups=groups, face_t=float(out["face_t"][0]), wall_t=float(out["wall_t"][0]))
    return design, out["cma_vec"][0].detach().numpy()


# =================================================================================================================
#  encode (numpy, exact inverse on the domain)
# =================================================================================================================
def _logit_in(x, lo, hi, name):
    t = (x - lo) / (hi - lo)
    if not math.isfinite(t) or not (0.0 < t < 1.0):
        raise CoordinateDomainError(f"{name}={x!r} outside open interval ({lo!r}, {hi!r})")
    return math.log(t) - math.log1p(-t)


def canonicalize(design):
    """Sort banks by f0 (strictly). The acoustics and mass are permutation-invariant over banks (tested); ties are
    outside the domain (strict order is required by the coordinate system)."""
    groups = sorted(design["groups"], key=lambda g: float(g["f0"]))
    f0s = [float(g["f0"]) for g in groups]
    if any(b <= a for a, b in zip(f0s, f0s[1:])):
        raise CoordinateDomainError(f"bank f0 values are not distinct: {f0s}")
    return dict(design, groups=groups)


def encode_np(design):
    """Design -> u [18]. Exact inverse of decode on the domain; raises CoordinateDomainError when the canonicalized
    design is outside it. Never clips."""
    d = canonicalize(design)
    D = float(d["D"])
    u = np.empty(DIM)
    u[0] = _logit_in(D, D_MIN, D_HI, "D")
    u[1] = _logit_in(float(d["face_t"]), FACE_LO, FACE_HI, "face_t")
    u[2] = _logit_in(float(d["wall_t"]), WALL_LO, WALL_HI, "wall_t")
    L_hi = _L_hi(D)
    f0_lo = f0_min_np(D)
    f0 = np.array([float(g["f0"]) for g in d["groups"]])
    cum = (f0 - f0_lo) / (F0_HI - f0_lo)
    p = np.diff(np.concatenate([[0.0], cum, [1.0]]))
    if not np.all(p > 0.0):
        raise CoordinateDomainError(f"f0 values {f0.tolist()} not strictly inside ({f0_lo}, {F0_HI})")
    u[3:7] = np.log(p[:NG]) - math.log(p[NG])
    for g, grp in enumerate(d["groups"]):
        r, L = float(grp["r_neck"]), float(grp["L_neck"])
        Lam = SIGMA_MIN_EFF / (D * _k2(f0[g]))
        u[7 + 2 * g] = _logit_in(r, max(R_LO, (Lam * R_FLOOR_FACTOR - L_hi) / EC), R_HI, f"bank{g}.r_neck")
        u[8 + 2 * g] = _logit_in(L, max(L_LO, Lam - EC * r), L_hi, f"bank{g}.L_neck")
    frac = np.array([float(g["area_frac"]) for g in d["groups"]])
    if not np.all(frac > 0.0):
        raise CoordinateDomainError("area fractions must be > 0 (inactive banks are outside the domain)")
    w = np.log(frac[:NG - 1] / frac[NG - 1])
    hi_prev = lo_prev = 0.0
    for j in range(NG - 1):
        u[15 + j] = _logit_in(float(w[j]), hi_prev - SPAN, lo_prev + SPAN, f"area log-ratio {j}")
        hi_prev, lo_prev = max(hi_prev, w[j]), min(lo_prev, w[j])
    return u
