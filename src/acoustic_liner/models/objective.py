# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 differentiable robust objective with coverage optimized INTERNALLY (prereg sec. 7; owner 2026-10-02).

    J(geometry, condition) = max_{0 <= f <= f_max(geometry, condition)}  W_tau(geometry, f, condition)

W_tau is the robust worst-case spectrum-weighted dOASPL over the canonical (alpha_bare x kappa) corner grid -- the hard
MIN when tau == 0 (exactly `oracle.robust_delta_oaspl(...)["worst"]`), or the smooth soft-min
    W_tau = -tau * log( mean_c exp(-d_c / tau) )
for tau > 0. (Prereg sec. 7 writes the log-SUM-exp; the log-MEAN-exp differs by the constant tau*log(C), so gradients
and the coverage maximizer are identical, and the no-treatment value W(f=0) stays exactly 0 dB.) f_max is the installed
mass cap: m_total = (m_areal + m_install) * f * A_wall <= m_target. f = 0 (no treatment, 0 dB) is always admissible,
exactly as in `physics.quality.quality_eval`.

Coverage optimization mirrors the canonical solver (26-point grid on [0, 1] restricted to the mass budget, argmax,
golden-section refine of the bracket) but is CONTINUOUS (the cap f_max itself is a candidate and the golden step is
carried to machine precision), and runs without gradients. Gradients then follow the envelope theorem:
    interior optimum:      dJ/dtheta = dW/dtheta at fixed f*                     (f* detached)
    mass cap binding:      dJ/dtheta = dW/dtheta + dW/df * df_max/dtheta         (f* = f_max(theta), differentiable)
The REPORTED performance is never this surrogate: canonical `quality_eval` remains the only accepted score.

The batched oracle reimplements `physics.differentiable.TorchOracle` (gradient-verified, single design, Python corner loop)
for [B] designs and [B, F, C] coverage x corner tensors, importing every constant from it (nothing re-typed). It is
validated against the canonical numpy oracle in `tests/unit/test_physgen_objective.py`. New module; no frozen module is
edited.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..physics import oracle as O2
from ..physics import quality as Q
from ..physics import differentiable as VT

DTYPE = torch.float64
NG = VT._NG
COV_GRID_N = Q.COV_GRID_N            # 26, canonical coverage grid density
GOLDEN_ITERS = 90                      # bracket 1/25 * 0.618^90 << float eps: machine-precision coverage optimum
BIND_RTOL = 1e-12                      # f* within this relative distance of f_max => the mass cap is binding
REFINE_MIN_GAIN = 1e-12                # dB; else a round-off 'win' drifts f* off exact no-treatment (f = 0)

_T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DTYPE)
_OMEGA = _T(2.0 * np.pi * VT._FINE)                                  # [900]
_COS = _T(np.cos(VT._THETA)); _SIN2 = _T(np.sin(2.0 * VT._THETA)); _WTH = _T(VT._WTHETA)
_I0 = torch.as_tensor(VT._I0, dtype=torch.long); _I1 = torch.as_tensor(VT._I1, dtype=torch.long); _W1 = _T(VT._W1)
_A_GRID = O2._grid_with_node(np.linspace(O2.ALPHA_BARE_RANGE[0], O2.ALPHA_BARE_RANGE[1], VT._N_A),
                             O2.ALPHA_BARE_NOMINAL)


class ObjectiveInputError(ValueError):
    """Malformed objective input (shape, dtype, non-finite, non-positive condition values)."""


# =================================================================================================================
#  batched physics
# =================================================================================================================
def alpha_band(D, f0, r, L, frac):
    """Diffuse liner absorption at the 8 design bands. D [B]; f0, r, L, frac [B, NG] -> [B, 8]."""
    S = math.pi * r ** 2
    L_eff = L + VT._END_CORR * r
    k = 2.0 * math.pi * f0 / VT._C
    V_cav = S / (L_eff * k ** 2)
    M_a = VT._RHO * L_eff / S
    C_a = V_cav / (VT._RHO * VT._C ** 2)
    R_r = (1.0 / torch.sqrt(M_a * C_a)) * M_a / VT._Q
    density = frac * D.unsqueeze(1) / torch.clamp(V_cav, min=1e-9)
    omega = _OMEGA.view(1, 1, -1)
    reactance = omega * M_a.unsqueeze(2) - 1.0 / (omega * C_a.unsqueeze(2))          # [B, NG, 900]
    Z_r = torch.complex(R_r.unsqueeze(2).expand_as(reactance), reactance)
    beta = (density.unsqueeze(2).to(torch.complex128) / Z_r).sum(dim=1)               # [B, 900]
    zeta = (1.0 / beta) / VT._Z_AIR
    zc = zeta.unsqueeze(2) * _COS.view(1, 1, -1).to(torch.complex128)                  # [B, 900, 48]
    Rc = (zc - 1.0) / (zc + 1.0)
    raw = torch.clamp(((1.0 - Rc.abs() ** 2) * _SIN2.view(1, 1, -1)) @ _WTH, 0.0, 1.0)   # [B, 900]
    return raw[:, _I0] * (1.0 - _W1) + raw[:, _I1] * _W1


def areal_mass(D, face_t, wall_t, f0, r, L, frac):
    """Corrected shared-wall liner areal mass [kg/m^2] (as physics.differentiable / physics.mass; backing_t = 0)."""
    S = math.pi * r ** 2
    V = S / ((L + VT._END_CORR * r) * (2.0 * math.pi * f0 / VT._C) ** 2)
    part = VT._RHO_WALL * wall_t.unsqueeze(1) * D.unsqueeze(1) * 2.0 * torch.sqrt(D.unsqueeze(1) / torch.clamp(V, min=1e-9))
    return VT._RHO_WALL * face_t + torch.sum(frac * part, dim=1)


def corner_grid(kappa):
    """Per-condition canonical (alpha_bare, kappa) corner grid, padded: returns ab [B, C], kk [B, C], mask [B, C]."""
    rows = []
    for kn in np.asarray(kappa, float):
        k_lo = max(0.0, kn - O2.DELTA_KAPPA_DEFAULT); k_hi = kn + O2.DELTA_KAPPA_DEFAULT
        kg = (np.array([kn]) if k_hi <= k_lo else O2._grid_with_node(np.linspace(k_lo, k_hi, VT._N_K), kn))
        rows.append([(a, k) for a in _A_GRID for k in kg])
    C = max(len(r) for r in rows)
    ab = np.zeros((len(rows), C)); kk = np.zeros((len(rows), C)); mask = np.zeros((len(rows), C), bool)
    for i, r in enumerate(rows):
        ab[i, :len(r)] = [a for a, _ in r]; kk[i, :len(r)] = [k for _, k in r]; mask[i, :len(r)] = True
        ab[i, len(r):] = r[-1][0]; kk[i, len(r):] = r[-1][1]                         # padded corners are masked
    return _T(ab), _T(kk), torch.as_tensor(mask)


def corner_doaspl(a_band, f, w_energy, A_wall, ab, kk):
    """dOASPL at every (coverage, corner). a_band [B,8]; f [B,F]; w_energy [B,8]; A_wall [B]; ab, kk [B,C] ->
    [B, F, C]."""
    S = A_wall.view(-1, 1, 1, 1)
    St = f.unsqueeze(2).unsqueeze(3) * S                                               # [B, F, 1, 1]
    A_other = (kk * A_wall.unsqueeze(1)).unsqueeze(1).unsqueeze(3)                     # [B, 1, C, 1]
    abx = ab.unsqueeze(1).unsqueeze(3)                                                 # [B, 1, C, 1]
    A0 = abx * S + A_other
    A1 = a_band.unsqueeze(1).unsqueeze(2) * St + abx * (S - St) + A_other              # [B, F, C, 8]
    nr = 10.0 * torch.log10(A1 / A0)
    return -10.0 * torch.log10(torch.sum(w_energy.view(-1, 1, 1, 8) * 10.0 ** (-nr / 10.0), dim=3))


def robust_value(d, mask, tau):
    """Hard (tau == 0) or log-mean-exp soft min over the corner axis of d [B, F, C]."""
    m = mask.unsqueeze(1)
    if tau == 0.0:
        return torch.where(m, d, torch.full_like(d, float("inf"))).min(dim=2).values
    z = torch.where(m, -d / tau, torch.full_like(d, float("-inf")))
    return -tau * (torch.logsumexp(z, dim=2) - torch.log(mask.sum(dim=1, keepdim=True).to(DTYPE)))


# =================================================================================================================
#  objective
# =================================================================================================================
def condition_tensors(conds):
    """List of condition dicts (weights/spl/m_total_target/A_wall/kappa) -> batched tensors. Validates values."""
    spl = _T([np.asarray(c["spl"], float) for c in conds])
    A_wall = _T([c["A_wall"] for c in conds]); m_t = _T([c["m_total_target"] for c in conds])
    kappa = np.array([c["kappa"] for c in conds], float)
    for nm, t in (("spl", spl), ("A_wall", A_wall), ("m_total_target", m_t), ("kappa", _T(kappa))):
        if not bool(torch.isfinite(t).all()):
            raise ObjectiveInputError(f"{nm} must be finite")
    if not bool((A_wall > 0).all()) or not bool((m_t >= 0).all()) or np.any(kappa < 0):
        raise ObjectiveInputError("A_wall must be > 0, m_total_target >= 0, kappa >= 0")
    e = 10.0 ** ((spl - spl.max(dim=1, keepdim=True).values) / 10.0)
    ab, kk, mask = corner_grid(kappa)
    return dict(w_energy=e / e.sum(dim=1, keepdim=True), A_wall=A_wall, m_target=m_t, ab=ab, kk=kk, mask=mask)


def _W(a_band, f, ct, tau):
    return robust_value(corner_doaspl(a_band, f, ct["w_energy"], ct["A_wall"], ct["ab"], ct["kk"]), ct["mask"], tau)


@torch.no_grad()
def _argmax_coverage(a_band, f_max, ct, tau):
    """Continuous canonical-style coverage solve on [0, f_max]: grid (+ the cap) argmax, golden refine of the bracket."""
    B = a_band.shape[0]
    grid = torch.linspace(0.0, 1.0, COV_GRID_N, dtype=DTYPE).expand(B, -1)
    cand = torch.cat([torch.minimum(grid, f_max.unsqueeze(1)), f_max.unsqueeze(1)], dim=1)   # capped grid + cap
    cand, _ = torch.sort(cand, dim=1)
    v = _W(a_band, cand, ct, tau)
    j = v.argmax(dim=1)
    idx = torch.arange(B)
    lo = cand[idx, torch.clamp(j - 1, min=0)]; hi = cand[idx, torch.clamp(j + 1, max=cand.shape[1] - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo.clone(), hi.clone()
    for _ in range(GOLDEN_ITERS):
        c = b - gr * (b - a); d = a + gr * (b - a)
        vc, vd = _W(a_band, torch.stack([c, d], 1), ct, tau).unbind(1)
        left = vc < vd
        a = torch.where(left, c, a); b = torch.where(left, b, d)
    f_g = 0.5 * (a + b)
    v_g = _W(a_band, f_g.unsqueeze(1), ct, tau)[:, 0]
    v_grid = v[idx, j]
    take = v_g > v_grid + REFINE_MIN_GAIN          # refinement must beat the grid point by more than round-off
    return torch.where(take, f_g, cand[idx, j])


def objective(geom, ct, tau=0.0):
    """geom: dict with D, face_t, wall_t [B] and f0, r, L, frac [B, NG] (e.g. design.coordinates.decode_t output).
    ct: `condition_tensors(...)`. Returns dict: J [B] (differentiable surrogate), worst_hard [B] (hard-min dOASPL at the
    chosen coverage), f_star, f_max, binding [B], m_total [B]."""
    if not (isinstance(tau, float) and math.isfinite(tau) and tau >= 0.0):
        raise ObjectiveInputError(f"tau must be a finite float >= 0; got {tau!r}")
    a_band = alpha_band(geom["D"], geom["f0"], geom["r"], geom["L"], geom["frac"])
    m_areal = areal_mass(geom["D"], geom["face_t"], geom["wall_t"], geom["f0"], geom["r"], geom["L"], geom["frac"])
    f_max = torch.clamp(ct["m_target"] / ((m_areal + VT._INSTALL_AREAL) * ct["A_wall"]), max=1.0)
    f_star = _argmax_coverage(a_band.detach(), f_max.detach(), ct, tau)
    binding = f_star >= f_max.detach() * (1.0 - BIND_RTOL)
    f_use = torch.where(binding, f_max, f_star)
    J = _W(a_band, f_use.unsqueeze(1), ct, tau)[:, 0]
    with torch.no_grad():
        hard = J if tau == 0.0 else _W(a_band, f_use.unsqueeze(1), ct, 0.0)[:, 0]
    m_total = (m_areal + VT._INSTALL_AREAL) * f_use * ct["A_wall"]
    return dict(J=J, worst_hard=hard.detach(), f_star=f_star, f_max=f_max.detach(), binding=binding,
                m_total=m_total.detach())


# =================================================================================================================
#  training objective: physics-only soft-coverage continuation (owner decision 2026-10-02)
# =================================================================================================================
COV_NODES_T = torch.linspace(0.0, 1.0, COV_GRID_N, dtype=DTYPE)   # canonical 26 normalized nodes, t = 0 ... 1
TAU_COV_START, TAU_COV_END = 0.05, 0.01                            # dB; annealing schedule endpoints (owner)


def train_objective(geom, ct, tau_cov, tau_corner=0.0, node_grad="cap_only", interior_f_max=None):
    """Differentiable soft maximum over the SAME physical coverage curve (no surrogate, no labels):

        J_train = tau_cov * log( (1/N) * sum_j exp( W_tau_corner(f_j) / tau_cov ) ),   f_j = t_j * f_max,

    t_j the canonical 26 nodes (t=0 and t=1 included). W(0) = 0 exactly, so a no-treatment design keeps the 0 dB node,
    while its negative-coverage nodes keep small weights whose gradients push the design toward being beneficial.
    Every node is <= the mass cap. For TRAINING ONLY: checkpoint selection and every reported result use the hard
    exact scorer (`physics.scorer.quality_eval_exact`).

    node_grad="cap_only" (default): the VALUE is exactly the formula above, but gradients reach f_max (hence the
    mass-only coordinates facesheet / wall thickness) only through the cap node t = 1; interior node POSITIONS are a
    fixed quadrature grid per evaluation (detached), like the envelope theorem's detached f*. Differentiating through
    the node positions ("all") manufactures mass gradients on designs whose optimum is not at the cap (measured
    median 3.6e-3 of the largest component at tau_cov = 0.01; exactly 0 in the hard objective).

    VALIDATION ONLY: `interior_f_max` freezes the interior node positions at a supplied (base-point) cap so that finite
    differences reproduce exactly the cap_only derivative (the chain-rule check); training never passes it."""
    if node_grad not in ("cap_only", "all"):
        raise ObjectiveInputError(f"node_grad must be 'cap_only' or 'all'; got {node_grad!r}")
    if not (isinstance(tau_cov, float) and math.isfinite(tau_cov) and tau_cov > 0.0):
        raise ObjectiveInputError(f"tau_cov must be a finite float > 0; got {tau_cov!r}")
    if not (isinstance(tau_corner, float) and math.isfinite(tau_corner) and tau_corner >= 0.0):
        raise ObjectiveInputError(f"tau_corner must be a finite float >= 0; got {tau_corner!r}")
    a_band = alpha_band(geom["D"], geom["f0"], geom["r"], geom["L"], geom["frac"])
    m_areal = areal_mass(geom["D"], geom["face_t"], geom["wall_t"], geom["f0"], geom["r"], geom["L"], geom["frac"])
    f_max = torch.clamp(ct["m_target"] / ((m_areal + VT._INSTALL_AREAL) * ct["A_wall"]), max=1.0)
    if node_grad == "all":
        f_nodes = COV_NODES_T.unsqueeze(0) * f_max.unsqueeze(1)                      # [B, 26], all <= f_max
    else:
        f_int = f_max.detach() if interior_f_max is None else torch.as_tensor(interior_f_max, dtype=DTYPE).detach()
        f_nodes = torch.cat([COV_NODES_T[:-1].unsqueeze(0) * f_int.unsqueeze(1), f_max.unsqueeze(1)], 1)
    W = _W(a_band, f_nodes, ct, tau_corner)                                            # [B, 26]
    J = tau_cov * (torch.logsumexp(W / tau_cov, dim=1) - math.log(COV_GRID_N))
    return dict(J=J, W_nodes=W, f_nodes=f_nodes, f_max=f_max)


def tau_cov_schedule(progress):
    """Geometric anneal TAU_COV_START -> TAU_COV_END over training progress in [0, 1] (shape TO_PIN in r1)."""
    p = min(max(float(progress), 0.0), 1.0)
    return float(TAU_COV_START * (TAU_COV_END / TAU_COV_START) ** p)


# =================================================================================================================
#  curriculum / loss
# =================================================================================================================
def curriculum(p, cfg):
    """Progress p in [0, 1] -> (phase, w_perf, tau_cov)."""
    a, b, c, w0 = cfg["stabilize"], cfg["ramp_end"], cfg["sharpen_end"], cfg["w_perf_start"]
    if p < a:
        return 1, w0, tau_cov_schedule(0.0)
    if p < b:
        return 2, w0 + (1.0 - w0) * (p - a) / (b - a), tau_cov_schedule(0.0)
    if p < c:
        return 3, 1.0, tau_cov_schedule((p - b) / (c - b))
    return 4, 1.0, tau_cov_schedule(1.0)
