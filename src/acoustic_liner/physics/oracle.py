# -*- coding: utf-8 -*-
r"""Case-F v2.0 Sabine oracle (diffuse-absorption, spectrum-weighted, robust dOASPL).

MIGRATED from `fairing_case/v2/oracle_v2.py` (frozen SHA-256
dac8a0ae35454a1f22a1c01caa4cc51852afe3055a8edb44b185358e2b60c136) for the §14 step-8 self-containment migration
(the historical migration record, which is not distributed). Changes vs the reference:
  * `sys.path` + `import resonator_model/fairing_oracle/mass_model` replaced by intra-package imports;
  * the ACOUSTIC functions (z_surface, diffuse_alpha*, partial_nr_band, delta_oaspl, robust_delta_oaspl, and the
    liner band absorption) are byte-identical to the reference -> exact numerical equivalence (fixture-verified);
  * MASS is rewired to the CORRECTED shared-wall model `physics.mass` (2*sqrt(n) partitions + facesheet +
    BOM install), REPLACING the reference's v1 `fairing_oracle.areal_mass` (4*sqrt(n)) + lumped `mu_A` install.
    This intentionally changes the mass-dependent labels (areal/total/FOM); the acoustic labels are unchanged.
    The reference v1 install parameter `mu_A` is SUPERSEDED by the named `physics.mass.InstallBOM`.

FRAMING: exact agreement with the reference proves faithful migration, NOT physical correctness -- this module
also carries independent physics/passivity/unit tests (see tests/unit).
"""
from __future__ import annotations

import numpy as np

from . import liner as OR  # reuse-only acoustics (Za, group density, constants)
from . import resonator as RM  # frozen resonator physics
from . import mass as MM  # CORRECTED shared-wall / BOM mass model (release)

ORACLE_V2_VERSION = "2.0.1-migrated"
Z_AIR = OR.Z_AIR
DESIGN_BAND_HZ = OR.DESIGN_BAND_HZ         # [50..250] 1/3-oct centers (8 bands)
ALPHA_BARE_RANGE = (0.02, 0.10)            # robust range (V2-SPEC sec.7)
ALPHA_BARE_NOMINAL = 0.05
KAPPA_OTHER_DEFAULT = 0.0                  # nominal A_other/A_wall (band-INDEPENDENT ratio)
DELTA_KAPPA_DEFAULT = 0.03                 # hybrid kappa half-width (GLOBAL; 0.0 => kappa known exactly)

# Representative wall area (matches the v1 model + the authoritative Falcon scenario A_wall). Only a fallback:
# the pipeline always passes an explicit A_wall from the scenario.
_R = 5.2 / 2.0
A_WALL_DEFAULT = np.pi * 5.2 * 6.7 + np.pi * _R * float(np.hypot(_R, 13.1 - 6.7))   # ~165.9 m^2


def falcon9_design_band() -> np.ndarray:
    """Falcon 9 SPL (dB) at the 8 design-band centers (from the frozen fairing table)."""
    return np.array([OR.FALCON9_SPL[f] for f in DESIGN_BAND_HZ], dtype=float)


# ---------------------------------------------------------------------------- #
#  Acoustics (byte-identical to the reference)
# ---------------------------------------------------------------------------- #
def z_surface(freqs: np.ndarray, groups, D: float) -> np.ndarray:
    """Complex specific surface impedance z_s(f) [Pa*s/m] of the parallel resonator-group liner.
    Admittances add: beta = sum_i n_i/Za_i ; z_s = 1/beta."""
    beta = np.zeros_like(freqs, dtype=complex)
    for g in groups:
        S = RM.neck_area(g["r_neck"])
        L_eff = RM.effective_neck_length(g["L_neck"], g["r_neck"])
        V_cav = RM.cavity_volume_for_target_f0(g["f0"], S, L_eff)
        Za = OR._resonator_Za(freqs, g["r_neck"], g["L_neck"], V_cav, g["Q"])
        beta += OR._group_density(g, D) / Za
    with np.errstate(divide="ignore", invalid="ignore"):
        z_s = np.where(np.abs(beta) > 0, 1.0 / beta, 1e30 + 0j)
    return z_s


def _gl_angle_nodes(n: int):
    """Gauss-Legendre nodes/weights for the integral over theta in [0, pi/2]."""
    x, w = np.polynomial.legendre.leggauss(n)
    theta = (np.pi / 4.0) * (x + 1.0)
    return theta, (np.pi / 4.0) * w


def alpha_oblique(z_s, cos_theta):
    """Locally-reacting plane-wave absorption at incidence angle(s). z_s specific [Pa*s/m]."""
    zeta = np.asarray(z_s, dtype=complex) / Z_AIR
    zc = np.multiply.outer(zeta, np.asarray(cos_theta))
    R = (zc - 1.0) / (zc + 1.0)
    return 1.0 - np.abs(R) ** 2


def diffuse_alpha_raw(z_s, n_angle: int = 48) -> np.ndarray:
    """UNCLIPPED diffuse (random-incidence) absorption via Paris' formula."""
    theta, w = _gl_angle_nodes(n_angle)
    a = alpha_oblique(z_s, np.cos(theta))
    return np.real((a * np.sin(2.0 * theta)) @ w)


def diffuse_alpha(z_s, n_angle: int = 48, tol: float = 1e-9) -> np.ndarray:
    """Diffuse absorption, only ROUNDOFF clipped. Raises if the raw value is materially outside [0,1]
    (non-passive impedance) instead of silently clipping it away."""
    raw = diffuse_alpha_raw(z_s, n_angle=n_angle)
    if np.any(raw < -tol) or np.any(raw > 1.0 + tol):
        raise ValueError("diffuse_alpha: raw value outside [0,1] beyond tolerance "
                         "(non-passive impedance?) -- min=%.3e max=%.3e"
                         % (float(np.min(raw)), float(np.max(raw))))
    return np.clip(raw, 0.0, 1.0)


def liner_diffuse_alpha_band(groups, D: float, n_angle: int = 48) -> np.ndarray:
    """Diffuse absorption of the liner at the 8 design-band centers."""
    centers = np.array(DESIGN_BAND_HZ, dtype=float)
    fine = np.linspace(DESIGN_BAND_HZ[0] * 0.9, DESIGN_BAND_HZ[-1] * 1.05, 900)
    a_fine = diffuse_alpha(z_surface(fine, groups, D), n_angle=n_angle)
    return np.interp(centers, fine, a_fine)


def _validate_fcov(f_cov: float, tol: float = 1e-9) -> float:
    if not (-tol <= f_cov <= 1.0 + tol):
        raise ValueError(f"f_cov must be in [0,1]; got {f_cov}")
    return float(min(1.0, max(0.0, f_cov)))


def partial_nr_band(alpha_liner_band, alpha_wall, f_cov: float, A_wall: float, A_other_band=0.0):
    """Per-band interior NR [dB] from the Sabine total-absorption ratio with partial coverage."""
    f_cov = _validate_fcov(f_cov)
    alpha_liner_band = np.asarray(alpha_liner_band, dtype=float)
    alpha_wall = np.asarray(alpha_wall, dtype=float)
    A_other_band = np.asarray(A_other_band, dtype=float)
    S_wall = float(A_wall)
    S_treated = f_cov * S_wall
    A0 = alpha_wall * S_wall + A_other_band
    A1 = alpha_liner_band * S_treated + alpha_wall * (S_wall - S_treated) + A_other_band
    return 10.0 * np.log10(A1 / A0)


def delta_oaspl(nr_band, spl_band) -> float:
    """Spectrum-weighted interior OASPL reduction [dB] (energy-weighted, NOT arithmetic mean of band dB):
        w_i = 10^(L_i/10)/sum 10^(L_j/10) ;  dOASPL = -10 log10( sum_i w_i 10^(-NR_i/10) )."""
    nr = np.asarray(nr_band, dtype=float)
    spl = np.asarray(spl_band, dtype=float)
    if nr.shape != spl.shape:
        raise ValueError(f"nr_band {nr.shape} and spl_band {spl.shape} shape mismatch")
    if not (np.all(np.isfinite(nr)) and np.all(np.isfinite(spl))):
        raise ValueError("non-finite value in delta_oaspl input")
    e = 10.0 ** ((spl - np.max(spl)) / 10.0)
    s = e.sum()
    if not (s > 0.0 and np.isfinite(s)):
        raise ValueError("zero-energy or non-finite spectrum")
    w = e / s
    return float(-10.0 * np.log10(np.sum(w * 10.0 ** (-nr / 10.0))))


def _grid_with_node(grid, node, atol=1e-12):
    """Return `grid` with `node` inserted iff it is not already a node within `atol`."""
    grid = np.asarray(grid, float)
    if np.any(np.isclose(grid, node, rtol=0.0, atol=atol)):
        return grid
    return np.sort(np.append(grid, node))


def robust_delta_oaspl(alpha_liner_band, spl_band, f_cov: float, A_wall: float,
                       kappa_nominal: float = 0.0, delta_kappa: float = DELTA_KAPPA_DEFAULT,
                       alpha_bare_range=ALPHA_BARE_RANGE,
                       alpha_bare_nominal: float = ALPHA_BARE_NOMINAL, n_a=9, n_k=5):
    """Spectrum-weighted dOASPL, robust JOINTLY over the (alpha_bare, kappa) box. worst = MIN dOASPL over the
    joint grid. delta_kappa=0 => kappa known exactly (point)."""
    if kappa_nominal < 0 or delta_kappa < 0:
        raise ValueError("kappa_nominal and delta_kappa must be >= 0")
    k_lo = max(0.0, kappa_nominal - delta_kappa)
    k_hi = kappa_nominal + delta_kappa
    assert k_lo <= kappa_nominal <= k_hi, "kappa_nominal outside its own range"
    a_grid = _grid_with_node(np.linspace(alpha_bare_range[0], alpha_bare_range[1], n_a), alpha_bare_nominal)
    k_grid = (np.array([kappa_nominal]) if k_hi <= k_lo
              else _grid_with_node(np.linspace(k_lo, k_hi, n_k), kappa_nominal))
    vals = []
    for ab in a_grid:
        for k in k_grid:
            nr = partial_nr_band(alpha_liner_band, ab, f_cov, A_wall, k * A_wall)
            vals.append(delta_oaspl(nr, spl_band))
    vals = np.asarray(vals)
    nr_nom = partial_nr_band(alpha_liner_band, alpha_bare_nominal, f_cov, A_wall, kappa_nominal * A_wall)
    nominal = delta_oaspl(nr_nom, spl_band)
    return dict(nominal=nominal, worst=float(vals.min()), best=float(vals.max()),
                n_grid=int(vals.size), kappa_range=[k_lo, k_hi], alpha_bare_range=list(alpha_bare_range))


# ---------------------------------------------------------------------------- #
#  Mass helper (CORRECTED model) + the v2.0 label
# ---------------------------------------------------------------------------- #
def _group_cavity_volume(g: dict) -> float:
    """Resonator cavity volume derived from f0 (same physics the acoustics use); required by the corrected
    mass model's local cell density."""
    S = RM.neck_area(g["r_neck"])
    L_eff = RM.effective_neck_length(g["L_neck"], g["r_neck"])
    return float(RM.cavity_volume_for_target_f0(g["f0"], S, L_eff))


def label_v2(design: dict) -> dict:
    """Score a Case-F v2 design against a SPECTRUM.
    design = { D, groups[{f0,r_neck,L_neck,area_frac,Q}], face_t, wall_t, f_cov, A_wall, spectrum (8 SPL dB),
               kappa_nominal, m_fixed }.
    OBJECTIVE (v2.0): pair (max dOASPL_robust_worst, min m_total). MASS uses the CORRECTED shared-wall model."""
    groups = design["groups"]
    D = float(design["D"])
    face_t = design.get("face_t", 0.0008)
    wall_t = design.get("wall_t", 0.0006)
    f_cov = _validate_fcov(float(design.get("f_cov", 1.0)))
    A_wall = float(design.get("A_wall", A_WALL_DEFAULT))
    if not (A_wall > 0.0 and np.isfinite(A_wall)):
        raise ValueError(f"A_wall must be positive; got {A_wall}")
    if "spectrum" not in design:
        raise ValueError("v2 labels require an explicit 'spectrum' (8 SPL dB at DESIGN_BAND_HZ)")
    spectrum = np.asarray(design["spectrum"], dtype=float)
    if spectrum.shape != (len(DESIGN_BAND_HZ),):
        raise ValueError(f"spectrum must have {len(DESIGN_BAND_HZ)} bands; got {spectrum.shape}")
    kappa_nominal = float(design.get("kappa_nominal", design.get("kappa_other", KAPPA_OTHER_DEFAULT)))
    if kappa_nominal < 0:
        raise ValueError(f"kappa_nominal must be >= 0; got {kappa_nominal}")
    delta_kappa = DELTA_KAPPA_DEFAULT              # GLOBAL immutable constant (identical conditions -> identical labels)
    m_fixed = float(design.get("m_fixed", 0.0))
    if m_fixed < 0:
        raise ValueError("m_fixed must be >= 0")

    a_band = liner_diffuse_alpha_band(groups, D)
    d = robust_delta_oaspl(a_band, spectrum, f_cov, A_wall, kappa_nominal, delta_kappa)

    # CORRECTED shared-wall mass (physics.mass): areal = facesheet + 2*sqrt(n) partitions (+ backing);
    # total = m_liner + BOM install (adhesive/fastener/frame/margin) + m_fixed. REPLACES v1 areal_mass + mu_A.
    groups_mass = [dict(g, cavity_volume=_group_cavity_volume(g)) for g in groups]
    m_areal = MM.liner_areal_mass(groups_mass, D, face_t, wall_t)
    mass_rec = MM.total_installed_mass(m_areal, f_cov, A_wall, m_fixed=m_fixed)
    m_total = mass_rec["m_total"]
    fom_worst = (d["worst"] / m_total) if m_total > 0.0 else None

    feas_depth = D <= OR.MAX_DEPTH_M + 1e-9
    feas_neck = all(g["r_neck"] >= OR.MIN_NECK_R_M for g in groups)
    feas_area = abs(sum(g["area_frac"] for g in groups) - 1.0) < 0.02
    feasible = bool(feas_depth and feas_neck and feas_area)

    return dict(
        dOASPL_worst_dB=d["worst"], total_installed_mass_kg=m_total,
        dOASPL_nominal_dB=d["nominal"], dOASPL_best_dB=d["best"], robust_ngrid=d["n_grid"],
        diffuse_alpha_band={int(f): float(a) for f, a in zip(DESIGN_BAND_HZ, a_band)},
        band_avg_diffuse_alpha=float(np.mean(a_band)),
        areal_mass_kg_m2=m_areal, mass_breakdown=mass_rec, f_cov=f_cov, A_wall=A_wall,
        kappa_nominal=kappa_nominal, delta_kappa=delta_kappa, kappa_eval_range=d["kappa_range"],
        treated_area_m2=MM.treated_area(f_cov, A_wall),
        dOASPL_per_kg_total=fom_worst,
        mass_model="corrected_shared_wall_v2",     # provenance: NOT the v1 4*sqrt(n) areal model
        feasible=feasible,
        feas_flags=dict(depth=feas_depth, neck=feas_neck, area=feas_area),
    )


def constrained_min_mass(design: dict, dOASPL_target: float) -> dict:
    """Is THIS design FEASIBLE and does its robust worst dOASPL meet the target, and at what total mass?"""
    m = label_v2(design)
    ok = m["feasible"] and (m["dOASPL_worst_dB"] >= dOASPL_target)
    return dict(feasible=m["feasible"], meets_target=bool(m["dOASPL_worst_dB"] >= dOASPL_target),
                dOASPL_worst_dB=m["dOASPL_worst_dB"], total_installed_mass_kg=m["total_installed_mass_kg"],
                objective=m["total_installed_mass_kg"] if ok else float("inf"))


def _golden_max(fn, a, b, tol=1e-5, n_iter=60):
    """Golden-section MAXIMIZE fn on [a,b] (fn assumed unimodal on the bracket)."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(n_iter):
        if b - a < tol:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = fn(d)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = fn(c)
    f_star = 0.5 * (a + b)
    return f_star, fn(f_star)


def min_coverage_for_target(design: dict, dOASPL_target: float, n_scan: int = 201,
                            refine_tol: float = 1e-4) -> dict:
    """Smallest f_cov (=> min installed mass) whose robust worst dOASPL >= target, for a fixed liner geometry."""
    if not label_v2(dict(design, f_cov=1.0))["feasible"]:
        return dict(feasible=False, reason="geometry infeasible (depth/neck/area)",
                    f_cov=None, total_installed_mass_kg=None)

    def worst(f):
        return label_v2(dict(design, f_cov=float(f)))["dOASPL_worst_dB"]

    fs = np.linspace(0.0, 1.0, n_scan)
    ws = np.array([worst(f) for f in fs])
    cross = np.nonzero(ws >= dOASPL_target)[0]
    k = int(np.argmax(ws))
    a_ref, b_ref = fs[max(0, k - 1)], fs[min(n_scan - 1, k + 1)]
    f_max, max_worst = _golden_max(worst, a_ref, b_ref)
    max_worst = max(max_worst, float(ws.max()))

    if cross.size == 0 and max_worst < dOASPL_target:
        return dict(feasible=False, reason="dOASPL_target unattainable at any coverage",
                    dOASPL_target=dOASPL_target, max_dOASPL_worst_dB=float(max_worst),
                    f_cov_at_max=float(f_max), f_cov=None, total_installed_mass_kg=None)

    if cross.size > 0:
        i = int(cross[0])
        lo = fs[i - 1] if i > 0 else 0.0
        hi = fs[i]
    else:
        below = fs[fs < f_max]
        lo = float(below[np.nonzero(ws[:below.size] < dOASPL_target)[0][-1]]) if below.size else 0.0
        hi = float(f_max)
    while hi - lo > refine_tol:
        mid = 0.5 * (lo + hi)
        if worst(mid) >= dOASPL_target:
            hi = mid
        else:
            lo = mid
    m = label_v2(dict(design, f_cov=float(hi)))
    return dict(feasible=True, dOASPL_target=dOASPL_target, f_cov=float(hi),
                total_installed_mass_kg=m["total_installed_mass_kg"],
                dOASPL_worst_dB=m["dOASPL_worst_dB"],
                max_dOASPL_worst_dB=float(max_worst), f_cov_at_max=float(f_max))


DELTA_KAPPA = DELTA_KAPPA_DEFAULT


# --------------------------------------------------------------------------- #
#  Corrected-mass helpers (REPLACE v1 fairing_oracle.areal_mass + physics.mass.total_mass)
# --------------------------------------------------------------------------- #
def _enrich_groups(design):
    """Attach the resonator cavity_volume (derived from f0 by the SAME physics the acoustics use); required by
    the corrected mass model and the feasibility check."""
    return [dict(g, cavity_volume=_group_cavity_volume(g)) for g in design["groups"]]


def _areal(design):
    """CORRECTED areal mass [kg/m^2] via the shared-wall model (2*sqrt(n) partitions + facesheet + backing)."""
    return float(MM.liner_areal_mass(_enrich_groups(design), design["D"], design["face_t"], design["wall_t"]))


def _total_mass(m_areal, f_cov, A_wall):
    """CORRECTED total installed mass [kg] (m_liner + named InstallBOM). f_cov=0 -> 0."""
    return float(MM.total_installed_mass(m_areal, f_cov, A_wall)["m_total"])


# --------------------------------------------------------------------------- #
#  Cheap per-design condition evaluator (a_band cached; robust worst)
# --------------------------------------------------------------------------- #
def _a_band(design):
    return np.asarray(liner_diffuse_alpha_band(design["groups"], design["D"]), float)


def _worst(a_band, spl, f_cov, A_wall, kappa):
    return robust_delta_oaspl(a_band, spl, f_cov, A_wall, kappa, DELTA_KAPPA)["worst"]


def _best_coverage(a_band, spl, A_wall, kappa):
    """Coverage that maximizes robust worst dOASPL, INCLUDING the f_cov=0 boundary (no treatment = exactly
    0 dB, strictly better than any harmful positive coverage)."""
    fs = np.linspace(0.0, 1.0, 26)
    vals = np.array([_worst(a_band, spl, f, A_wall, kappa) for f in fs])
    j = int(vals.argmax())
    if fs[j] <= 0.0:
        return 0.0, float(vals[j])
    a, b = fs[max(0, j - 1)], fs[min(len(fs) - 1, j + 1)]
    fstar, vstar = _golden_max(lambda f: _worst(a_band, spl, f, A_wall, kappa), a, b)
    if vstar < vals[j]:
        fstar, vstar = float(fs[j]), float(vals[j])
    return float(fstar), float(vstar)


def _refine_min_coverage(a_band, spl, A_wall, kappa, wv, cov_grid, target, iters=24):
    """Min coverage whose robust worst dOASPL >= target, given a PRECOMPUTED robust curve `wv` on `cov_grid`:
    bracket on the grid, then bisection-REFINE inside the bracket. Accurate (not grid-quantized) and cheap
    (the expensive curve is computed once per geom and shared across all targets)."""
    idx = np.nonzero(np.asarray(wv) >= target)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    lo = float(cov_grid[i - 1]) if i > 0 else 0.0
    hi = float(cov_grid[i])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _worst(a_band, spl, mid, A_wall, kappa) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _min_coverage(a_band, spl, A_wall, kappa, target):
    fs = np.linspace(0.005, 1.0, 60)
    vals = np.array([_worst(a_band, spl, f, A_wall, kappa) for f in fs])
    ok = np.nonzero(vals >= target)[0]
    if ok.size == 0:
        return None
    lo = fs[ok[0] - 1] if ok[0] > 0 else 0.0
    hi = fs[ok[0]]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _worst(a_band, spl, mid, A_wall, kappa) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)
_COV_GRID_N = 25                # coverage bracketing-grid resolution for the Pareto min-coverage sweep
_MIN_COV_GRID = 0.02            # smallest coverage on the bracketing grid
_COV_REFINE_ITERS = 24          # bisection refine iterations inside the bracket (accurate min-coverage)
