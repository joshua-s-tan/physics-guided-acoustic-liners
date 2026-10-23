# -*- coding: utf-8 -*-
"""FEM boundary models: the CORRECTED acoustic source boundary (§4.3) and the explicit non-wall `A_other`
boundary (§4.2), plus the coverage <-> mesh-area <-> mass-area invariant (step 6).

Two defects in the historical v2 interior solver (`fairing_case/v2/fem_interior_v2.py`) are corrected here:

  §4.3 SOURCE BOUNDARY -- v1 (`fem_interior_nr.solve_interior`) assembles, on the base facets:
         radiation (Robin) bilinear   rad(u,v) = 1j*k * r * u*v          (rho c radiation, beta_base = 1)
         incident source (linear)     src(v)   = 2j*k*p0 * r * v
       v2 kept `src` but DROPPED `rad`, leaving the base rigid apart from the source -- changing the source
       impedance and modal response. The production source model is RESTORED to v1's (radiation + incident)
       and named `SOURCE_MODEL` so every FEM artifact records it. A fixed-velocity/Neumann source, if wanted,
       must be a NAMED separate option, never a silent replacement.

  §4.2 A_other BOUNDARY -- OWNER DECISION (2026-09-26): the FALCON scenario uses EQUIVALENT DISTRIBUTED
       (volumetric) damping for A_other (see `damping.py`), NOT an explicit payload boundary, because real
       Falcon payload/structure geometry and absorbing area are mission-dependent and unavailable -- no
       `S_other` is invented. The explicit-boundary admittance here (`other_boundary_admittance`, with the
       `A_other/S_other <= 0.95` guard) is therefore reserved for the GENERIC reference-cavity benchmark, whose
       fully-specified internal boundary lets us compare explicit-boundary vs volumetric damping and treat
       their difference as model-form uncertainty. It must NOT be used to fabricate a Falcon payload boundary.

The coefficient helpers take `k` (not `f`), so they are independent of the exact sound speed and cancel in a
v1-vs-v2 continuity comparison. `assert_coverage_area_consistency` enforces the step-6 invariant.
"""
from __future__ import annotations
import math
from typing import Callable, Optional

# Named, artifact-recorded production models (review §4.3: "record the source model in every FEM artifact").
SOURCE_MODEL = "v1_base_radiation_plus_incident"          # rad Robin (beta_base=1) + incident 2j k p0
A_OTHER_MODEL = "explicit_other_boundary_admittance"      # Robin on S_other; NOT volumetric loss
MAX_ALPHA_OTHER = 0.95                                     # A_other/S_other ceiling (§4.2)
DEFAULT_COVERAGE_TOL = 0.02                                # relative tolerance for the coverage/area invariant


class FEMBoundaryError(ValueError):
    """A boundary-model input is invalid (nonpositive S_other, over-unity alpha_other, bad coverage/areas)."""


# --------------------------------------------------------------------------- #
#  §4.3 restored source boundary (base): radiation Robin + incident source
# --------------------------------------------------------------------------- #
def base_radiation_robin_coeff(k: float) -> complex:
    """Coefficient of the base RADIATION Robin bilinear term `coeff * r * u*v` (per the axisym r-weighted
    weak form). Restored to v1: `1j*k` (a rho c / anechoic base radiation, normalized admittance beta=1).
    v2 had DROPPED this term."""
    return 1j * float(k)


def incident_source_coeff(k: float, p0: float = 1.0) -> complex:
    """Coefficient of the incident SOURCE linear term `coeff * r * v` on the base. Unchanged from v1: `2j*k*p0`."""
    return 2j * float(k) * float(p0)


def base_admittance_normalized() -> float:
    """Normalized base radiation admittance beta_base = rho c * Y_base implied by the v1 radiation term = 1.0
    (an anechoic / rho c base). Exposed so the FEM artifact can record the source impedance explicitly."""
    return 1.0


# --------------------------------------------------------------------------- #
#  §4.2 explicit non-wall A_other boundary
# --------------------------------------------------------------------------- #
def alpha_other(A_other: float, S_other: float) -> float:
    """Diffuse absorption assigned to the designated non-wall boundary so its Sabine absorption over its own
    area reproduces A_other = kappa * A_wall:  alpha_other = A_other / S_other. Enforces the §4.2 ceiling:
    A_other/S_other <= 0.95 (else S_other is too small to carry that absorption -- enlarge it)."""
    A = float(A_other)
    S = float(S_other)
    if not (math.isfinite(A) and A >= 0):
        raise FEMBoundaryError(f"A_other must be finite and >= 0; got {A_other}")
    if not (math.isfinite(S) and S > 0):
        raise FEMBoundaryError(f"S_other must be finite and > 0 (designate a non-wall absorbing area); got {S_other}")
    a = A / S
    if a > MAX_ALPHA_OTHER:
        raise FEMBoundaryError(
            f"alpha_other = A_other/S_other = {a:.3f} > {MAX_ALPHA_OTHER}: designated S_other={S} too small "
            f"for A_other={A}; enlarge the represented non-wall boundary")
    return a


def _require_generic_benchmark(cavity):
    """Fail-closed guard: explicit-boundary A_other is a GENERIC-BENCHMARK-ONLY mode (§4.2). The object must
    be an ACTUAL validated `ReferenceCavity` (not a duck-typed look-alike), which by construction has
    `scenario_kind=='generic_boundary_benchmark'`, `is_vehicle==False`, derived `S_other`, and a canonical
    `geometry_hash`. A vehicle scenario (e.g. Falcon) is REJECTED -- it must use volumetric damping."""
    from .verification.reference_cavity import ReferenceCavity  # local import avoids any import-order coupling
    if not isinstance(cavity, ReferenceCavity):
        raise FEMBoundaryError(
            f"explicit A_other boundary requires a validated ReferenceCavity; got {type(cavity).__name__} "
            f"(a duck-typed look-alike is not accepted; a vehicle must use volumetric damping)")
    # ReferenceCavity.__post_init__ already guarantees kind/is_vehicle; re-assert defensively.
    if cavity.scenario_kind != "generic_boundary_benchmark" or cavity.is_vehicle is not False:
        raise FEMBoundaryError("ReferenceCavity is not a valid generic benchmark")
    h = cavity.geometry_hash()
    if not (isinstance(h, str) and len(h) == 64):
        raise FEMBoundaryError("generic benchmark must have a valid 64-char canonical geometry_hash")


def _require_finite_positive(name, v):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0):
        raise FEMBoundaryError(f"{name} must be finite and > 0; got {v!r}")
    return float(v)


def other_boundary_admittance(freqs, A_other: float, cavity,
                              admittance_from_alpha: Callable[[float], complex],
                              Z_air: float = 415.0,
                              alpha_from_admittance: Optional[Callable[[complex], float]] = None,
                              roundtrip_tol: float = 1e-6, require_roundtrip: bool = False) -> list:
    """Explicit non-wall boundary admittance for a GENERIC reference-cavity benchmark ONLY (§4.2), machine-
    enforced: `cavity` must be a validated `ReferenceCavity`; a vehicle (Falcon) is rejected. `S_other` is
    DERIVED from the cavity; alpha_other = A_other/S_other (<=0.95 guard, via `cavity.alpha_other`).
    `admittance_from_alpha(alpha) -> Y` returns PHYSICAL specific admittance Y; the FEM Robin coefficient uses
    the DIMENSIONLESS beta = Z_air*Y (see `to_normalized_beta`). Every Y is validated finite + passive.

    ROUND-TRIP: `require_roundtrip=True` (the PRODUCTION generic-benchmark evidence path) makes the absorption
    round-trip MANDATORY -- `alpha_from_admittance` must be supplied and must reproduce alpha_other within
    `roundtrip_tol`. Production MUST pass the project's DIFFUSE-absorption forward/inverse mapping here (the
    normal-incidence mapping is a unit-test stand-in only)."""
    _require_generic_benchmark(cavity)
    Z_air = _require_finite_positive("Z_air", Z_air)
    if not (isinstance(roundtrip_tol, (int, float)) and math.isfinite(roundtrip_tol) and roundtrip_tol >= 0):
        raise FEMBoundaryError(f"roundtrip_tol must be finite and >= 0; got {roundtrip_tol}")
    fs = list(freqs)                                   # promises one result per VALID frequency
    if not fs:
        raise FEMBoundaryError("freqs must be a non-empty sequence")
    for f in fs:
        if not (isinstance(f, (int, float)) and math.isfinite(float(f)) and float(f) > 0):
            raise FEMBoundaryError(f"every frequency must be finite and > 0; got {f!r}")
    if require_roundtrip and alpha_from_admittance is None:
        raise FEMBoundaryError(
            "require_roundtrip=True demands the diffuse-absorption inverse `alpha_from_admittance`; it must "
            "not be omitted for production generic-benchmark evidence")
    a = cavity.alpha_other(A_other)                    # derived S_other + <=0.95 guard live in the cavity
    y = complex(admittance_from_alpha(a))
    assert_passive_admittance(y, Z_air)                # finite + passive
    if alpha_from_admittance is not None:
        a_rt = float(alpha_from_admittance(y))
        if not math.isfinite(a_rt):
            raise FEMBoundaryError(f"round-trip alpha is not finite: {a_rt}")
        if abs(a_rt - a) > roundtrip_tol:
            raise FEMBoundaryError(f"absorption round-trip failed: alpha_other={a} -> {a_rt} (tol {roundtrip_tol})")
    return [y for _ in fs]


def to_normalized_beta(Y, Z_air: float = 415.0) -> complex:
    """Convert PHYSICAL specific admittance Y to the DIMENSIONLESS Robin coefficient beta = Z_air * Y. The
    FEM Robin term is `1j*k*beta*u*v`; passing a physical Y where beta is expected is a ~Z_air (~415x) error.
    This is the single authoritative conversion (tested)."""
    Z_air = _require_finite_positive("Z_air", Z_air)
    return Z_air * complex(Y)


# --------------------------------------------------------------------------- #
#  step-6 invariant: coverage <-> mesh area <-> mass area
# --------------------------------------------------------------------------- #
def assert_coverage_area_consistency(requested_coverage: float, achieved_mesh_coverage: float,
                                     treated_mesh_area: float, mesh_wall_area: float,
                                     mass_treated_area: float, scenario_wall_area: float,
                                     tol: float = DEFAULT_COVERAGE_TOL) -> dict:
    """Enforce (step 6):

        requested coverage
        ~= achieved FEM facet coverage
         = treated FEM area / mesh wall area
        ~= mass-model treated area / scenario wall area

    Any mismatch beyond `tol` (relative) raises FEMBoundaryError -- the run is invalid. Returns the achieved
    ratios for the FEM artifact. Guards against the FEM and mass models silently describing different coverage."""
    for nm, v in (("requested_coverage", requested_coverage), ("achieved_mesh_coverage", achieved_mesh_coverage),
                  ("treated_mesh_area", treated_mesh_area), ("mesh_wall_area", mesh_wall_area),
                  ("mass_treated_area", mass_treated_area), ("scenario_wall_area", scenario_wall_area),
                  ("tol", tol)):
        if not math.isfinite(float(v)):
            raise FEMBoundaryError(f"{nm} must be finite; got {v}")
    if not (0.0 < tol < 1.0):
        raise FEMBoundaryError(f"tol must be in (0,1); got {tol}")
    if not (mesh_wall_area > 0 and scenario_wall_area > 0):
        raise FEMBoundaryError("mesh_wall_area and scenario_wall_area must be > 0")
    for nm, v in (("requested_coverage", requested_coverage), ("achieved_mesh_coverage", achieved_mesh_coverage)):
        if not (0.0 <= float(v) <= 1.0):
            raise FEMBoundaryError(f"{nm} must be in [0,1]; got {v}")
    if treated_mesh_area < 0 or mass_treated_area < 0:
        raise FEMBoundaryError("treated areas must be >= 0")

    ratio_mesh = treated_mesh_area / mesh_wall_area          # facet-area coverage the FEM actually applied
    ratio_mass = mass_treated_area / scenario_wall_area      # coverage the mass model actually charged

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-12)

    checks = {
        "achieved_vs_requested": _rel(achieved_mesh_coverage, requested_coverage),
        "meshcov_vs_achieved":   _rel(ratio_mesh, achieved_mesh_coverage),
        "masscov_vs_requested":  _rel(ratio_mass, requested_coverage),
        "masscov_vs_meshcov":    _rel(ratio_mass, ratio_mesh),
    }
    bad = {k: v for k, v in checks.items() if v > tol}
    if bad:
        raise FEMBoundaryError(
            f"coverage/area invariant violated beyond tol={tol}: {bad}; "
            f"ratios mesh={ratio_mesh:.4f} mass={ratio_mass:.4f} requested={requested_coverage:.4f}")
    return dict(ratio_mesh_coverage=ratio_mesh, ratio_mass_coverage=ratio_mass, checks=checks, tol=tol)


def assert_passive_admittance(Y, Z_air: float, tol: float = 1e-9) -> None:
    """Validate a PHYSICAL specific admittance Y is FINITE and PASSIVE (§4.2 item 10). Convention e^{+iwt}
    with Robin term +1j*k*beta*u*v, beta = Z_air*Y: a passive surface has Re(beta) >= 0 (absorbs/reflects,
    never generates). `Z_air` must be finite > 0 and `tol` finite >= 0. Accepts a scalar or an iterable."""
    Z_air = _require_finite_positive("Z_air", Z_air)
    if not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol >= 0):
        raise FEMBoundaryError(f"tol must be finite and >= 0; got {tol}")
    ys = Y if hasattr(Y, "__iter__") else [Y]
    for y in ys:
        yv = complex(y)
        if not (math.isfinite(yv.real) and math.isfinite(yv.imag)):
            raise FEMBoundaryError(f"admittance {y} is not finite")
        beta = Z_air * yv
        if beta.real < -tol:
            raise FEMBoundaryError(f"admittance {y} is ACTIVE (Re(beta)={beta.real:.3e} < 0): not passive")


def normal_incidence_alpha(zeta: complex) -> float:
    """Normal-incidence absorption alpha = 1 - |R|^2, R = (zeta - 1)/(zeta + 1), for a normalized specific
    impedance zeta. Used for an admittance absorption round-trip sanity check (§4.2 item 10)."""
    z = complex(zeta)
    R = (z - 1.0) / (z + 1.0)
    return 1.0 - abs(R) ** 2


def source_model_record(p0: float = 1.0) -> dict:
    """The SOURCE model block to embed in every FEM artifact (§4.3 'record the source model'). This records
    ONLY the source boundary -- it does NOT assert any A_other model, so it can never mislabel Falcon's
    volumetric damping. The A_other block is recorded separately: `damping.loss_factor_record` for Falcon
    (volumetric) or `explicit_a_other_record` for the generic-boundary benchmark."""
    return dict(source_model=SOURCE_MODEL, base_admittance_normalized=base_admittance_normalized(), p0=float(p0))


def explicit_a_other_record(A_other: float, cavity) -> dict:
    """The A_other block for a GENERIC-BOUNDARY-BENCHMARK artifact ONLY (explicit Robin boundary). Consumes a
    VALIDATED generic benchmark (fail-closed via `_require_generic_benchmark`), derives S_other from it, and
    records the canonical geometry hash for provenance. Falcon (vehicle) artifacts are rejected here -- they
    record `damping.loss_factor_record` (volumetric) instead."""
    _require_generic_benchmark(cavity)
    S_other = float(cavity.S_other)
    return dict(a_other_model=A_OTHER_MODEL, A_other=float(A_other), S_other=S_other,
                alpha_other=cavity.alpha_other(A_other), max_alpha_other=MAX_ALPHA_OTHER,
                scenario_kind=cavity.scenario_kind, geometry_hash=cavity.geometry_hash())  # mandatory, never None
