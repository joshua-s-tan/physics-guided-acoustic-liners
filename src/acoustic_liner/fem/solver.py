r"""Migrated production axisymmetric (m=0) interior FEM solver (remediation §4.3/§4.2; step 6 integration).

SELF-CONTAINED in the release package (the frozen axisym r-weighted forms are copied here, not imported from
the historical tree). Implements the CORRECTED production boundary model:

  * SOURCE boundary (base, z=0): restored v1 model -- radiation Robin `1j*k*r*u*v` (beta_base=1) PLUS incident
    source `2j*k*p0*r*v` (`fem_boundary.SOURCE_MODEL`). The historical v2 dropped the radiation term.
  * WALL (r>0, z>0): Robin `1j*k*beta*r*u*v`, beta = Z_air*Y (DIMENSIONLESS; `fem.boundary.to_normalized_beta`).
  * AXIS (r=0): symmetry -- natural BC, no term (its axisym area is ~0).
  * A_other: FALCON uses equivalent distributed volumetric damping `k^2 -> k^2(1 - i*eta)` (`damping.py`).
    The generic-benchmark explicit boundary is a separate path (`fem.boundary.other_boundary_admittance`).

Boundary facets are tagged into DISJOINT sets (source/wall/axis) with per-facet axisymmetric areas, so the
real-mesh facet-partition gate and the coverage<->mesh<->mass invariant run on actual integrated areas.
"""
from __future__ import annotations

import math
from typing import Callable, Optional
import numpy as np
from ..physics.constants import C_AIR  # single authoritative source
from ..physics.constants import RHO_AIR
from ..physics.constants import Z_AIR
from .errors import FEMSolverError
from .forms import _forms
from .mesh import assert_boundary_partition
from .mesh import build_fairing_interior_mesh
from .mesh import mesh_physical_volume
from .mesh import select_coverage_facets
from .mesh import tag_boundary_facets


# Coverage-consistency threshold, expressed as an ABSOLUTE coverage-fraction error |achieved - requested|
# (coverage is itself a fraction of the wall, so its discretization error is naturally absolute, not relative
# -- a relative metric spuriously blows up as requested coverage -> 0). This is a MESH-RESOLUTION ACCEPTANCE
# THRESHOLD established BEFORE any scientific-outcome runs (not an a-priori preregistration): chosen after
# PROBING the demonstrator mesh (mesh_size 0.35), where the largest wall facet is ~3.3% of the wall and the
# measured WORST absolute error over the FULL range coverages {0.1..0.9} x anchors {low,high,mid} was 0.0150.
# Requested design coverage stays AUTHORITATIVE; achieved facet-area coverage is recorded separately. A valid
# coverage exceeding this threshold must be handled by REFINING the mesh, never by raising the threshold
# (mesh_size 0.2 already lowers the worst to 0.0089).
COVERAGE_CONSISTENCY_TOL = 0.02
COVERAGE_TOL_PROVENANCE = dict(
    kind="mesh_resolution_acceptance_threshold_pre_outcome",
    metric="absolute_coverage_fraction_error", threshold=0.02,
    probed_mesh_size=0.35, max_wall_facet_fraction=0.0332, measured_worst_abs_error=0.0150,
    probed_coverages=tuple(round(0.1 * i, 1) for i in range(1, 10)), probed_anchors=("low", "high", "mid"),
    finer_mesh_worst_abs_error={0.2: 0.0089},
    rule="requested coverage authoritative; refine mesh (do not raise threshold) if a valid coverage exceeds it")


# Fixed production tolerances (named so the release verifier can recompute against the same values, not magic
# numbers). The FEM facet-partition gate runs at 4x the default fem-consistency tol (0.005 -> 0.02); the
# benchmark diffuse round-trip closes to 1e-6. A non-default consistency_tol still scales the runner's gate,
# but release artifacts are produced on the default path, which is what the verifier recomputes against.
FEM_CONSISTENCY_TOL_DEFAULT = 0.005
FEM_PARTITION_TOL = FEM_CONSISTENCY_TOL_DEFAULT * 4       # = 0.02 (default-path production value)


def _require_passive_beta(name: str, beta) -> complex:
    """A normalized wall Robin coefficient beta (= Z_air*Y) must be finite and PASSIVE (Re(beta) >= 0).
    beta == 0 (rigid) is allowed."""
    b = complex(beta)
    if not (math.isfinite(b.real) and math.isfinite(b.imag)):
        raise FEMSolverError(f"{name} must be finite; got {beta!r}")
    if b.real < -1e-9:
        raise FEMSolverError(f"{name} is ACTIVE (Re={b.real:.3e} < 0): not passive")
    return b


def solve_interior(mesh, freq: float, tagged: dict, wall_beta: complex,
                   A_other: float = 0.0, volume: Optional[float] = None, p0: float = 1.0,
                   treated_facets: Optional[set] = None, bare_beta: Optional[complex] = None,
                   base_radiation: bool = True, extra_robin: Optional[list] = None) -> float:
    """One-frequency axisymmetric interior r-weighted mean-square pressure with the CORRECTED production
    boundary model. Source = base radiation Robin (1j*k) + incident (2j*k*p0). Wall Robin uses `wall_beta`
    (dimensionless beta = Z_air*Y); if `treated_facets`+`bare_beta` are given, treated wall facets use
    `wall_beta` and the rest use `bare_beta` (partial coverage). A_other>0 applies volumetric damping
    k^2 -> k^2(1 - i*eta) with eta from `damping.volumetric_loss_factor` (requires `volume`).
    `base_radiation=False` reproduces the HISTORICAL DEFECTIVE model (radiation term dropped) for the
    continuity contrast test -- production always uses True."""
    if not (math.isfinite(freq) and freq > 0):
        raise FEMSolverError(f"freq must be finite and > 0; got {freq}")
    if not (isinstance(p0, (int, float)) and math.isfinite(float(p0))):
        raise FEMSolverError(f"p0 must be finite; got {p0!r}")
    if not isinstance(base_radiation, bool):
        raise FEMSolverError(f"base_radiation must be a genuine bool; got {type(base_radiation).__name__}")
    wall_beta = _require_passive_beta("wall_beta", wall_beta)
    skfem, BilinearForm, LinearForm, FacetBasis, spsolve, axisym_stiffness, axisym_mass = _forms()
    k = 2.0 * math.pi * freq / C_AIR

    # A_other: reject negative/non-finite (no silent zero); 0 is allowed (no damping).
    A = 0.0 if A_other is None else float(A_other)
    if not (math.isfinite(A) and A >= 0):
        raise FEMSolverError(f"A_other must be finite and >= 0; got {A_other!r}")
    eta = 0.0
    if A > 0:
        from . import damping
        if volume is None or not (math.isfinite(float(volume)) and float(volume) > 0):
            raise FEMSolverError(f"A_other>0 requires a finite positive cavity volume; got {volume!r}")
        eta = damping.volumetric_loss_factor(A, float(volume), freq)
    kc2 = (k ** 2) * (1.0 - 1j * eta)

    el = skfem.ElementTriP1()
    basis = skfem.Basis(mesh, el)
    A = axisym_stiffness.assemble(basis).astype(complex) - kc2 * axisym_mass.assemble(basis).astype(complex)

    sets = tagged["sets"]
    base = np.array(sorted(sets["source"]), dtype=int)
    wall_all = sets["wall"]

    def _robin(facets, beta):
        if beta == 0 or len(facets) == 0:
            return
        fb = FacetBasis(mesh, el, facets=np.array(sorted(facets), dtype=int))
        @BilinearForm(dtype=complex)
        def robin(u, v, w):
            return 1j * k * beta * w.x[0] * u * v
        nonlocal A
        A = A + robin.assemble(fb)

    # base radiation (beta_base = 1) -- restored v1 term (dropped by the historical defective model)
    fb_base = FacetBasis(mesh, el, facets=base)
    if base_radiation:
        @BilinearForm(dtype=complex)
        def rad(u, v, w):
            return 1j * k * w.x[0] * u * v
        A = A + rad.assemble(fb_base)

    # wall Robin: full coverage (wall_beta everywhere) or partial (treated vs bare)
    if treated_facets is None:
        _robin(wall_all, wall_beta)
    else:
        treated = set(treated_facets)
        foreign = treated - wall_all
        if foreign:
            raise FEMSolverError(f"treated_facets contains non-wall facets {sorted(foreign)[:5]}")
        untreated = wall_all - treated
        if untreated and bare_beta is None:
            raise FEMSolverError("partial coverage leaves wall facets untreated but bare_beta was not given "
                                 "(silently rigid is not allowed)")
        _robin(treated, wall_beta)
        if untreated:
            _robin(untreated, _require_passive_beta("bare_beta", bare_beta))

    # extra explicit Robin boundaries (e.g. the generic-benchmark payload boundary): list of (facets, beta)
    if extra_robin:
        for facets, beta in extra_robin:
            _robin(set(facets), _require_passive_beta("extra_robin_beta", beta))

    # incident source on the base
    @LinearForm(dtype=complex)
    def src(v, w):
        return 2j * k * p0 * w.x[0] * v
    b = src.assemble(fb_base)

    M = axisym_mass.assemble(basis)
    p = spsolve(A.tocsc(), b)
    if not np.all(np.isfinite(p)):
        raise FEMSolverError(f"solver produced a non-finite field at {freq} Hz")
    ms = float(np.real(np.vdot(p, M @ p)))
    vol = float(np.real(np.ones(len(p)) @ (M @ np.ones(len(p)))))
    if not (math.isfinite(ms) and ms >= 0 and math.isfinite(vol) and vol > 0):
        raise FEMSolverError(f"non-finite/nonpositive mean-square ({ms}) or measure ({vol}) at {freq} Hz")
    return ms / vol


def run_scenario_fem(scenario, freqs, wall_beta, *, kappa: Optional[float] = None, coverage: float = 1.0,
                     bare_beta: Optional[complex] = None, p0: float = 1.0, mesh_size: float = 0.35,
                     anchor: str = "low", consistency_tol: float = FEM_CONSISTENCY_TOL_DEFAULT) -> dict:
    """SCENARIO-BOUND production runner (the ONLY path that should generate release FEM artifacts).

        scenario -> mesh(scenario.geometry) -> measured wall area + physical volume (2*pi*int r dA)
        -> assert_fem_consistency(measured area/volume/kappa) -> A_other = scenario.A_other(kappa)
        -> validated boundary coefficients -> solve(volume = MEASURED mesh volume) -> finite evidence record.

    `wall_beta` may be a complex scalar or a callable freq->beta (dimensionless beta = Z_air*Y). Falcon uses
    volumetric A_other damping (via the MEASURED volume, never a free scalar), which is why the original
    geometry mismatch cannot reappear through a scalar input."""
    from . import scenarios as S
    from . import damping
    from . import boundary as fem_boundary
    from ..evaluation import requests as R
    # require an ACTUAL PhysicalScenario at the boundary (not a duck-typed look-alike)
    if not isinstance(scenario, S.PhysicalScenario):
        raise FEMSolverError(f"run_scenario_fem requires a PhysicalScenario; got {type(scenario).__name__}")
    # validate coverage/anchor UNCONDITIONALLY (before any branch): NaN/>1 must NOT silently become full cov.
    if not (isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
            and math.isfinite(float(coverage)) and 0.0 <= float(coverage) <= 1.0):
        raise FEMSolverError(f"coverage must be a finite number in [0,1]; got {coverage!r}")
    coverage = float(coverage)
    if anchor not in ("low", "high", "mid"):
        raise FEMSolverError(f"anchor must be one of low|high|mid; got {anchor!r}")
    # materialize + validate the frequency sequence ONCE (a generator cannot be exhausted downstream)
    fs = [float(f) for f in freqs]
    if not fs:
        raise FEMSolverError("freqs must be a non-empty sequence")
    for f in fs:
        if not (math.isfinite(f) and f > 0):
            raise FEMSolverError(f"every frequency must be finite and > 0; got {f}")

    g = scenario.geometry
    mesh = build_fairing_interior_mesh(g.R_fair, g.H_barrel, g.H_total, mesh_size=mesh_size)
    tagged = tag_boundary_facets(mesh)
    mesh_wall_area = sum(tagged["area_by_id"][f] for f in tagged["sets"]["wall"])
    mesh_volume = mesh_physical_volume(mesh)
    # the artifact-producing path itself RUNS the boundary-partition gate (not just a separate test)
    partition = assert_boundary_partition(
        tagged, {"source": math.pi * g.R_fair ** 2, "wall": scenario.A_wall, "axis": 0.0}, tol=consistency_tol * 4)
    k = float(scenario.kappa_nominal if kappa is None else kappa)
    A_other = scenario.A_other(k)
    # SCENARIO CONSISTENCY on the REAL mesh-integrated area AND physical volume (fail-closed, survives -O).
    achieved = S.assert_fem_consistency(scenario, mesh_wall_area, mesh_volume, A_other, expected_kappa=k,
                                        tol=consistency_tol)
    # Determine treated facets + achieved coverage for EVERY case (full and zero included), and ALWAYS gate.
    if coverage >= 1.0:
        treated = None                                   # full: wall_beta everywhere in the solve
        treated_mesh_area = mesh_wall_area
    elif coverage <= 0.0:
        treated = set()                                  # zero: all wall bare (needs bare_beta)
        treated_mesh_area = 0.0
        if bare_beta is None:
            raise FEMSolverError("zero coverage makes the whole wall bare and requires bare_beta")
    else:
        treated = select_coverage_facets(mesh, tagged, coverage, anchor=anchor)
        if bare_beta is None:
            raise FEMSolverError("partial coverage requires bare_beta")
        treated_mesh_area = sum(tagged["area_by_id"][f] for f in treated)
    achieved_mesh_coverage = treated_mesh_area / mesh_wall_area
    # coverage<->mesh-area<->mass-area invariant, gated for ALL coverages via the ABSOLUTE coverage-fraction
    # error. REQUESTED coverage stays authoritative: mass treated area uses REQUESTED coverage (never
    # redefined to the selected facets); achieved mesh coverage is recorded separately; the gate FAILS (never
    # raises the threshold) beyond it -- the remedy is a finer mesh.
    mass_treated_area = coverage * scenario.A_wall
    abs_cov_err = abs(achieved_mesh_coverage - coverage)
    area_gap = abs(treated_mesh_area - mass_treated_area)
    passed = abs_cov_err <= COVERAGE_CONSISTENCY_TOL
    if not passed:
        raise FEMSolverError(
            f"coverage/area invariant violated: requested {coverage} vs achieved {achieved_mesh_coverage:.4f} "
            f"(abs err {abs_cov_err:.4f} > {COVERAGE_CONSISTENCY_TOL}); REFINE the mesh (do not raise the "
            f"threshold). Try a smaller mesh_size.")
    coverage_consistency = dict(
        requested_coverage=coverage, achieved_mesh_coverage=achieved_mesh_coverage,
        treated_mesh_area=treated_mesh_area, mass_treated_area=mass_treated_area,
        mesh_wall_area=mesh_wall_area, abs_coverage_error=abs_cov_err, area_gap_m2=area_gap,
        tol=COVERAGE_CONSISTENCY_TOL, passed=True, tol_provenance=COVERAGE_TOL_PROVENANCE)

    def _cj(z):
        return None if z is None else dict(real=float(complex(z).real), imag=float(complex(z).imag))

    ms_by_freq, wall_beta_by_freq = {}, {}
    for f in fs:
        beta = complex(wall_beta(f) if callable(wall_beta) else wall_beta)
        wall_beta_by_freq[f] = _cj(beta)                    # store the ACTUAL applied beta per frequency
        ms_by_freq[f] = solve_interior(mesh, f, tagged, wall_beta=beta, A_other=A_other,
                                       volume=mesh_volume, p0=p0, treated_facets=treated, bare_beta=bare_beta)
    return dict(
        scenario_name=scenario.name, scenario_sha256=R.canonical_scenario_sha256(scenario),
        measured_wall_area=mesh_wall_area, measured_volume=mesh_volume, kappa=k, A_other=A_other,
        coverage=coverage, anchor=anchor, freqs=fs, consistency=achieved, boundary_partition=partition,
        coverage_consistency=coverage_consistency,
        source_model=fem_boundary.source_model_record(p0=p0),
        damping_model=damping.loss_factor_record(A_other, mesh_volume, k, fs),
        boundary_coeffs=dict(wall_beta_by_freq=wall_beta_by_freq, wall_beta_is_callable=callable(wall_beta),
                             bare_beta=_cj(bare_beta)),
        ms_by_freq=ms_by_freq)
